"""
cluster_scaler.py
=================
K-Means Cluster Scaler for QAOA-based VRP.

MOTIVATION (from Azad et al., 2023):
    Qubit requirement for VRP(n, k) scales as N = n x (n-1).
    At n=10 nodes: N=90 qubits - far beyond any NISQ simulator.
    At n=4  nodes: N=12 qubits - exactly what the paper simulates.

STRATEGY:
    1. Generate 50 random city coordinates (the full problem).
    2. Use K-Means + recursive splitting to enforce <= MAX_CLUSTER nodes
       per cluster, guaranteeing NISQ compatibility.
    3. Each cluster is solved independently by QAOA.
    4. Stitch clusters into a global route via nearest-neighbour
       on cluster centroids (inter-cluster heuristic).

OUTPUT:
    - cluster_map.png        : Visual map of all clusters
    - cluster_summary.csv    : Per-cluster qubit cost and node list
    - distance_matrices/     : One .npy distance matrix per cluster
                               (ready to feed into QAOA circuit)
"""
import sys, os
# Ensure project root is on path and cwd is project root
_ROOT = os.path.dirname(os.path.dirname(os.path.abspath(__file__)))
if _ROOT not in sys.path:
    sys.path.insert(0, os.path.join(_ROOT, "src"))
os.chdir(_ROOT)


import numpy as np
import matplotlib.pyplot as plt
import matplotlib.cm as cm
import pandas as pd
import os
from sklearn.cluster import KMeans
from scipy.spatial.distance import cdist

# ── Configuration ──────────────────────────────────────────────────────────────
RANDOM_SEED  = 42
N_CITIES     = 50
MAX_CLUSTER  = 4       # hard ceiling: 4 delivery nodes + depot = 5 nodes = 20 qubits
GRID_SIZE    = 100

np.random.seed(RANDOM_SEED)

# ── Step 1: Generate City Coordinates ─────────────────────────────────────────
print("=" * 62)
print("  CLUSTER SCALER FOR NISQ-COMPATIBLE VRP")
print("=" * 62)

coords = np.random.uniform(0, GRID_SIZE, size=(N_CITIES, 2))
depot  = coords[0]

print(f"\n[1] Generated {N_CITIES} city coordinates.")
print(f"    Depot (index 0) at: ({depot[0]:.2f}, {depot[1]:.2f})")

# ── Step 2: Size-Constrained Clustering via Recursive Splitting ────────────────
# Standard K-Means does not enforce equal cluster sizes.
# Fix: after initial clustering, recursively split any cluster
# that exceeds MAX_CLUSTER nodes into two sub-clusters.

def split_until_small(node_indices, node_coords, max_size, seed=42):
    """
    Recursively split a cluster using K-Means (k=2) until
    every sub-cluster has <= max_size nodes.
    Returns list of (indices_array, coords_array) tuples.
    """
    if len(node_indices) <= max_size:
        return [(node_indices, node_coords)]
    km = KMeans(n_clusters=2, random_state=seed, n_init=10)
    sub_labels = km.fit_predict(node_coords)
    result = []
    for sub_id in range(2):
        mask = sub_labels == sub_id
        result.extend(
            split_until_small(node_indices[mask], node_coords[mask], max_size, seed)
        )
    return result

delivery_coords  = coords[1:]
delivery_indices = np.arange(1, N_CITIES)

# Initial K-Means with enough clusters to approach target size
n_init = int(np.ceil(N_CITIES / MAX_CLUSTER))
km_init     = KMeans(n_clusters=n_init, random_state=RANDOM_SEED, n_init=10)
init_labels = km_init.fit_predict(delivery_coords)

final_clusters = []
for cid in range(n_init):
    mask = init_labels == cid
    final_clusters.extend(
        split_until_small(delivery_indices[mask], delivery_coords[mask], MAX_CLUSTER)
    )

N_CLUSTERS = len(final_clusters)
print(f"\n[2] Size-constrained K-Means produced {N_CLUSTERS} clusters "
      f"(all <= {MAX_CLUSTER} delivery nodes).")

# ── Step 3: Validate Qubit Cost and Save Distance Matrices ────────────────────
print("\n[3] Qubit cost validation (N = n x (n-1), depot counted as node 0):\n")
print(f"    {'Cluster':<10} {'Delivery':<10} {'Total+depot':<14} "
      f"{'Qubits':<10} {'NISQ-Safe?'}")
print("    " + "-" * 58)

os.makedirs("distance_matrices", exist_ok=True)
cluster_data  = []
centroid_list = []
all_safe      = True

for cid, (node_indices, cluster_coords) in enumerate(final_clusters):
    sub_coords = np.vstack([depot, cluster_coords])   # inject depot
    n_nodes    = len(sub_coords)
    n_qubits   = n_nodes * (n_nodes - 1)
    safe       = n_qubits <= 20
    if not safe:
        all_safe = False
    label = "YES" if safe else "NO"

    dist_matrix = cdist(sub_coords, sub_coords, metric='euclidean')
    np.save(f"distance_matrices/cluster_{cid:02d}.npy", dist_matrix)

    centroid = cluster_coords.mean(axis=0)
    centroid_list.append(centroid)

    print(f"    Cluster {cid:<4} {len(node_indices):<10} {n_nodes:<14} "
          f"{n_qubits:<10} {'OK' if safe else 'EXCEEDS'}")

    cluster_data.append({
        "cluster_id"       : cid,
        "n_delivery_nodes" : len(node_indices),
        "n_total_nodes"    : n_nodes,
        "n_qubits"         : n_qubits,
        "nisq_safe"        : label,
        "node_indices"     : list(map(int, node_indices)),
        "centroid_x"       : round(centroid[0], 3),
        "centroid_y"       : round(centroid[1], 3),
    })

centroid_array = np.array(centroid_list)
df = pd.DataFrame(cluster_data)

# ── Step 4: Inter-Cluster Stitching ───────────────────────────────────────────
# After QAOA solves each cluster independently, we need a global ordering
# of clusters. We apply nearest-neighbour starting from the cluster whose
# centroid is closest to the depot.
print("\n[4] Inter-cluster stitching (nearest-neighbour on centroids)...\n")

visited         = [False] * N_CLUSTERS
global_sequence = []
dist_to_depot   = np.linalg.norm(centroid_array - depot, axis=1)
current         = int(np.argmin(dist_to_depot))

for _ in range(N_CLUSTERS):
    visited[current] = True
    global_sequence.append(current)
    dists = np.linalg.norm(centroid_array - centroid_array[current], axis=1)
    dists[[i for i, v in enumerate(visited) if v]] = np.inf
    if np.all(np.isinf(dists)):
        break
    current = int(np.argmin(dists))

print(f"    Global cluster visit order: {global_sequence}")

# ── Step 5: Save CSV ──────────────────────────────────────────────────────────
df.to_csv("cluster_summary.csv", index=False)
print(f"\n[5] Saved cluster_summary.csv")
print(f"    Saved {N_CLUSTERS} distance matrices in distance_matrices/")

# ── Step 6: Visualisation ─────────────────────────────────────────────────────
colors = cm.tab20(np.linspace(0, 1, min(N_CLUSTERS, 20)))

fig, axes = plt.subplots(1, 2, figsize=(17, 7))
fig.suptitle(
    "K-Means Cluster Scaler  —  NISQ-Compatible VRP Partitioning\n"
    f"({N_CITIES} cities  →  {N_CLUSTERS} clusters, "
    f"max {MAX_CLUSTER} delivery nodes each, max 20 qubits per QAOA call)",
    fontsize=13, fontweight='bold'
)

# -- Left: City map with clusters
ax = axes[0]
for cid, (node_indices, cluster_coords) in enumerate(final_clusters):
    c = colors[cid % 20]
    ax.scatter(cluster_coords[:, 0], cluster_coords[:, 1],
               color=c, s=55, zorder=3)
    ax.annotate(f"C{cid}", centroid_list[cid] + np.array([0, 1.5]),
                fontsize=6.5, ha='center', va='bottom',
                color=c, fontweight='bold')

# Stitching path between centroids
seq_c = centroid_array[global_sequence]
ax.plot(seq_c[:, 0], seq_c[:, 1], 'k--', lw=1.0, alpha=0.4, label='Stitch path')
ax.plot([depot[0], seq_c[0, 0]], [depot[1], seq_c[0, 1]],
        'k--', lw=1.0, alpha=0.4)

for step, cid in enumerate(global_sequence):
    ax.annotate(str(step + 1), centroid_list[cid] + np.array([1.8, -2.5]),
                fontsize=5.5, color='grey')

ax.scatter(*depot, color='crimson', marker='D', s=140, zorder=5,
           label='Depot', edgecolors='black')
ax.set_title("City Map — Clusters and Global Stitching Order")
ax.set_xlabel("X Coordinate")
ax.set_ylabel("Y Coordinate")
ax.legend(fontsize=9)
ax.set_xlim(-5, 105)
ax.set_ylim(-5, 105)
ax.grid(True, alpha=0.3)

# -- Right: Qubit cost per cluster
ax2 = axes[1]
qubit_counts = df["n_qubits"].values
bar_colors   = ['#2ecc71' if q <= 20 else '#e74c3c' for q in qubit_counts]
bars = ax2.bar(range(N_CLUSTERS), qubit_counts, color=bar_colors,
               edgecolor='black', linewidth=0.6)
ax2.axhline(y=20, color='darkorange', linestyle='--', linewidth=2,
            label='NISQ Limit (20 qubits)')
ax2.axhline(y=12, color='steelblue', linestyle=':', linewidth=1.8,
            label='Paper baseline VRP(4,2): 12 qubits')
ax2.set_xlabel("Cluster ID")
ax2.set_ylabel("Qubits Required  [ N = n x (n-1) ]")
ax2.set_title("Per-Cluster Qubit Cost\n(Green = NISQ-Safe   Red = Exceeds limit)")
ax2.set_xticks(range(N_CLUSTERS))
ax2.tick_params(axis='x', labelsize=7)
ax2.legend(fontsize=9)
ax2.set_ylim(0, max(qubit_counts) + 4)
ax2.grid(True, axis='y', alpha=0.3)
for bar, q in zip(bars, qubit_counts):
    ax2.text(bar.get_x() + bar.get_width() / 2, bar.get_height() + 0.2,
             str(q), ha='center', va='bottom', fontsize=7.5, fontweight='bold')

plt.tight_layout()
plt.savefig("cluster_map.png", dpi=150, bbox_inches='tight')
plt.close()
print("\n[6] Saved cluster_map.png")

# ── Final Summary ──────────────────────────────────────────────────────────────
print("\n" + "=" * 62)
print("  FINAL SUMMARY")
print("=" * 62)
print(f"  Total cities              : {N_CITIES}")
print(f"  Clusters formed           : {N_CLUSTERS}")
print(f"  Max delivery nodes/cluster: {MAX_CLUSTER}")
print(f"  Max qubits per cluster    : {int(qubit_counts.max())}")
print(f"  All clusters NISQ-safe    : {'YES' if all_safe else 'CHECK RED BARS'}")
print(f"  Distance matrices         : distance_matrices/cluster_XX.npy")
print(f"  Cluster summary CSV       : cluster_summary.csv")
print(f"  Cluster visual            : cluster_map.png")
print("=" * 62)
print()
print("  NEXT STEP:")
print("  Feed distance_matrices/cluster_XX.npy as the D matrix into")
print("  your QAOA circuit (Shisheer or SECQUOIA QUBO formulation).")
print("  Use global_sequence to stitch inter-cluster routes.")
print()
