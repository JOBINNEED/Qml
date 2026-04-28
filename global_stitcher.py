"""
global_stitcher.py
==================
Step 5 of the Hybrid Quantum-Classical VRP Pipeline.

PURPOSE:
    Takes the per-cluster QAOA solutions and assembles them into a single
    coherent global delivery route across all 50 cities.

ALGORITHM — Two-level nearest-neighbour stitching:
    Level 1 (intra-cluster):
        Decode each QAOA bitstring (position encoding x[i][t]) into an
        ordered visit sequence for the cluster's nodes.

    Level 2 (inter-cluster):
        Order the clusters themselves using a nearest-neighbour heuristic
        on cluster centroids, starting from the cluster closest to the depot.
        This matches the ordering already computed by cluster_scaler.py.

    Final route:
        depot → cluster_A[0..k] → depot → cluster_B[0..k] → depot → ...
        Each cluster sub-route starts and ends at the depot (multi-vehicle
        interpretation: one vehicle per cluster).

OUTPUTS:
    outputs/global_route.png          — full 50-city route visualisation
    outputs/global_route_results.json — route data and total cost
"""

import numpy as np
import matplotlib.pyplot as plt
import matplotlib.cm as cm
import pandas as pd
import json
import os


# ══════════════════════════════════════════════════════════════════════════════
# Bitstring → ordered route decoder
# ══════════════════════════════════════════════════════════════════════════════

def decode_bitstring(bitstring: str, n_nodes: int) -> list:
    """
    Decode a QAOA position-encoded bitstring into a city visit order.

    The position encoding uses n² bits arranged as a flat n×n matrix:
        bit[i*n + t] = x[i][t]  (city i at time step t)

    We read off the time-step assignment for each city and sort by time.

    Parameters
    ----------
    bitstring : str
        Binary string of length n².
    n_nodes : int
        Number of nodes in the cluster (including depot at index 0).

    Returns
    -------
    list[int]
        Ordered list of node indices (0 = depot).
        Falls back to [0, 1, 2, ..., n-1] if the bitstring is invalid.
    """
    n = n_nodes
    expected_len = n * n

    # Pad or truncate to expected length
    bs = bitstring.ljust(expected_len, '0')[:expected_len]

    try:
        matrix = np.array([int(b) for b in bs], dtype=int).reshape(n, n)
    except Exception:
        return list(range(n))

    # For each city i, find which time step t has x[i][t] = 1
    assignment = {}
    for i in range(n):
        active_times = np.where(matrix[i] == 1)[0]
        if len(active_times) == 1:
            assignment[i] = int(active_times[0])
        else:
            # Ambiguous or empty row — assign a default time
            assignment[i] = i

    # Sort cities by their assigned time step
    ordered = sorted(assignment.keys(), key=lambda city: assignment[city])
    return ordered


# ══════════════════════════════════════════════════════════════════════════════
# Intra-cluster route cost
# ══════════════════════════════════════════════════════════════════════════════

def route_cost(ordered_nodes: list, dist_matrix: np.ndarray) -> float:
    """
    Compute the total travel distance for an ordered node sequence,
    starting and ending at the depot (index 0 in the local matrix).
    """
    cost = 0.0
    route = [0] + [n for n in ordered_nodes if n != 0] + [0]
    for k in range(len(route) - 1):
        i, j = route[k], route[k + 1]
        if i < dist_matrix.shape[0] and j < dist_matrix.shape[0]:
            cost += dist_matrix[i, j]
    return cost


# ══════════════════════════════════════════════════════════════════════════════
# Main stitcher
# ══════════════════════════════════════════════════════════════════════════════

def stitch_global_route(cluster_results: list,
                        cluster_summary_path: str = "cluster_summary.csv",
                        coords: np.ndarray = None,
                        output_dir: str = "outputs") -> dict:
    """
    Assemble per-cluster QAOA solutions into a global 50-city route.

    Parameters
    ----------
    cluster_results : list[dict]
        Output from qaoa_solver.solve_cluster() for each cluster.
        Each dict must contain: cluster_id, best_bitstring, n_nodes.
    cluster_summary_path : str
        Path to cluster_summary.csv from cluster_scaler.py.
    coords : np.ndarray, shape (N_CITIES, 2)
        Global city coordinates (index 0 = depot).
    output_dir : str
        Directory for output files.

    Returns
    -------
    dict with keys:
        global_route, total_cost, cluster_routes, inter_cluster_order
    """
    os.makedirs(output_dir, exist_ok=True)

    # ── Load cluster metadata ─────────────────────────────────────────────────
    df_summary = pd.read_csv(cluster_summary_path)
    depot_coords = coords[0] if coords is not None else np.array([0.0, 0.0])

    # Build centroid array for inter-cluster ordering
    centroids = df_summary[["centroid_x", "centroid_y"]].values
    n_clusters = len(df_summary)

    # ── Inter-cluster ordering (nearest-neighbour on centroids) ───────────────
    visited = [False] * n_clusters
    dist_to_depot = np.linalg.norm(centroids - depot_coords, axis=1)
    current = int(np.argmin(dist_to_depot))
    inter_order = []

    for _ in range(n_clusters):
        visited[current] = True
        inter_order.append(current)
        dists = np.linalg.norm(centroids - centroids[current], axis=1)
        dists[[i for i, v in enumerate(visited) if v]] = np.inf
        if np.all(np.isinf(dists)):
            break
        current = int(np.argmin(dists))

    print(f"\n  Inter-cluster visit order: {inter_order}")

    # ── Decode each cluster's QAOA bitstring ──────────────────────────────────
    cluster_routes = {}   # cluster_id → local ordered node indices
    total_cost = 0.0

    # Build a lookup from cluster_id → result dict
    result_lookup = {r["cluster_id"]: r for r in cluster_results}

    for cid in inter_order:
        row = df_summary[df_summary["cluster_id"] == cid].iloc[0]
        node_indices = json.loads(row["node_indices"])   # global city indices

        if cid in result_lookup:
            res = result_lookup[cid]
            local_order = decode_bitstring(res["best_bitstring"], res["n_nodes"])
            # Map local indices back to global city indices
            # local index 0 = depot, local index k = node_indices[k-1]
            global_order = []
            for local_idx in local_order:
                if local_idx == 0:
                    global_order.append(0)   # depot
                elif local_idx - 1 < len(node_indices):
                    global_order.append(node_indices[local_idx - 1])

            # Load distance matrix for cost calculation
            dm_path = f"distance_matrices/cluster_{cid:02d}.npy"
            if os.path.exists(dm_path):
                dm = np.load(dm_path)
                c_cost = route_cost(local_order, dm)
                total_cost += c_cost
            else:
                c_cost = 0.0

            cluster_routes[cid] = {
                "global_node_order": global_order,
                "local_order"      : local_order,
                "cost"             : c_cost,
                "node_indices"     : node_indices,
            }
        else:
            # No QAOA result — use sequential fallback
            global_order = [0] + node_indices + [0]
            cluster_routes[cid] = {
                "global_node_order": global_order,
                "local_order"      : list(range(len(node_indices) + 1)),
                "cost"             : 0.0,
                "node_indices"     : node_indices,
            }

    # ── Build flat global route ───────────────────────────────────────────────
    global_route = [0]   # start at depot
    for cid in inter_order:
        route = cluster_routes[cid]["global_node_order"]
        # Add non-depot cities from this cluster
        for city in route:
            if city != 0:
                global_route.append(city)
        global_route.append(0)   # return to depot between clusters

    print(f"\n  Global route length : {len(global_route)} waypoints")
    print(f"  Total route cost    : {total_cost:.4f}")

    # ── Visualise ─────────────────────────────────────────────────────────────
    if coords is not None:
        _plot_global_route(global_route, cluster_routes, inter_order,
                           df_summary, coords, total_cost, output_dir)

    # ── Save JSON ─────────────────────────────────────────────────────────────
    output = {
        "global_route"        : global_route,
        "total_cost"          : total_cost,
        "inter_cluster_order" : inter_order,
        "n_clusters"          : n_clusters,
        "cluster_routes"      : {
            str(k): {
                "global_node_order": v["global_node_order"],
                "cost"             : v["cost"],
                "node_indices"     : v["node_indices"],
            }
            for k, v in cluster_routes.items()
        }
    }

    json_path = os.path.join(output_dir, "global_route_results.json")
    with open(json_path, "w") as f:
        json.dump(output, f, indent=4)
    print(f"  Saved: {json_path}")

    return output


# ══════════════════════════════════════════════════════════════════════════════
# Visualisation
# ══════════════════════════════════════════════════════════════════════════════

def _plot_global_route(global_route, cluster_routes, inter_order,
                       df_summary, coords, total_cost, output_dir):
    """Draw the full 50-city global route with cluster colour coding."""
    n_clusters = len(df_summary)
    colors = cm.tab20(np.linspace(0, 1, min(n_clusters, 20)))

    fig, axes = plt.subplots(1, 2, figsize=(17, 7))
    fig.suptitle(
        f"Global VRP Solution — Hybrid Quantum-Classical Pipeline\n"
        f"Total Route Cost: {total_cost:.2f}  |  "
        f"{len(coords)-1} delivery cities  |  {n_clusters} clusters",
        fontsize=13, fontweight='bold'
    )

    # ── Left: full route with cluster colours ─────────────────────────────────
    ax = axes[0]
    depot = coords[0]

    for idx, (cid, row) in enumerate(df_summary.iterrows()):
        node_indices = json.loads(row["node_indices"])
        c = colors[idx % 20]
        cluster_coords = coords[node_indices]
        ax.scatter(cluster_coords[:, 0], cluster_coords[:, 1],
                   color=c, s=60, zorder=3, label=f"C{cid}" if idx < 10 else "")
        for ni in node_indices:
            ax.annotate(str(ni), coords[ni] + np.array([0.5, 0.5]),
                        fontsize=5.5, color=c)

    # Draw route arrows
    for k in range(len(global_route) - 1):
        i, j = global_route[k], global_route[k + 1]
        if i < len(coords) and j < len(coords):
            xi, yi = coords[i]
            xj, yj = coords[j]
            ax.annotate("",
                xy=(xj, yj), xytext=(xi, yi),
                arrowprops=dict(arrowstyle="-|>", color='grey',
                                lw=0.8, mutation_scale=10, alpha=0.6))

    ax.scatter(*depot, color='crimson', marker='D', s=180, zorder=5,
               edgecolors='black', label='Depot')
    ax.set_title("Full 50-City Route (colour = cluster)")
    ax.set_xlabel("X Coordinate")
    ax.set_ylabel("Y Coordinate")
    ax.legend(fontsize=7, ncol=2, loc='upper right')
    ax.grid(True, alpha=0.3)

    # ── Right: per-cluster cost bar chart ─────────────────────────────────────
    ax2 = axes[1]
    cids   = [cid for cid in inter_order if cid in cluster_routes]
    costs  = [cluster_routes[cid]["cost"] for cid in cids]
    bar_colors = [colors[i % 20] for i in range(len(cids))]

    bars = ax2.bar(range(len(cids)), costs, color=bar_colors,
                   edgecolor='black', linewidth=0.6)
    ax2.set_xticks(range(len(cids)))
    ax2.set_xticklabels([f"C{c}" for c in cids], fontsize=8)
    ax2.set_xlabel("Cluster ID")
    ax2.set_ylabel("Sub-route Cost")
    ax2.set_title(f"Per-Cluster Route Cost\nTotal = {total_cost:.2f}")
    ax2.grid(True, axis='y', alpha=0.3)
    for bar, cost in zip(bars, costs):
        ax2.text(bar.get_x() + bar.get_width() / 2,
                 bar.get_height() + 0.5,
                 f"{cost:.1f}", ha='center', va='bottom', fontsize=7)

    plt.tight_layout()
    path = os.path.join(output_dir, "global_route.png")
    plt.savefig(path, dpi=150, bbox_inches='tight')
    plt.close()
    print(f"  Saved: {path}")
