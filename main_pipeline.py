"""
main_pipeline.py
================
Master Orchestrator — Hybrid Quantum-Classical VRP Pipeline.

Runs all 5 pipeline stages end-to-end:

    Stage 1 — Cluster Scaler
        K-Means + recursive splitting partitions 50 cities into NISQ-safe
        micro-clusters (≤ 4 delivery nodes + depot = ≤ 20 qubits each).

    Stage 2 — QUBO Builder
        Each cluster's distance matrix is converted into a Qiskit
        QuadraticProgram and then into an Ising Hamiltonian.

    Stage 3 — QAOA Solver
        QAOA with SPSA optimizer runs on each cluster's Hamiltonian.
        Circuit depth is swept (p = 1, 2, 3) and the best result kept.

    Stage 4 — Global Stitcher
        Per-cluster optimal sub-routes are decoded and assembled into a
        single coherent 50-city delivery route via nearest-neighbour
        centroid heuristic.

    Stage 5 — Summary Report
        All results are aggregated into a final JSON + printed summary.

USAGE:
    python main_pipeline.py [--clusters N] [--depths p1 p2 ...] [--iters N]

    --clusters  : number of clusters to solve (default: all)
    --depths    : QAOA depths to sweep (default: 1 2 3)
    --iters     : SPSA iterations per depth (default: 50)
    --fast      : shortcut for --clusters 3 --depths 1 --iters 30
"""

import argparse
import json
import os
import time
import numpy as np
import pandas as pd
import matplotlib.pyplot as plt
import matplotlib.cm as cm

# ── Pipeline modules ──────────────────────────────────────────────────────────
from qaoa_solver    import solve_cluster
from global_stitcher import stitch_global_route


# ══════════════════════════════════════════════════════════════════════════════
# CLI arguments
# ══════════════════════════════════════════════════════════════════════════════

def parse_args():
    parser = argparse.ArgumentParser(
        description="Hybrid Quantum-Classical VRP Pipeline"
    )
    parser.add_argument("--clusters", type=int, default=None,
                        help="Max clusters to solve (default: all)")
    parser.add_argument("--depths", type=int, nargs="+", default=[1, 2, 3],
                        help="QAOA circuit depths to sweep (default: 1 2 3)")
    parser.add_argument("--iters", type=int, default=50,
                        help="SPSA iterations per depth (default: 50)")
    parser.add_argument("--penalty", type=float, default=1000.0,
                        help="QUBO constraint penalty weight (default: 1000)")
    parser.add_argument("--fast", action="store_true",
                        help="Quick test: 3 clusters, depth 1, 30 iters")
    return parser.parse_args()


# ══════════════════════════════════════════════════════════════════════════════
# Stage 1 — Cluster Scaler
# ══════════════════════════════════════════════════════════════════════════════

def run_stage1():
    """
    Run cluster_scaler.py to partition 50 cities into NISQ-safe clusters.
    Generates: cluster_summary.csv, distance_matrices/cluster_XX.npy,
               cluster_map.png
    """
    print("\n" + "█"*65)
    print("  STAGE 1 — CLUSTER SCALER")
    print("█"*65)

    # Check if already done
    if (os.path.exists("cluster_summary.csv") and
            os.path.exists("distance_matrices")):
        n_existing = len([f for f in os.listdir("distance_matrices")
                          if f.endswith(".npy")])
        if n_existing > 0:
            print(f"  Found existing cluster data ({n_existing} clusters).")
            print("  Skipping re-generation. Delete cluster_summary.csv to re-run.")
            df = pd.read_csv("cluster_summary.csv")
            return df

    # Run the scaler
    import importlib.util, sys
    spec = importlib.util.spec_from_file_location("cluster_scaler", "cluster_scaler.py")
    mod  = importlib.util.module_from_spec(spec)
    spec.loader.exec_module(mod)

    df = pd.read_csv("cluster_summary.csv")
    print(f"\n  Stage 1 complete: {len(df)} clusters generated.")
    return df


# ══════════════════════════════════════════════════════════════════════════════
# Stage 2+3 — QUBO Build + QAOA Solve (per cluster)
# ══════════════════════════════════════════════════════════════════════════════

def run_stages_2_3(df_summary, max_clusters, p_depths, max_iters, penalty):
    """
    For each cluster: build QUBO Hamiltonian, run QAOA, collect results.
    """
    print("\n" + "█"*65)
    print("  STAGE 2+3 — QUBO BUILD + QAOA SOLVE")
    print("█"*65)

    cluster_ids = df_summary["cluster_id"].tolist()
    if max_clusters is not None:
        cluster_ids = cluster_ids[:max_clusters]

    print(f"\n  Clusters to solve : {len(cluster_ids)}")
    print(f"  QAOA depths       : {p_depths}")
    print(f"  SPSA iterations   : {max_iters} per depth")
    print(f"  Penalty weight    : {penalty}")

    all_results = []
    t0 = time.time()

    for cid in cluster_ids:
        matrix_path = f"distance_matrices/cluster_{cid:02d}.npy"
        if not os.path.exists(matrix_path):
            print(f"\n  WARNING: {matrix_path} not found — skipping cluster {cid}")
            continue

        t_start = time.time()
        result = solve_cluster(
            distance_matrix_path=matrix_path,
            cluster_id=cid,
            p_depths=p_depths,
            max_iterations=max_iters,
            penalty_weight=penalty,
            output_dir="outputs",
        )
        elapsed = time.time() - t_start
        result["wall_time_s"] = round(elapsed, 2)
        all_results.append(result)

        print(f"\n  Cluster {cid:02d} done in {elapsed:.1f}s  "
              f"| cost={result['best_cost']:.2f}  "
              f"| depth={result['best_depth']}")

    total_time = time.time() - t0
    print(f"\n  Stage 2+3 complete: {len(all_results)} clusters solved "
          f"in {total_time:.1f}s")
    return all_results


# ══════════════════════════════════════════════════════════════════════════════
# Stage 4 — Global Stitcher
# ══════════════════════════════════════════════════════════════════════════════

def run_stage4(cluster_results, df_summary):
    """
    Assemble per-cluster sub-routes into the final 50-city global route.
    """
    print("\n" + "█"*65)
    print("  STAGE 4 — GLOBAL STITCHER")
    print("█"*65)

    # Reconstruct global city coordinates (same seed as cluster_scaler.py)
    np.random.seed(42)
    coords = np.random.uniform(0, 100, size=(50, 2))

    global_result = stitch_global_route(
        cluster_results=cluster_results,
        cluster_summary_path="cluster_summary.csv",
        coords=coords,
        output_dir="outputs",
    )

    print(f"\n  Stage 4 complete.")
    print(f"  Global route waypoints : {len(global_result['global_route'])}")
    print(f"  Total route cost       : {global_result['total_cost']:.4f}")
    return global_result


# ══════════════════════════════════════════════════════════════════════════════
# Stage 5 — Summary
# ══════════════════════════════════════════════════════════════════════════════

def run_stage5(cluster_results, global_result, df_summary, args):
    """
    Aggregate all results, print final summary, save pipeline_summary.json,
    and generate the pipeline overview chart.
    """
    print("\n" + "█"*65)
    print("  STAGE 5 — PIPELINE SUMMARY")
    print("█"*65)

    # ── Aggregate stats ───────────────────────────────────────────────────────
    costs      = [r["best_cost"] for r in cluster_results if not np.isnan(r["best_cost"])]
    depths     = [r["best_depth"] for r in cluster_results]
    qubits     = [r["n_qubits"] for r in cluster_results]
    wall_times = [r.get("wall_time_s", 0) for r in cluster_results]

    summary = {
        "pipeline_config": {
            "n_cities"       : 50,
            "n_clusters"     : len(df_summary),
            "clusters_solved": len(cluster_results),
            "qaoa_depths"    : args.depths,
            "spsa_iters"     : args.iters,
            "penalty_weight" : args.penalty,
        },
        "qubit_stats": {
            "max_qubits_used"  : int(max(qubits)) if qubits else 0,
            "min_qubits_used"  : int(min(qubits)) if qubits else 0,
            "avg_qubits_used"  : float(np.mean(qubits)) if qubits else 0,
            "nisq_limit"       : 20,
            "all_nisq_safe"    : bool(max(qubits) <= 20) if qubits else True,
        },
        "cost_stats": {
            "total_global_cost": global_result["total_cost"],
            "avg_cluster_cost" : float(np.mean(costs)) if costs else 0,
            "min_cluster_cost" : float(min(costs)) if costs else 0,
            "max_cluster_cost" : float(max(costs)) if costs else 0,
        },
        "performance": {
            "total_wall_time_s"  : sum(wall_times),
            "avg_time_per_cluster": float(np.mean(wall_times)) if wall_times else 0,
        },
        "global_route": {
            "waypoints"          : len(global_result["global_route"]),
            "inter_cluster_order": global_result["inter_cluster_order"],
        },
        "cluster_results": [
            {
                "cluster_id"    : r["cluster_id"],
                "best_depth"    : r["best_depth"],
                "best_cost"     : r["best_cost"],
                "n_qubits"      : r["n_qubits"],
                "wall_time_s"   : r.get("wall_time_s", 0),
                "best_bitstring": r["best_bitstring"],
            }
            for r in cluster_results
        ]
    }

    # Save JSON
    os.makedirs("outputs", exist_ok=True)
    with open("outputs/pipeline_summary.json", "w") as f:
        json.dump(summary, f, indent=4)
    print(f"\n  Saved: outputs/pipeline_summary.json")

    # ── Overview chart ────────────────────────────────────────────────────────
    _plot_pipeline_overview(cluster_results, global_result, df_summary)

    # ── Print summary ─────────────────────────────────────────────────────────
    print("\n" + "="*65)
    print("  FINAL PIPELINE SUMMARY")
    print("="*65)
    print(f"  Cities              : 50")
    print(f"  Clusters formed     : {len(df_summary)}")
    print(f"  Clusters solved     : {len(cluster_results)}")
    print(f"  Max qubits used     : {max(qubits) if qubits else 'N/A'}")
    print(f"  NISQ-safe (≤20 q)   : {'YES' if (max(qubits) <= 20 if qubits else True) else 'NO'}")
    print(f"  Total route cost    : {global_result['total_cost']:.4f}")
    print(f"  Total compute time  : {sum(wall_times):.1f}s")
    print("="*65)
    print("\n  Output files:")
    for f in sorted(os.listdir("outputs")):
        print(f"    outputs/{f}")
    print("="*65 + "\n")

    return summary


def _plot_pipeline_overview(cluster_results, global_result, df_summary):
    """4-panel pipeline overview chart."""
    fig, axes = plt.subplots(2, 2, figsize=(14, 10))
    fig.suptitle("Hybrid Quantum-Classical VRP Pipeline — Overview",
                 fontsize=14, fontweight='bold')

    cids   = [r["cluster_id"] for r in cluster_results]
    costs  = [r["best_cost"] for r in cluster_results]
    qubits = [r["n_qubits"] for r in cluster_results]
    depths = [r["best_depth"] for r in cluster_results]
    times  = [r.get("wall_time_s", 0) for r in cluster_results]

    colors = cm.tab20(np.linspace(0, 1, max(len(cids), 1)))

    # Panel 1: Cost per cluster
    ax = axes[0, 0]
    ax.bar(range(len(cids)), costs, color=colors[:len(cids)],
           edgecolor='black', lw=0.5)
    ax.set_xticks(range(len(cids)))
    ax.set_xticklabels([f"C{c}" for c in cids], fontsize=7, rotation=45)
    ax.set_ylabel("QAOA Cost")
    ax.set_title("Per-Cluster QAOA Cost")
    ax.grid(True, axis='y', alpha=0.3)

    # Panel 2: Qubit count per cluster
    ax = axes[0, 1]
    bar_colors = ['#2ecc71' if q <= 20 else '#e74c3c' for q in qubits]
    ax.bar(range(len(cids)), qubits, color=bar_colors,
           edgecolor='black', lw=0.5)
    ax.axhline(20, color='darkorange', ls='--', lw=2, label='NISQ limit (20)')
    ax.set_xticks(range(len(cids)))
    ax.set_xticklabels([f"C{c}" for c in cids], fontsize=7, rotation=45)
    ax.set_ylabel("Qubits")
    ax.set_title("Qubits per Cluster (green = NISQ-safe)")
    ax.legend(fontsize=9)
    ax.grid(True, axis='y', alpha=0.3)

    # Panel 3: Best depth per cluster
    ax = axes[1, 0]
    ax.bar(range(len(cids)), depths, color='steelblue',
           edgecolor='black', lw=0.5)
    ax.set_xticks(range(len(cids)))
    ax.set_xticklabels([f"C{c}" for c in cids], fontsize=7, rotation=45)
    ax.set_ylabel("Best QAOA Depth (p)")
    ax.set_title("Optimal Circuit Depth per Cluster")
    ax.grid(True, axis='y', alpha=0.3)

    # Panel 4: Wall time per cluster
    ax = axes[1, 1]
    ax.bar(range(len(cids)), times, color='#9b59b6',
           edgecolor='black', lw=0.5)
    ax.set_xticks(range(len(cids)))
    ax.set_xticklabels([f"C{c}" for c in cids], fontsize=7, rotation=45)
    ax.set_ylabel("Wall Time (s)")
    ax.set_title(f"Compute Time per Cluster\nTotal: {sum(times):.1f}s")
    ax.grid(True, axis='y', alpha=0.3)

    plt.tight_layout()
    plt.savefig("outputs/pipeline_overview.png", dpi=150, bbox_inches='tight')
    plt.close()
    print("  Saved: outputs/pipeline_overview.png")


# ══════════════════════════════════════════════════════════════════════════════
# Entry point
# ══════════════════════════════════════════════════════════════════════════════

def main():
    args = parse_args()

    if args.fast:
        args.clusters = 3
        args.depths   = [1]
        args.iters    = 30
        print("  [FAST MODE] clusters=3, depth=1, iters=30")

    print("\n" + "█"*65)
    print("  HYBRID QUANTUM-CLASSICAL VRP PIPELINE")
    print("  NISQ-Era Scalable Architecture")
    print("█"*65)
    print(f"\n  Config:")
    print(f"    Max clusters : {args.clusters or 'all'}")
    print(f"    QAOA depths  : {args.depths}")
    print(f"    SPSA iters   : {args.iters}")
    print(f"    Penalty      : {args.penalty}")

    t_pipeline_start = time.time()

    # Stage 1
    df_summary = run_stage1()

    # Stages 2+3
    cluster_results = run_stages_2_3(
        df_summary   = df_summary,
        max_clusters = args.clusters,
        p_depths     = args.depths,
        max_iters    = args.iters,
        penalty      = args.penalty,
    )

    if not cluster_results:
        print("\nERROR: No clusters were solved. Check distance_matrices/ directory.")
        return

    # Stage 4
    global_result = run_stage4(cluster_results, df_summary)

    # Stage 5
    summary = run_stage5(cluster_results, global_result, df_summary, args)

    total_time = time.time() - t_pipeline_start
    print(f"  Pipeline completed in {total_time:.1f}s\n")


if __name__ == "__main__":
    main()
