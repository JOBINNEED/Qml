"""
qaoa_solver.py
==============
Step 4 of the Hybrid Quantum-Classical VRP Pipeline.

PURPOSE:
    Executes QAOA on the Ising Hamiltonian produced by qubo_builder.py
    for a single VRP cluster. Sweeps circuit depth p = 1..5 and records
    convergence, then returns the best bitstring (= optimal sub-route).

OPTIMIZER CHOICE — SPSA:
    Simultaneous Perturbation Stochastic Approximation is used instead of
    gradient-based methods because:
      • Quantum circuits have noisy, non-differentiable landscapes.
      • SPSA estimates the gradient with only 2 circuit evaluations per
        step regardless of parameter count (vs 2p for finite differences).
      • It naturally escapes shallow local minima ("barren plateaus").

OUTPUTS:
    outputs/qaoa_convergence_<cluster_id>.png
    outputs/qaoa_results_<cluster_id>.csv
    Returns dict with best bitstring and cost for the global stitcher.
"""

import numpy as np
import pandas as pd
import matplotlib.pyplot as plt
import os

from qiskit_algorithms import QAOA
from qiskit_algorithms.optimizers import SPSA
from qiskit.primitives import StatevectorSampler
from qiskit_optimization.converters import QuadraticProgramToQubo

from qubo_builder import VRPQuboBuilder


# ══════════════════════════════════════════════════════════════════════════════
# Core solver
# ══════════════════════════════════════════════════════════════════════════════

def solve_cluster(distance_matrix_path: str,
                  cluster_id: int = 0,
                  p_depths: list = None,
                  max_iterations: int = 100,
                  penalty_weight: float = 1000.0,
                  output_dir: str = "outputs") -> dict:
    """
    Run QAOA depth sweep on one VRP cluster.

    Parameters
    ----------
    distance_matrix_path : str
        Path to .npy distance matrix for this cluster.
    cluster_id : int
        Cluster index (used for output file naming).
    p_depths : list[int]
        QAOA circuit depths to sweep. Default [1, 2, 3].
    max_iterations : int
        SPSA iterations per depth level.
    penalty_weight : float
        QUBO constraint penalty coefficient.
    output_dir : str
        Directory for plots and CSV output.

    Returns
    -------
    dict with keys:
        cluster_id, best_depth, best_cost, best_bitstring,
        n_nodes, n_qubits, results_df
    """
    if p_depths is None:
        p_depths = [1, 2, 3]

    os.makedirs(output_dir, exist_ok=True)

    print(f"\n{'='*65}")
    print(f"  QAOA SOLVER  —  Cluster {cluster_id:02d}")
    print(f"{'='*65}")

    # ── Step 1: Build Hamiltonian ─────────────────────────────────────────────
    builder = VRPQuboBuilder(distance_matrix_path, penalty_weight=penalty_weight)
    qp      = builder.build_quadratic_program()
    operator, offset, _ = builder.convert_to_ising(qp)

    n_qubits = operator.num_qubits
    print(f"\n  Hamiltonian : {n_qubits} qubits, {len(operator)} Pauli terms")
    print(f"  Depths      : {p_depths}")
    print(f"  SPSA iters  : {max_iterations} per depth")

    # ── Step 2: Depth sweep ───────────────────────────────────────────────────
    records = []
    sampler = StatevectorSampler()

    for p in p_depths:
        print(f"\n  --- p = {p} ---")

        spsa = SPSA(
            maxiter=max_iterations,
            learning_rate=0.01,
            perturbation=0.1,
        )

        qaoa = QAOA(
            sampler=sampler,
            optimizer=spsa,
            reps=p,
        )

        try:
            result = qaoa.compute_minimum_eigenvalue(operator)
            eigenvalue = float(np.real(result.eigenvalue))
            cost       = eigenvalue + offset

            # Extract best bitstring from measurement distribution
            best_bitstring = _extract_bitstring(result, n_qubits)

            print(f"    eigenvalue : {eigenvalue:.4f}")
            print(f"    cost       : {cost:.4f}")
            print(f"    bitstring  : {best_bitstring}")

            records.append({
                "cluster_id"   : cluster_id,
                "depth"        : p,
                "eigenvalue"   : eigenvalue,
                "cost"         : cost,
                "bitstring"    : best_bitstring,
            })

        except Exception as exc:
            print(f"    ERROR at p={p}: {exc}")
            records.append({
                "cluster_id"   : cluster_id,
                "depth"        : p,
                "eigenvalue"   : np.nan,
                "cost"         : np.nan,
                "bitstring"    : "ERROR",
            })

    # ── Step 3: Pick best result ──────────────────────────────────────────────
    df = pd.DataFrame(records)
    valid = df.dropna(subset=["cost"])

    if valid.empty:
        print("  WARNING: All depths failed. Returning random bitstring.")
        best_row = {"depth": 0, "cost": np.nan,
                    "bitstring": "0" * n_qubits}
    else:
        best_row = valid.loc[valid["cost"].idxmin()].to_dict()

    print(f"\n  Best depth  : {best_row['depth']}")
    print(f"  Best cost   : {best_row['cost']}")
    print(f"  Best bits   : {best_row['bitstring']}")

    # ── Step 4: Save outputs ──────────────────────────────────────────────────
    csv_path = os.path.join(output_dir, f"qaoa_results_cluster{cluster_id:02d}.csv")
    df.to_csv(csv_path, index=False)
    print(f"\n  Saved: {csv_path}")

    _plot_convergence(df, cluster_id, output_dir)

    return {
        "cluster_id"    : cluster_id,
        "best_depth"    : int(best_row["depth"]),
        "best_cost"     : float(best_row["cost"]) if not np.isnan(best_row["cost"]) else 0.0,
        "best_bitstring": str(best_row["bitstring"]),
        "n_nodes"       : builder.n_nodes,
        "n_qubits"      : n_qubits,
        "results_df"    : df,
    }


# ══════════════════════════════════════════════════════════════════════════════
# Bitstring extraction helper
# ══════════════════════════════════════════════════════════════════════════════

def _extract_bitstring(result, n_qubits: int) -> str:
    """
    Pull the most-probable bitstring from a QAOA result object.
    Handles both Qiskit 1.x result formats gracefully.
    """
    # Qiskit 1.x: result.best_measurement is a dict
    if hasattr(result, "best_measurement") and result.best_measurement:
        bm = result.best_measurement
        if isinstance(bm, dict) and "bitstring" in bm:
            return bm["bitstring"]

    # Fallback: eigenstate is a dict of {bitstring: amplitude}
    if hasattr(result, "eigenstate") and result.eigenstate is not None:
        es = result.eigenstate
        if isinstance(es, dict) and es:
            return max(es, key=lambda k: abs(es[k]) ** 2)

    # Last resort: zero string
    return "0" * n_qubits


# ══════════════════════════════════════════════════════════════════════════════
# Visualisation
# ══════════════════════════════════════════════════════════════════════════════

def _plot_convergence(df: pd.DataFrame, cluster_id: int, output_dir: str):
    """Plot cost and eigenvalue vs circuit depth for one cluster."""
    valid = df.dropna(subset=["cost"])
    if valid.empty:
        return

    fig, (ax1, ax2) = plt.subplots(1, 2, figsize=(12, 4))
    fig.suptitle(f"QAOA Convergence — Cluster {cluster_id:02d}",
                 fontsize=13, fontweight='bold')

    # Cost vs depth
    ax1.plot(valid["depth"], valid["cost"], "o-", lw=2, ms=8, color="#2E86AB")
    ax1.axhline(valid["cost"].min(), color="#A23B72", ls="--", lw=1.5,
                label=f"Best: {valid['cost'].min():.2f}")
    ax1.set_xlabel("Circuit Depth (p)")
    ax1.set_ylabel("Cost (eigenvalue + offset)")
    ax1.set_title("Cost vs Depth")
    ax1.legend()
    ax1.grid(True, alpha=0.3)
    ax1.set_xticks(valid["depth"])

    # Eigenvalue vs depth
    ax2.plot(valid["depth"], valid["eigenvalue"], "s-", lw=2, ms=8, color="#F18F01")
    ax2.set_xlabel("Circuit Depth (p)")
    ax2.set_ylabel("Eigenvalue")
    ax2.set_title("Eigenvalue vs Depth")
    ax2.grid(True, alpha=0.3)
    ax2.set_xticks(valid["depth"])

    plt.tight_layout()
    path = os.path.join(output_dir, f"qaoa_convergence_cluster{cluster_id:02d}.png")
    plt.savefig(path, dpi=150, bbox_inches='tight')
    plt.close()
    print(f"  Saved: {path}")


# ══════════════════════════════════════════════════════════════════════════════
# Standalone entry point (single cluster test)
# ══════════════════════════════════════════════════════════════════════════════

def main():
    matrix_path = "distance_matrices/cluster_00.npy"
    if not os.path.exists(matrix_path):
        print(f"ERROR: {matrix_path} not found. Run cluster_scaler.py first.")
        return

    result = solve_cluster(
        distance_matrix_path=matrix_path,
        cluster_id=0,
        p_depths=[1, 2, 3],
        max_iterations=50,
        penalty_weight=1000.0,
    )

    print("\n" + "="*65)
    print("  QAOA SOLVER COMPLETE")
    print("="*65)
    print(f"  Cluster     : {result['cluster_id']}")
    print(f"  Best depth  : {result['best_depth']}")
    print(f"  Best cost   : {result['best_cost']:.4f}")
    print(f"  Bitstring   : {result['best_bitstring']}")
    print(f"  Qubits used : {result['n_qubits']}")
    print("="*65 + "\n")


if __name__ == "__main__":
    main()
