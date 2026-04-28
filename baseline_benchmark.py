"""
baseline_benchmark.py
=====================
Step 1 of the Hybrid Quantum-Classical VRP Pipeline.

PURPOSE:
    Establish a mathematical ground truth by solving a small 3-node VRP
    (depot + 2 delivery cities, 2 vehicles) with:
        (a) IBM CPLEX  — exact classical solver (optimal solution)
        (b) QAOA       — quantum approximate solver (Qiskit 1.x native API)

    The comparison proves the foundational concept: QAOA can recover
    near-optimal routing solutions on small instances, validating the
    quantum formulation before scaling up.

QUBIT COUNT:
    n=3 nodes → N = n*(n-1) = 6 binary variables → 6 qubits (NISQ-safe)

OUTPUTS:
    outputs/baseline_nodes_distances.csv   — node coordinates
    outputs/baseline_distance_matrix.csv   — raw distance matrix
    outputs/cplex_route.png                — classical optimal route
    outputs/qaoa_route.png                 — quantum route
    outputs/baseline_routing_results.json  — cost comparison JSON
    outputs/baseline_comparison.png        — side-by-side comparison chart
"""

import numpy as np
import matplotlib.pyplot as plt
import math
import os
import json
import pandas as pd

# ── Qiskit 1.x imports ────────────────────────────────────────────────────────
from qiskit_algorithms import QAOA
from qiskit_algorithms.optimizers import SPSA, COBYLA
from qiskit.primitives import StatevectorSampler
from qiskit_optimization import QuadraticProgram
from qiskit_optimization.algorithms import MinimumEigenOptimizer

# ── CPLEX (optional — graceful fallback if not installed) ─────────────────────
try:
    import cplex
    from cplex.exceptions import CplexError
    CPLEX_AVAILABLE = True
except ImportError:
    CPLEX_AVAILABLE = False
    print("WARNING: CPLEX not installed. Classical baseline will be skipped.")
    print("         Install with: pip install cplex")

# ── Output directory ──────────────────────────────────────────────────────────
os.makedirs("outputs", exist_ok=True)

# ── Problem parameters ────────────────────────────────────────────────────────
N_NODES = 3   # depot (0) + 2 delivery cities
K_VEHICLES = 2

print("=" * 65)
print("  BASELINE BENCHMARK  —  Classical CPLEX vs Quantum QAOA")
print("=" * 65)
print(f"\n  Nodes     : {N_NODES}  (depot at index 0 + {N_NODES-1} delivery cities)")
print(f"  Vehicles  : {K_VEHICLES}")
print(f"  Qubits    : {N_NODES * (N_NODES - 1)}  [N = n×(n-1)]")
print(f"  NISQ-safe : {'YES' if N_NODES*(N_NODES-1) <= 20 else 'NO'}")


# ══════════════════════════════════════════════════════════════════════════════
# 1. PROBLEM INSTANCE GENERATOR
# ══════════════════════════════════════════════════════════════════════════════

class Initializer:
    """Randomly places n nodes in a 2-D plane and computes Euclidean distances."""

    def __init__(self, n, seed=1543):
        self.n = n
        self.seed = seed

    def generate_instance(self):
        np.random.seed(self.seed)
        xc = (np.random.rand(self.n) - 0.5) * 10
        yc = (np.random.rand(self.n) - 0.5) * 10

        dist = np.zeros((self.n, self.n))
        for i in range(self.n):
            for j in range(i + 1, self.n):
                d = (xc[i] - xc[j]) ** 2 + (yc[i] - yc[j]) ** 2
                dist[i, j] = d
                dist[j, i] = d
        return xc, yc, dist


initializer = Initializer(N_NODES)
xc, yc, instance = initializer.generate_instance()

# Persist node data
pd.DataFrame({'node': range(N_NODES), 'xc': xc, 'yc': yc}).to_csv(
    "outputs/baseline_nodes_distances.csv", index=False
)
np.savetxt("outputs/baseline_distance_matrix.csv", instance, delimiter=",")
print(f"\n[1] Generated {N_NODES}-node instance. Saved coordinates and distance matrix.")


# ══════════════════════════════════════════════════════════════════════════════
# 2. CLASSICAL SOLVER — IBM CPLEX
# ══════════════════════════════════════════════════════════════════════════════

class ClassicalOptimizer:
    """
    Solves VRP exactly using IBM CPLEX Integer Linear Programming.
    Formulation follows the Miller-Tucker-Zemlin (MTZ) sub-tour elimination.
    """

    def __init__(self, instance, n, K):
        self.instance = instance
        self.n = n
        self.K = K

    def compute_allowed_combinations(self):
        f = math.factorial
        return f(self.n) / f(self.K) / f(self.n - self.K)

    def cplex_solution(self):
        instance = self.instance
        n, K = self.n, self.K

        my_obj  = list(instance.reshape(1, n**2)[0]) + [0.0] * (n - 1)
        my_ub   = [1] * (n**2 + n - 1)
        my_lb   = [0] * n**2 + [0.1] * (n - 1)
        my_ctype = "I" * n**2 + "C" * (n - 1)

        my_rhs = (
            2 * ([K] + [1] * (n - 1))
            + [1 - 0.1] * ((n - 1) ** 2 - (n - 1))
            + [0] * n
        )
        my_sense = (
            "E" * (2 * n)
            + "L" * ((n - 1) ** 2 - (n - 1))
            + "E" * n
        )

        my_prob = cplex.Cplex()
        self._populate(my_prob, my_obj, my_ub, my_lb, my_ctype, my_sense, my_rhs)
        my_prob.solve()

        x    = np.array(my_prob.solution.get_values())
        cost = my_prob.solution.get_objective_value()
        return x, cost

    def _populate(self, prob, my_obj, my_ub, my_lb, my_ctype, my_sense, my_rhs):
        n = self.n
        prob.objective.set_sense(prob.objective.sense.minimize)
        prob.variables.add(obj=my_obj, lb=my_lb, ub=my_ub, types=my_ctype)
        for stream in [prob.set_log_stream, prob.set_error_stream,
                       prob.set_warning_stream, prob.set_results_stream]:
            stream(None)

        rows = []
        for ii in range(n):
            rows.append([[x for x in range(n * ii, n * (ii + 1))], [1] * n])
        for ii in range(n):
            rows.append([[x for x in range(ii, n**2, n)], [1] * n])
        for ii in range(n):
            for jj in range(n):
                if ii != jj and ii * jj > 0:
                    rows.append([[ii + jj * n, n**2 + ii - 1, n**2 + jj - 1], [1, 1, -1]])
        for ii in range(n):
            rows.append([[ii * (n + 1)], [1]])

        prob.linear_constraints.add(lin_expr=rows, senses=my_sense, rhs=my_rhs)


x_classical, classical_cost, z_classical = None, 0.0, None

if CPLEX_AVAILABLE:
    print("\n[2] Running CPLEX exact solver...")
    try:
        classical_optimizer = ClassicalOptimizer(instance, N_NODES, K_VEHICLES)
        combos = classical_optimizer.compute_allowed_combinations()
        print(f"    Feasible combinations: {int(combos)}")
        x_classical, classical_cost = classical_optimizer.cplex_solution()
        z_classical = [x_classical[ii] for ii in range(N_NODES**2)
                       if ii // N_NODES != ii % N_NODES]
        print(f"    ✓ CPLEX cost: {classical_cost:.4f}")
        print(f"    Route vector z: {[round(v, 2) for v in z_classical]}")
    except Exception as e:
        print(f"    ✗ CPLEX error: {e}")
        CPLEX_AVAILABLE = False
else:
    print("\n[2] CPLEX not available — skipping classical solve.")


# ══════════════════════════════════════════════════════════════════════════════
# 3. QUANTUM FORMULATION — QUBO / ISING HAMILTONIAN
# ══════════════════════════════════════════════════════════════════════════════

class QuantumOptimizer:
    """
    Converts the VRP distance matrix into a QUBO / Ising Hamiltonian
    and solves it with Qiskit's native QAOA (Qiskit 1.x API).

    Binary encoding (edge-based):
        x[i→j] = 1 if vehicle travels directly from city i to city j
        Variables: n*(n-1) binary vars (all i≠j pairs)

    Penalty weight A = max(distance) * 100 ensures constraint violations
    are always more expensive than any feasible route.
    """

    def __init__(self, instance, n, K):
        self.instance = instance
        self.n = n
        self.K = K

    # ------------------------------------------------------------------
    # Build QUBO matrices Q, g, c
    # ------------------------------------------------------------------
    def binary_representation(self, x_sol=None):
        instance, n, K = self.instance, self.n, self.K
        A = np.max(instance) * 100          # large penalty weight

        instance_vec = instance.reshape(n**2)
        w = np.array([v for v in instance_vec if v > 0], dtype=float)
        # Pad to n*(n-1) if needed
        w_full = np.zeros(n * (n - 1))
        w_full[:len(w)] = w

        Id_n     = np.eye(n)
        Im_n_1   = np.ones((n - 1, n - 1))
        Iv_n_1   = np.ones(n);  Iv_n_1[0] = 0
        Iv_n     = np.ones(n - 1)
        neg_Iv   = np.ones(n) - Iv_n_1

        v = np.zeros((n, n * (n - 1)))
        for ii in range(n):
            count = ii - 1
            for jj in range(n * (n - 1)):
                if jj // (n - 1) == ii:
                    count = ii
                if jj // (n - 1) != ii and jj % (n - 1) == count:
                    v[ii][jj] = 1.0

        vn = np.sum(v[1:], axis=0)

        Q = A * (np.kron(Id_n, Im_n_1) + v.T @ v)
        g = (w_full
             - 2 * A * (np.kron(Iv_n_1, Iv_n) + vn.T)
             - 2 * A * K * (np.kron(neg_Iv, Iv_n) + v[0].T))
        c = 2 * A * (n - 1) + 2 * A * (K ** 2)

        cost = 0.0
        if x_sol is not None:
            x = np.around(x_sol)
            cost = float(x @ Q @ x + g @ x + c)

        return Q, g, c, cost

    # ------------------------------------------------------------------
    # Build Qiskit QuadraticProgram
    # ------------------------------------------------------------------
    def construct_problem(self, Q, g, c) -> QuadraticProgram:
        n_vars = self.n * (self.n - 1)
        qp = QuadraticProgram(name="VRP_Baseline")
        for i in range(n_vars):
            qp.binary_var(str(i))
        qp.minimize(constant=c, linear=dict(enumerate(g)),
                    quadratic={(i, j): Q[i, j]
                               for i in range(n_vars)
                               for j in range(n_vars) if Q[i, j] != 0})
        return qp

    # ------------------------------------------------------------------
    # Solve with QAOA (Qiskit 1.x)
    # ------------------------------------------------------------------
    def solve_problem(self, qp):
        sampler = StatevectorSampler()
        optimizer = SPSA(maxiter=100)
        qaoa = QAOA(sampler=sampler, optimizer=optimizer, reps=1)
        solver = MinimumEigenOptimizer(qaoa)
        result = solver.solve(qp)
        _, _, _, cost = self.binary_representation(x_sol=result.x)
        return result.x, cost


print("\n[3] Building QUBO / Ising Hamiltonian...")
quantum_optimizer = QuantumOptimizer(instance, N_NODES, K_VEHICLES)
Q, g, c, _ = quantum_optimizer.binary_representation()

# Verify binary formulation against CPLEX (if available)
if z_classical is not None:
    _, _, _, binary_cost = quantum_optimizer.binary_representation(x_sol=z_classical)
    print(f"    Binary cost of CPLEX solution : {binary_cost:.4f}")
    print(f"    CPLEX objective cost          : {classical_cost:.4f}")
    match = "✓ MATCH" if abs(binary_cost - classical_cost) < 1.0 else "⚠ MISMATCH (check penalty A)"
    print(f"    Verification                  : {match}")

qp = quantum_optimizer.construct_problem(Q, g, c)
print(f"    Variables: {qp.get_num_vars()}, Binary: {qp.get_num_binary_vars()}")


# ══════════════════════════════════════════════════════════════════════════════
# 4. QUANTUM SOLVE — QAOA
# ══════════════════════════════════════════════════════════════════════════════

print("\n[4] Running QAOA (SPSA optimizer, p=1)...")
quantum_solution, quantum_cost = quantum_optimizer.solve_problem(qp)
print(f"    ✓ QAOA cost: {quantum_cost:.4f}")
print(f"    Solution bitstring: {quantum_solution.astype(int).tolist()}")

# Reconstruct full n×n route matrix from edge-based solution
x_quantum = np.zeros(N_NODES ** 2)
kk = 0
for ii in range(N_NODES ** 2):
    if ii // N_NODES != ii % N_NODES:
        x_quantum[ii] = quantum_solution[kk]
        kk += 1


# ══════════════════════════════════════════════════════════════════════════════
# 5. VISUALISATION
# ══════════════════════════════════════════════════════════════════════════════

def plot_route(ax, xc, yc, x, cost, n, title):
    """Draw a VRP route on a given matplotlib axis."""
    ax.scatter(xc, yc, s=220, zorder=4, color='steelblue', edgecolors='black')
    for i in range(n):
        ax.annotate(str(i), (xc[i] + 0.15, yc[i] + 0.15), fontsize=13,
                    color='black', fontweight='bold')
    ax.plot(xc[0], yc[0], 'r*', ms=22, zorder=5, label='Depot')
    ax.grid(True, alpha=0.3)

    for ii in range(n ** 2):
        if x[ii] > 0.5:
            ix, iy = ii // n, ii % n
            ax.annotate("",
                xy=(xc[iy], yc[iy]),
                xytext=(xc[ix], yc[ix]),
                arrowprops=dict(arrowstyle="-|>", color='darkorange',
                                lw=2.0, mutation_scale=18))

    ax.set_title(f"{title}\nCost = {cost:.2f}", fontsize=12, fontweight='bold')
    ax.legend(fontsize=9)


# Side-by-side comparison
fig, axes = plt.subplots(1, 2, figsize=(13, 5))
fig.suptitle("Baseline Benchmark: Classical CPLEX vs Quantum QAOA\n"
             f"({N_NODES} nodes, {K_VEHICLES} vehicles, {N_NODES*(N_NODES-1)} qubits)",
             fontsize=13, fontweight='bold')

if x_classical is not None:
    plot_route(axes[0], xc, yc, x_classical, classical_cost, N_NODES, "Classical (CPLEX)")
else:
    axes[0].text(0.5, 0.5, "CPLEX not available", ha='center', va='center',
                 transform=axes[0].transAxes, fontsize=12)
    axes[0].set_title("Classical (CPLEX) — N/A")

plot_route(axes[1], xc, yc, x_quantum, quantum_cost, N_NODES, "Quantum (QAOA + SPSA)")

plt.tight_layout()
plt.savefig("outputs/baseline_comparison.png", dpi=150, bbox_inches='tight')
plt.close()
print("\n[5] Saved outputs/baseline_comparison.png")

# Individual route plots (legacy compatibility)
if x_classical is not None:
    fig2, ax2 = plt.subplots(figsize=(6, 5))
    plot_route(ax2, xc, yc, x_classical, classical_cost, N_NODES, "Classical (CPLEX)")
    plt.tight_layout()
    plt.savefig("outputs/cplex_route.png", dpi=150, bbox_inches='tight')
    plt.close()

fig3, ax3 = plt.subplots(figsize=(6, 5))
plot_route(ax3, xc, yc, x_quantum, quantum_cost, N_NODES, "Quantum QAOA")
plt.tight_layout()
plt.savefig("outputs/qaoa_route.png", dpi=150, bbox_inches='tight')
plt.close()


# ══════════════════════════════════════════════════════════════════════════════
# 6. SAVE RESULTS JSON
# ══════════════════════════════════════════════════════════════════════════════

results = {
    "problem": {
        "n_nodes": N_NODES,
        "k_vehicles": K_VEHICLES,
        "n_qubits": N_NODES * (N_NODES - 1),
        "nisq_safe": N_NODES * (N_NODES - 1) <= 20
    },
    "classical_cplex": {
        "available": CPLEX_AVAILABLE,
        "cost": float(classical_cost),
        "route_vector": [float(v) for v in x_classical] if x_classical is not None else [],
        "z_vector": [float(v) for v in z_classical] if z_classical is not None else []
    },
    "quantum_qaoa": {
        "cost": float(quantum_cost),
        "route_vector": [float(v) for v in x_quantum],
        "solution_bitstring": quantum_solution.astype(int).tolist()
    },
    "comparison": {
        "cost_gap": float(quantum_cost - classical_cost) if CPLEX_AVAILABLE else None,
        "cost_gap_pct": float(
            100 * (quantum_cost - classical_cost) / max(abs(classical_cost), 1e-9)
        ) if CPLEX_AVAILABLE else None
    }
}

with open("outputs/baseline_routing_results.json", "w") as f:
    json.dump(results, f, indent=4)

print("    Saved outputs/baseline_routing_results.json")


# ══════════════════════════════════════════════════════════════════════════════
# 7. SUMMARY
# ══════════════════════════════════════════════════════════════════════════════

print("\n" + "=" * 65)
print("  BASELINE BENCHMARK SUMMARY")
print("=" * 65)
if CPLEX_AVAILABLE:
    gap = quantum_cost - classical_cost
    gap_pct = 100 * gap / max(abs(classical_cost), 1e-9)
    print(f"  CPLEX  (classical optimal) cost : {classical_cost:.4f}")
    print(f"  QAOA   (quantum approx.)   cost : {quantum_cost:.4f}")
    print(f"  Optimality gap                  : {gap:.4f}  ({gap_pct:.1f}%)")
else:
    print(f"  QAOA cost : {quantum_cost:.4f}  (no CPLEX baseline)")
print(f"  Qubits used : {N_NODES * (N_NODES - 1)}  (NISQ-safe ≤ 20)")
print("=" * 65)
print("\n  Outputs:")
print("    outputs/baseline_comparison.png")
print("    outputs/cplex_route.png")
print("    outputs/qaoa_route.png")
print("    outputs/baseline_routing_results.json")
print("\n  NEXT: Run cluster_scaler.py → qaoa_solver.py → main_pipeline.py")
print("=" * 65 + "\n")
