"""
qubo_builder.py
===============
QUBO Formulation Bridge for QAOA-based VRP.

PURPOSE:
    Converts a distance matrix (from cluster_scaler.py) into a Qiskit
    QuadraticProgram, then into an Ising Hamiltonian ready for QAOA.

ENCODING (position-based TSP):
    Binary variable x[i][t] = 1  iff  city i is visited at time step t.
    Total variables : n²   (n cities × n time steps)
    Qubit count     : n²

    For NISQ safety: n² ≤ 20  →  n ≤ 4 nodes (depot + 3 delivery)

OBJECTIVE:
    Minimize total travel distance:
        Σ_{i,j,t}  D[i][j] · x[i][t] · x[j][(t+1) mod n]

CONSTRAINTS (penalty method):
    1. Each city visited exactly once : Σ_t x[i][t] = 1  ∀ i
    2. Each time step has one city    : Σ_i x[i][t] = 1  ∀ t

    Both are encoded as  A · (Σ x - 1)²  added to the objective.
    Expanding:  A·(Σ x)² - 2A·(Σ x) + A
              = A·Σ_{p≠q} x_p·x_q  +  A·Σ_p x_p²  - 2A·Σ_p x_p  + A
    Since x is binary: x_p² = x_p, so the net linear contribution is
              A·x_p - 2A·x_p = -A·x_p  per variable in the group.
"""

import numpy as np
from qiskit_optimization import QuadraticProgram
from qiskit_optimization.converters import QuadraticProgramToQubo
import os


class VRPQuboBuilder:
    """
    Builds a QUBO / Ising Hamiltonian for a VRP sub-cluster.

    Parameters
    ----------
    distance_matrix_path : str
        Path to a .npy file containing an n×n Euclidean distance matrix.
        Index 0 is always the depot.
    penalty_weight : float
        Penalty coefficient A for constraint violations.
        Rule of thumb: A > max(D) * n  to dominate the objective.
    """

    def __init__(self, distance_matrix_path: str, penalty_weight: float = 1000.0):
        self.distance_matrix = np.load(distance_matrix_path)
        self.n_nodes = self.distance_matrix.shape[0]
        self.penalty = penalty_weight
        self._path = distance_matrix_path

        print(f"\n{'='*60}")
        print(f"  VRP QUBO BUILDER")
        print(f"{'='*60}")
        print(f"  File    : {distance_matrix_path}")
        print(f"  Nodes   : {self.n_nodes}  (depot = index 0)")
        print(f"  Qubits  : {self.n_nodes**2}  (n² position encoding)")
        print(f"  Penalty : {self.penalty}")
        nisq = "YES" if self.n_nodes ** 2 <= 20 else "NO — reduce cluster size"
        print(f"  NISQ-safe: {nisq}")

    # ──────────────────────────────────────────────────────────────────────────
    def build_quadratic_program(self) -> QuadraticProgram:
        """
        Construct the Qiskit QuadraticProgram for TSP/VRP.

        Returns
        -------
        QuadraticProgram
        """
        n = self.n_nodes
        A = self.penalty
        D = self.distance_matrix

        qp = QuadraticProgram(name="VRP_Cluster")

        # Declare binary variables x_{city}_{timestep}
        for i in range(n):
            for t in range(n):
                qp.binary_var(name=f"x_{i}_{t}")

        # ── Objective: travel cost ────────────────────────────────────────────
        # Σ_{i,j,t} D[i][j] · x[i][t] · x[j][(t+1) mod n]
        linear_obj    = {}
        quadratic_obj = {}

        for t in range(n):
            t_next = (t + 1) % n
            for i in range(n):
                for j in range(n):
                    if i != j and D[i][j] > 0:
                        vi = f"x_{i}_{t}"
                        vj = f"x_{j}_{t_next}"
                        key = (vi, vj)
                        quadratic_obj[key] = quadratic_obj.get(key, 0.0) + D[i][j]

        # ── Constraint 1: each city visited exactly once ──────────────────────
        # A · Σ_i (Σ_t x[i][t] - 1)²
        for i in range(n):
            group = [f"x_{i}_{t}" for t in range(n)]
            # Linear: net -A per variable (from x² - 2x = -x for binary)
            for v in group:
                linear_obj[v] = linear_obj.get(v, 0.0) - A
            # Quadratic cross terms: +2A for each pair
            for a in range(len(group)):
                for b in range(a + 1, len(group)):
                    key = (group[a], group[b])
                    quadratic_obj[key] = quadratic_obj.get(key, 0.0) + 2 * A

        # ── Constraint 2: each time step has exactly one city ─────────────────
        # A · Σ_t (Σ_i x[i][t] - 1)²
        for t in range(n):
            group = [f"x_{i}_{t}" for i in range(n)]
            for v in group:
                linear_obj[v] = linear_obj.get(v, 0.0) - A
            for a in range(len(group)):
                for b in range(a + 1, len(group)):
                    key = (group[a], group[b])
                    quadratic_obj[key] = quadratic_obj.get(key, 0.0) + 2 * A

        # Constant from both constraints: 2 · A · n  (one A·1 per constraint group)
        constant = 2 * A * n

        qp.minimize(constant=constant, linear=linear_obj, quadratic=quadratic_obj)

        print(f"\n  QuadraticProgram built:")
        print(f"    Variables  : {qp.get_num_vars()}")
        print(f"    Linear     : {len(linear_obj)}")
        print(f"    Quadratic  : {len(quadratic_obj)}")
        return qp

    # ──────────────────────────────────────────────────────────────────────────
    def convert_to_ising(self, qp: QuadraticProgram):
        """
        Convert QuadraticProgram → QUBO → Ising Hamiltonian.

        Returns
        -------
        operator : SparsePauliOp
            Ising cost Hamiltonian (Pauli-Z terms).
        offset : float
            Constant energy offset.
        qubo : QuadraticProgram
            Intermediate QUBO form (for inspection).
        """
        print(f"\n  Converting to Ising Hamiltonian...")
        converter = QuadraticProgramToQubo()
        qubo = converter.convert(qp)
        operator, offset = qubo.to_ising()

        print(f"    QUBO vars      : {qubo.get_num_vars()}")
        print(f"    Ising qubits   : {operator.num_qubits}")
        print(f"    Pauli terms    : {len(operator)}")
        print(f"    Energy offset  : {offset:.4f}")
        return operator, offset, qubo

    # ──────────────────────────────────────────────────────────────────────────
    def print_hamiltonian_summary(self, operator, offset):
        n = self.n_nodes
        n_q = operator.num_qubits
        print(f"\n{'='*60}")
        print(f"  HAMILTONIAN SUMMARY")
        print(f"{'='*60}")
        print(f"  Qubits required : {n_q}  (expected n²={n**2})")
        print(f"  Pauli terms     : {len(operator)}")
        print(f"  Energy offset   : {offset:.4f}")
        status = "✓ NISQ-SAFE" if n_q <= 20 else "✗ EXCEEDS 20-qubit limit"
        print(f"  NISQ status     : {status}")
        print(f"\n  First 8 Pauli terms:")
        for i, (pauli, coeff) in enumerate(operator.to_list()[:8]):
            print(f"    {i+1:2d}. {pauli}  :  {coeff:.4f}")
        if len(operator) > 8:
            print(f"    ... ({len(operator)-8} more terms)")
        print(f"{'='*60}\n")


# ── Standalone test ────────────────────────────────────────────────────────────
def main():
    test_path = "distance_matrices/cluster_00.npy"
    if not os.path.exists(test_path):
        print(f"ERROR: {test_path} not found. Run cluster_scaler.py first.")
        return

    builder = VRPQuboBuilder(test_path, penalty_weight=1000.0)
    qp      = builder.build_quadratic_program()
    op, off, qubo = builder.convert_to_ising(qp)
    builder.print_hamiltonian_summary(op, off)

    print("QUBO builder test complete.")
    print(f"Feed this {op.num_qubits}-qubit Hamiltonian into qaoa_solver.py")


if __name__ == "__main__":
    main()
