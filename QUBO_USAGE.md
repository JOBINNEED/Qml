# QUBO Builder Usage Guide

## Overview
`qubo_builder.py` converts distance matrices from `cluster_scaler.py` into Qiskit-compatible QUBO formulations and Ising Hamiltonians for QAOA execution.

## Quick Start

### 1. Generate Distance Matrices
First, run the cluster scaler to generate distance matrices:
```bash
python cluster_scaler.py
```
This creates `distance_matrices/cluster_XX.npy` files.

### 2. Build QUBO and Hamiltonian
Run the QUBO builder on a specific cluster:
```bash
python qubo_builder.py
```

By default, it tests with `distance_matrices/cluster_00.npy`.

## Using the VRPQuboBuilder Class

### Basic Usage
```python
from qubo_builder import VRPQuboBuilder

# Initialize with a distance matrix
builder = VRPQuboBuilder("distance_matrices/cluster_00.npy", penalty_weight=1000.0)

# Build the QuadraticProgram
qp = builder.build_quadratic_program()

# Convert to Ising Hamiltonian
operator, offset, qubo = builder.convert_to_ising(qp)

# Print summary
builder.print_hamiltonian_summary(operator, offset)
```

### Processing Multiple Clusters
```python
import os
import glob

# Process all clusters
for matrix_file in sorted(glob.glob("distance_matrices/cluster_*.npy")):
    print(f"\nProcessing {matrix_file}...")
    builder = VRPQuboBuilder(matrix_file)
    qp = builder.build_quadratic_program()
    operator, offset, qubo = builder.convert_to_ising(qp)
    
    # Use operator in your QAOA circuit
    # ... your QAOA code here ...
```

## QUBO Formulation Details

### Variable Encoding
- Binary variables: `x[i][t]` where i = city, t = time step
- `x[i][t] = 1` means city i is visited at time step t
- Total variables: n² (n cities × n time steps)
- **Qubit count = n²**

### NISQ Compatibility
- **NISQ limit**: 20 qubits
- **Recommended**: n ≤ 4 nodes (4² = 16 qubits)
- **Maximum**: n ≤ 4 delivery nodes + 1 depot = 5 total nodes (25 qubits, exceeds limit)

### Constraints
1. **Each city visited exactly once**: Σₜ x[i][t] = 1 for all i
2. **Each time step visits one city**: Σᵢ x[i][t] = 1 for all t

Both constraints are enforced via penalty terms with weight `penalty_weight`.

### Objective Function
Minimize total travel distance:
```
Σᵢⱼₜ D[i][j] × x[i][t] × x[j][(t+1) mod n]
```

## Output

The script provides:
1. **QuadraticProgram**: Qiskit optimization problem object
2. **Ising Hamiltonian**: Pauli-Z operator representation
3. **Offset**: Constant term for energy calculation
4. **Qubit count**: Number of qubits required
5. **NISQ safety check**: Whether the problem fits within 20 qubits

## Integration with QAOA

The Ising Hamiltonian can be directly fed into QAOA circuits:

```python
from qiskit.algorithms import QAOA
from qiskit.algorithms.optimizers import COBYLA
from qiskit.primitives import Sampler

# Build Hamiltonian
builder = VRPQuboBuilder("distance_matrices/cluster_00.npy")
qp = builder.build_quadratic_program()
operator, offset, qubo = builder.convert_to_ising(qp)

# Setup QAOA
optimizer = COBYLA(maxiter=100)
qaoa = QAOA(sampler=Sampler(), optimizer=optimizer, reps=1)

# Run QAOA
result = qaoa.compute_minimum_eigenvalue(operator)

# Interpret results
print(f"Optimal value: {result.eigenvalue + offset}")
print(f"Optimal parameters: {result.optimal_parameters}")
```

## Parameters

### VRPQuboBuilder Constructor
- `distance_matrix_path` (str): Path to .npy distance matrix file
- `penalty_weight` (float, default=1000.0): Constraint penalty coefficient
  - Higher values enforce constraints more strictly
  - Lower values may allow constraint violations
  - Recommended: 1000.0 for most cases

## Notes

1. **Index 0 is always the depot** in distance matrices
2. The penalty weight should be large enough to prevent constraint violations
3. For clusters with n > 4 nodes, consider re-running `cluster_scaler.py` with `MAX_CLUSTER = 3`
4. The Hamiltonian includes both objective and constraint penalty terms

## Troubleshooting

### "distance_matrices/cluster_00.npy not found"
Run `cluster_scaler.py` first to generate distance matrices.

### "Exceeds NISQ limit"
Reduce cluster size in `cluster_scaler.py` by setting `MAX_CLUSTER = 3` or lower.

### Large penalty terms dominating
Adjust `penalty_weight` parameter. Try values between 100 and 10000.

## References

- Based on VRP_Challenge.py QUBO formulation
- Qiskit Optimization: https://qiskit.org/documentation/optimization/
- QAOA Tutorial: https://qiskit.org/textbook/ch-applications/qaoa.html
