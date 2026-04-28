"""
Create a test distance matrix for qubo_builder.py verification
"""
import numpy as np
import os

# Create distance_matrices directory
os.makedirs("distance_matrices", exist_ok=True)

# Create a simple 4-node distance matrix (depot + 3 delivery nodes)
# This gives us 4 * 4 = 16 qubits (NISQ-safe!)
n_nodes = 4

# Generate random coordinates
np.random.seed(42)
coords = np.random.uniform(0, 100, size=(n_nodes, 2))

# Calculate Euclidean distance matrix
distance_matrix = np.zeros((n_nodes, n_nodes))
for i in range(n_nodes):
    for j in range(n_nodes):
        if i != j:
            distance_matrix[i, j] = np.linalg.norm(coords[i] - coords[j])

# Save as cluster_00.npy
np.save("distance_matrices/cluster_00.npy", distance_matrix)

print(f"Created test distance matrix: distance_matrices/cluster_00.npy")
print(f"Shape: {distance_matrix.shape}")
print(f"Expected qubits: {n_nodes}² = {n_nodes ** 2}")
print("\nDistance matrix:")
print(distance_matrix)
