"""
This script extracts the EXACT MDS implementation from transaction_analysis.py
and runs it 10 times to verify the stress value consistency.
"""
import pandas as pd
import numpy as np
from scipy.spatial.distance import pdist, squareform
from sklearn.manifold import MDS
from sklearn.preprocessing import LabelEncoder
import sklearn

print("="*60)
print("MDS STRESS VALUE VERIFICATION")
print("="*60)
print(f"sklearn version: {sklearn.__version__}")
print()

# Load data (EXACT copy from transaction_analysis.py)
customers = pd.read_csv('customers.csv')

# Encode state field (EXACT copy from transaction_analysis.py)
customers_valid = customers[customers['state'].notna()].copy()
label_encoder = LabelEncoder()
customers_valid['state_encoded'] = label_encoder.fit_transform(customers_valid['state'])

# Form pairwise distance matrix (EXACT copy from transaction_analysis.py)
state_encoded_values = customers_valid['state_encoded'].values.reshape(-1, 1)
distance_matrix = squareform(pdist(state_encoded_values, metric='euclidean'))

print(f"Dataset info:")
print(f"  - Total customers: {len(customers_valid)}")
print(f"  - Unique states: {customers_valid['state'].nunique()} {sorted(customers_valid['state'].unique())}")
print(f"  - Distance matrix shape: {distance_matrix.shape}")
print()

# Apply classical MDS (EXACT copy from transaction_analysis.py)
print("Running MDS with random_state=42 (as specified in code):")
print()

results = []
for i in range(10):
    mds = MDS(n_components=2, dissimilarity='precomputed', random_state=42)
    embedding = mds.fit_transform(distance_matrix)
    stress_value = mds.stress_
    results.append(stress_value)
    print(f"Run {i+1}: Stress = {stress_value:.4f}")

print()
print("="*60)
print("VERIFICATION RESULTS:")
print("="*60)
print(f"All values identical: {len(set(results)) == 1}")
print(f"Verified MDS Stress Value: {results[0]:.4f}")
print()
print(f"User reported value: 1.9408")
print(f"Difference: {abs(results[0] - 1.9408):.4f}")
print()

if abs(results[0] - 1.9408) < 0.01:
    print("✓ Values match within tolerance")
elif abs(results[0] - 2.3870) < 0.01:
    print("✓ Value matches 2.3870 (previously reported)")
    print("⚠ User's value 1.9408 differs - likely due to:")
    print("  - Different sklearn version")
    print("  - Different random initialization")
    print("  - Code modification")
else:
    print("⚠ Neither value matches - investigation needed")
