import pandas as pd
import numpy as np
from scipy.spatial.distance import pdist, squareform
from sklearn.manifold import MDS
from sklearn.preprocessing import LabelEncoder

# Load data
customers = pd.read_csv('customers.csv')

# Encode state field
customers_valid = customers[customers['state'].notna()].copy()
label_encoder = LabelEncoder()
customers_valid['state_encoded'] = label_encoder.fit_transform(customers_valid['state'])

# Form pairwise distance matrix
state_encoded_values = customers_valid['state_encoded'].values.reshape(-1, 1)
distance_matrix = squareform(pdist(state_encoded_values, metric='euclidean'))

# Test with different n_init values
print("Testing with random_state=42 and different n_init values:")
for n_init in [1, 4, 10]:
    mds = MDS(n_components=2, dissimilarity='precomputed', random_state=42, n_init=n_init)
    embedding = mds.fit_transform(distance_matrix)
    print(f"n_init={n_init}: MDS Stress = {mds.stress_:.4f}")

print("\nTesting with different max_iter values:")
for max_iter in [300, 500, 1000]:
    mds = MDS(n_components=2, dissimilarity='precomputed', random_state=42, max_iter=max_iter)
    embedding = mds.fit_transform(distance_matrix)
    print(f"max_iter={max_iter}: MDS Stress = {mds.stress_:.4f}")

print("\nTesting with normalized_stress parameter:")
for norm_stress in [True, False]:
    try:
        mds = MDS(n_components=2, dissimilarity='precomputed', random_state=42, normalized_stress=norm_stress)
        embedding = mds.fit_transform(distance_matrix)
        print(f"normalized_stress={norm_stress}: MDS Stress = {mds.stress_:.4f}")
    except Exception as e:
        print(f"normalized_stress={norm_stress}: Error - {e}")
