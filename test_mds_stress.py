import pandas as pd
import numpy as np
from scipy.spatial.distance import pdist, squareform
from sklearn.manifold import MDS
from sklearn.preprocessing import LabelEncoder
import sklearn

print(f"sklearn version: {sklearn.__version__}")

# Load data
customers = pd.read_csv('customers.csv')

# Encode state field
customers_valid = customers[customers['state'].notna()].copy()
label_encoder = LabelEncoder()
customers_valid['state_encoded'] = label_encoder.fit_transform(customers_valid['state'])

print(f"Number of customers: {len(customers_valid)}")
print(f"Number of unique states: {customers_valid['state'].nunique()}")
print(f"Unique states: {sorted(customers_valid['state'].unique())}")

# Form pairwise distance matrix
state_encoded_values = customers_valid['state_encoded'].values.reshape(-1, 1)
distance_matrix = squareform(pdist(state_encoded_values, metric='euclidean'))

print(f"Distance matrix shape: {distance_matrix.shape}")
print(f"Distance matrix min/max: {distance_matrix.min():.4f} / {distance_matrix.max():.4f}")

# Apply classical MDS with different random states
for rs in [42, 0, 1, 2, None]:
    mds = MDS(n_components=2, dissimilarity='precomputed', random_state=rs)
    embedding = mds.fit_transform(distance_matrix)
    print(f"MDS Stress (random_state={rs}): {mds.stress_:.4f}")
