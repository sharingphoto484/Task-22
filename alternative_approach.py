import pandas as pd
import numpy as np
from sklearn.linear_model import LogisticRegression
from sklearn.decomposition import PCA
from sklearn.metrics import roc_auc_score
import hashlib

print("="*70)
print("ALTERNATIVE APPROACH: Using ONLY Regular_Season.csv & Playoffs.csv")
print("="*70)

# Load ONLY Regular_Season.csv and Playoffs.csv (not nba.csv for modeling)
regular_raw = pd.read_csv('Regular_Season.csv')
playoffs_raw = pd.read_csv('Playoffs.csv')

print(f"\nRegular_Season.csv: {len(regular_raw)} rows")
print(f"Playoffs.csv: {len(playoffs_raw)} rows")

# Normalize Season_type
regular_raw['Season_type'] = 'Regular_Season'
playoffs_raw['Season_type'] = 'Playoffs'

# Find common seasons
years_regular = set(regular_raw['year'].unique())
years_playoffs = set(playoffs_raw['year'].unique())

# For roster continuity, we need nba.csv
nba_raw = pd.read_csv('nba.csv')
nba_raw['Season_type'] = nba_raw['Season_type'].replace('Regular%20Season', 'Regular_Season')
years_nba = set(nba_raw['year'].unique())

common_years = sorted(years_nba & years_regular & years_playoffs)
print(f"\nCommon seasons: {common_years}")

# Filter to common years
regular = regular_raw[regular_raw['year'].isin(common_years)].copy()
playoffs = playoffs_raw[playoffs_raw['year'].isin(common_years)].copy()

# Filter to non-missing
required_fields = ['year', 'PLAYER_ID', 'PLAYER', 'TEAM', 'GP', 'MIN', 'PTS']
regular = regular.dropna(subset=required_fields).copy()

print(f"\nRegular season rows after filtering: {len(regular)}")

# Create playoff participation labels
playoff_keys = set()
for _, row in playoffs.iterrows():
    if pd.notna(row.get('PLAYER_ID')) and pd.notna(row.get('TEAM')) and pd.notna(row.get('year')):
        key = (row['year'], str(row['PLAYER_ID']), row['TEAM'])
        playoff_keys.add(key)

print(f"Unique playoff appearances: {len(playoff_keys)}")

regular['made_playoffs'] = regular.apply(
    lambda row: 1 if (row['year'], str(row['PLAYER_ID']), row['TEAM']) in playoff_keys else 0,
    axis=1
)

print(f"Playoff participation rate: {regular['made_playoffs'].mean():.4f}")

# Standardize within year
def standardize_within_year(df, col):
    return df.groupby('year')[col].transform(
        lambda x: (x - x.mean()) / x.std() if x.std() > 0 else 0
    )

regular['GP_std'] = standardize_within_year(regular, 'GP')
regular['MIN_std'] = standardize_within_year(regular, 'MIN')
regular['PTS_std'] = standardize_within_year(regular, 'PTS')

# Assign folds
def assign_fold(player_id):
    hash_obj = hashlib.md5(str(player_id).encode())
    hash_int = int(hash_obj.hexdigest(), 16)
    return hash_int % 5

regular['fold'] = regular['PLAYER_ID'].apply(assign_fold)

# Cross-validated logistic regression
X = regular[['GP_std', 'MIN_std', 'PTS_std']].values
y = regular['made_playoffs'].values
folds = regular['fold'].values

oof_preds = np.zeros(len(regular))
fold_coefficients = []

for fold_id in range(5):
    train_idx = folds != fold_id
    val_idx = folds == fold_id

    X_train, y_train = X[train_idx], y[train_idx]
    X_val = X[val_idx]

    model = LogisticRegression(max_iter=1000, random_state=42)
    model.fit(X_train, y_train)

    oof_preds[val_idx] = model.predict_proba(X_val)[:, 1]
    fold_coefficients.append(model.coef_[0])

cv_auc = roc_auc_score(y, oof_preds)
mean_coef_min = np.mean([coef[1] for coef in fold_coefficients])

# Fit on full data
model_full = LogisticRegression(max_iter=1000, random_state=42)
model_full.fit(X, y)
full_coef_min = model_full.coef_[0][1]

print(f"\nCross-Validated AUC: {cv_auc:.4f}")
print(f"Mean MIN coefficient (across folds): {mean_coef_min:.4f}")
print(f"Full data MIN coefficient: {full_coef_min:.4f}")

# PCA
pca = PCA(n_components=3)
pca_all = pca.fit_transform(X)
pc1_scores = pca_all[:, 0]
pc1_loadings = pca.components_[0]

if pc1_loadings[1] < 0:
    pc1_loadings = -pc1_loadings
    pc1_scores = -pc1_scores

variance_pct = pca.explained_variance_ratio_[0] * 100

pc1_model = LogisticRegression(max_iter=1000, random_state=42)
pc1_model.fit(pc1_scores.reshape(-1, 1), y)
pc1_preds = pc1_model.predict_proba(pc1_scores.reshape(-1, 1))[:, 1]
pc1_auc = roc_auc_score(y, pc1_preds)

print(f"\nVariance Explained by PC1: {variance_pct:.4f}%")
print(f"AUC for PC1 Score Model: {pc1_auc:.4f}")

print("\n" + "="*70)
print("COMPARISON:")
print("="*70)
print(f"Method: Using only Regular_Season.csv & Playoffs.csv")
print(f"CV AUC: {cv_auc:.4f}")
print(f"MIN coef: {mean_coef_min:.4f}")
print(f"PCA variance: {variance_pct:.4f}%")
print(f"PC1 AUC: {pc1_auc:.4f}")
