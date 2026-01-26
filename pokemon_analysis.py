# ==========================================
# Pokemon Strategic Gaming Analytics Script
# Requirements: pandas, numpy, matplotlib, scikit-learn, openpyxl
# Input files: pokemon.xlsx, moves.xlsx, evolution_trees.xlsx
# Output files: ridge_scatter.png, rf_importance.png, dbscan_clusters.png,
#               pca_scatter.png, gbr_residuals.png
# ==========================================

import pandas as pd
import numpy as np
import matplotlib.pyplot as plt
from sklearn.linear_model import Ridge
from sklearn.ensemble import RandomForestClassifier, GradientBoostingRegressor
from sklearn.cluster import DBSCAN
from sklearn.decomposition import PCA
from sklearn.preprocessing import StandardScaler
from sklearn.model_selection import train_test_split
from sklearn.metrics import mean_squared_error, f1_score, r2_score

# ---------- Load Excel Files ----------
pokemon_df = pd.read_excel('pokemon.xlsx')
moves_df = pd.read_excel('moves.xlsx')
evolution_df = pd.read_excel('evolution_trees.xlsx')

# ---------- Protocol 1: Ridge Regression for Base Experience ----------
print("=" * 60)
print("PROTOCOL 1: Ridge Regression - Base Experience Prediction")
print("=" * 60)

# Remove rows where base_experience OR attack OR defense OR special_attack is null
ridge_cols = ['base_experience', 'attack', 'defense', 'special_attack']
ridge_data = pokemon_df.dropna(subset=ridge_cols)

X_ridge = ridge_data[['attack', 'defense', 'special_attack']]
y_ridge = ridge_data['base_experience']

# 70/30 split
X_train_r, X_test_r, y_train_r, y_test_r = train_test_split(
    X_ridge, y_ridge, test_size=0.30, random_state=42
)

# Ridge regression with alpha=1.0, solver='auto', random_state=42
ridge_model = Ridge(alpha=1.0, solver='auto', random_state=42)
ridge_model.fit(X_train_r, y_train_r)

# Predict and compute MSE
y_pred_ridge = ridge_model.predict(X_test_r)
mse_ridge = mean_squared_error(y_test_r, y_pred_ridge)
print(f"Mean Squared Error: {round(mse_ridge, 2)}")

# Scatter plot: actual vs predicted
plt.figure(figsize=(8, 6))
plt.scatter(y_test_r, y_pred_ridge, alpha=0.5, edgecolors='k')
plt.xlabel('Genuine Base Experience')
plt.ylabel('Forecasted Base Experience')
plt.title('Ridge Regression: Prediction Fidelity')
plt.plot([y_test_r.min(), y_test_r.max()], [y_test_r.min(), y_test_r.max()], 'r--', lw=2)
plt.tight_layout()
plt.savefig('ridge_scatter.png', dpi=150)
plt.close()

# ---------- Protocol 2: Random Forest Classification for Primary Type ----------
print("\n" + "=" * 60)
print("PROTOCOL 2: Random Forest Classifier - Primary Type")
print("=" * 60)

# Remove rows where primary_type OR hp OR attack OR defense OR speed is null
rf_cols = ['primary_type', 'hp', 'attack', 'defense', 'speed']
rf_data = pokemon_df.dropna(subset=rf_cols)

X_rf = rf_data[['hp', 'attack', 'defense', 'speed']]
y_rf = rf_data['primary_type']

# 75/25 stratified split
X_train_rf, X_test_rf, y_train_rf, y_test_rf = train_test_split(
    X_rf, y_rf, test_size=0.25, stratify=y_rf, random_state=42
)

# RandomForestClassifier
rf_model = RandomForestClassifier(
    n_estimators=100, max_depth=10, min_samples_split=5, random_state=42
)
rf_model.fit(X_train_rf, y_train_rf)

# Predict and compute macro F1 score
y_pred_rf = rf_model.predict(X_test_rf)
macro_f1 = f1_score(y_test_rf, y_pred_rf, average='macro')
print(f"Macro F1 Score: {round(macro_f1, 4)}")

# Bar chart: feature importances
feature_names = ['hp', 'attack', 'defense', 'speed']
importances = rf_model.feature_importances_

plt.figure(figsize=(8, 6))
plt.bar(feature_names, importances, color='steelblue', edgecolor='black')
plt.xlabel('Predictor Identifiers')
plt.ylabel('Importance Magnitudes')
plt.title('Random Forest: Predictor Influence')
plt.tight_layout()
plt.savefig('rf_importance.png', dpi=150)
plt.close()

# ---------- Protocol 3: DBSCAN Clustering for Move Potency ----------
print("\n" + "=" * 60)
print("PROTOCOL 3: DBSCAN Clustering - Technique Potency")
print("=" * 60)

# Remove rows where power OR accuracy is null
dbscan_cols = ['power', 'accuracy']
dbscan_data = moves_df.dropna(subset=dbscan_cols)

# Fit StandardScaler on filtered data
scaler_dbscan = StandardScaler()
dbscan_scaled = scaler_dbscan.fit_transform(dbscan_data[['power', 'accuracy']])

# DBSCAN clustering
dbscan_model = DBSCAN(eps=0.5, min_samples=5, metric='euclidean')
dbscan_labels = dbscan_model.fit_predict(dbscan_scaled)

# Count distinct clusters excluding -1
unique_labels = set(dbscan_labels)
n_clusters = len([l for l in unique_labels if l != -1])
print(f"Number of Clusters (excluding noise): {n_clusters}")

# Scatter plot with cluster colors
plt.figure(figsize=(8, 6))
scatter = plt.scatter(dbscan_data['power'], dbscan_data['accuracy'],
                      c=dbscan_labels, cmap='viridis', alpha=0.6, edgecolors='k')
plt.colorbar(scatter, label='Cluster Assignment')
plt.xlabel('Power')
plt.ylabel('Accuracy')
plt.title('DBSCAN: Technique Potency Clusters')
plt.tight_layout()
plt.savefig('dbscan_clusters.png', dpi=150)
plt.close()

# ---------- Protocol 4: PCA on Pokemon Stats ----------
print("\n" + "=" * 60)
print("PROTOCOL 4: PCA - Capability Score Decomposition")
print("=" * 60)

# Remove rows where any of hp, attack, defense, special_attack, special_defense, speed is null
pca_cols = ['hp', 'attack', 'defense', 'special_attack', 'special_defense', 'speed']
pca_data = pokemon_df.dropna(subset=pca_cols)

# Fit StandardScaler on filtered data
scaler_pca = StandardScaler()
pca_scaled = scaler_pca.fit_transform(pca_data[pca_cols])

# PCA with 2 components
pca_model = PCA(n_components=2, random_state=42)
pca_transformed = pca_model.fit_transform(pca_scaled)

# Sum of explained variance ratio as percentage
explained_var_sum = sum(pca_model.explained_variance_ratio_) * 100
print(f"Explained Variance (2 components): {round(explained_var_sum, 2)}%")

# Scatter plot: PC1 vs PC2
plt.figure(figsize=(8, 6))
plt.scatter(pca_transformed[:, 0], pca_transformed[:, 1], alpha=0.5, edgecolors='k')
plt.xlabel('First Principal Component')
plt.ylabel('Second Principal Component')
plt.title('PCA: Orthogonal Projection of Pokemon Stats')
plt.tight_layout()
plt.savefig('pca_scatter.png', dpi=150)
plt.close()

# ---------- Protocol 5: Gradient Boosting Regression for PP ----------
print("\n" + "=" * 60)
print("PROTOCOL 5: Gradient Boosting Regressor - PP Prediction")
print("=" * 60)

# Remove rows where pp OR power OR accuracy is null
gbr_cols = ['pp', 'power', 'accuracy']
gbr_data = moves_df.dropna(subset=gbr_cols)

X_gbr = gbr_data[['power', 'accuracy']]
y_gbr = gbr_data['pp']

# 80/20 split
X_train_gbr, X_test_gbr, y_train_gbr, y_test_gbr = train_test_split(
    X_gbr, y_gbr, test_size=0.20, random_state=42
)

# GradientBoostingRegressor
gbr_model = GradientBoostingRegressor(
    n_estimators=100, learning_rate=0.1, max_depth=3, random_state=42
)
gbr_model.fit(X_train_gbr, y_train_gbr)

# Predict and compute R-squared
y_pred_gbr = gbr_model.predict(X_test_gbr)
r2_gbr = r2_score(y_test_gbr, y_pred_gbr)
print(f"R-squared Score: {round(r2_gbr, 4)}")

# Residual plot: predicted vs residuals
residuals = y_test_gbr - y_pred_gbr
plt.figure(figsize=(8, 6))
plt.scatter(y_pred_gbr, residuals, alpha=0.5, edgecolors='k')
plt.axhline(y=0, color='r', linestyle='--', lw=2)
plt.xlabel('Predicted PP Values')
plt.ylabel('Residuals (Actual - Predicted)')
plt.title('Gradient Boosting: Residual Diagnostic')
plt.tight_layout()
plt.savefig('gbr_residuals.png', dpi=150)
plt.close()

# ---------- Protocol 6: Dominant Elemental Type ----------
print("\n" + "=" * 60)
print("PROTOCOL 6: Dominant Elemental Type in Moves")
print("=" * 60)

# Count total non-null type rows
type_data = moves_df[moves_df['type'].notna()]['type']
total_type_rows = len(type_data)

# Count frequency of each type
type_counts = type_data.value_counts()

# Find maximum frequency
max_freq = type_counts.max()

# Get types with max frequency, select alphabetically first
top_types = type_counts[type_counts == max_freq].index.tolist()
dominant_type = sorted([t.lower() for t in top_types])[0]
print(f"Dominant Type: {dominant_type}")

# ---------- Protocol 7: Evolutionary Branching Intensity ----------
print("\n" + "=" * 60)
print("PROTOCOL 7: Evolutionary Branching Coefficient")
print("=" * 60)

# Filter where is_root == 1, count distinct id values
root_data = evolution_df[evolution_df['is_root'] == 1]
family_count = root_data['id'].nunique()

# Filter where is_final_stage == 1, count total rows
final_data = evolution_df[evolution_df['is_final_stage'] == 1]
final_count = len(final_data)

# Compute branching coefficient
branching_coefficient = final_count / family_count
print(f"Branching Coefficient: {round(branching_coefficient, 3)}")

# ---------- Protocol 8: Cross-Dataset Power Magnitude Hierarchy ----------
print("\n" + "=" * 60)
print("PROTOCOL 8: Cross-Dataset Power Comparison")
print("=" * 60)

# Mean attack from pokemon.xlsx (non-null only)
mean_attack = pokemon_df['attack'].dropna().mean()

# Mean power from moves.xlsx (non-null only)
mean_power = moves_df['power'].dropna().mean()

# Compare
if mean_attack > mean_power:
    power_hierarchy = "pokemon"
else:
    power_hierarchy = "moves"

print(f"Mean Attack (Pokemon): {round(mean_attack, 2)}")
print(f"Mean Power (Moves): {round(mean_power, 2)}")
print(f"Power Hierarchy Winner: {power_hierarchy}")

# ---------- Strategic Insight ----------
print("\n" + "=" * 60)
print("STRATEGIC INSIGHT")
print("=" * 60)

# Answer for open-ended question integrating clustering and evolutionary complexity
strategic_insight = (
    "Optimal competitive team composition leverages the DBSCAN-identified move potency clusters "
    f"({n_clusters} distinct technique groups) by selecting Pokemon from evolutionary families "
    f"with high branching coefficients ({round(branching_coefficient, 3)}) to maximize tactical "
    "versatility through diverse final-stage options with complementary move type coverage."
)
print(strategic_insight)

# ---------- Final Summary Output ----------
print("\n" + "=" * 60)
print("KEY OUTPUTS SUMMARY")
print("=" * 60)
print(f"Protocol 1 - Ridge MSE: {round(mse_ridge, 2)}")
print(f"Protocol 2 - Macro F1: {round(macro_f1, 4)}")
print(f"Protocol 3 - DBSCAN Clusters: {n_clusters}")
print(f"Protocol 4 - PCA Variance: {round(explained_var_sum, 2)}%")
print(f"Protocol 5 - GBR R-squared: {round(r2_gbr, 4)}")
print(f"Protocol 6 - Dominant Type: {dominant_type}")
print(f"Protocol 7 - Branching Coefficient: {round(branching_coefficient, 3)}")
print(f"Protocol 8 - Power Hierarchy: {power_hierarchy}")
