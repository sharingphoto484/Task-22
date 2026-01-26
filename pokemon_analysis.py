# ==========================================
# Pokemon Strategic Gaming Analytics Script
# Requirements: pandas, numpy, matplotlib, scikit-learn, openpyxl
# Input files: pokemon.xlsx, moves.xlsx, evolution_trees.xlsx
# Output files: ridge_scatter.png, rf_importance.png, dbscan_clusters.png,
#               pca_projection.png, gbr_residuals.png
# ==========================================

import pandas as pd
import numpy as np
import matplotlib.pyplot as plt
from sklearn.linear_model import Ridge, LinearRegression
from sklearn.ensemble import RandomForestClassifier, GradientBoostingRegressor
from sklearn.cluster import DBSCAN, KMeans
from sklearn.decomposition import PCA
from sklearn.discriminant_analysis import LinearDiscriminantAnalysis
from sklearn.preprocessing import StandardScaler, LabelEncoder
from sklearn.model_selection import train_test_split
from sklearn.metrics import mean_squared_error, f1_score, r2_score, silhouette_score
import warnings
warnings.filterwarnings('ignore')

# ---------- Load Datasets ----------
pokemon = pd.read_excel('pokemon.xlsx')
moves = pd.read_excel('moves.xlsx')
evolution_trees = pd.read_excel('evolution_trees.xlsx')

print("=" * 60)
print("POKEMON STRATEGIC GAMING ANALYTICS")
print("=" * 60)

# ---------- Protocol 1: Ridge Regression ----------
print("\n--- Protocol 1: Ridge Regression (Experience Prediction) ---")

# Filter rows where base_experience, attack, defense, special_attack are not null
ridge_cols = ['base_experience', 'attack', 'defense', 'special_attack']
ridge_data = pokemon.dropna(subset=ridge_cols).copy()

X_ridge = ridge_data[['attack', 'defense', 'special_attack']]
y_ridge = ridge_data['base_experience']

# 70/30 split
X_train_r, X_test_r, y_train_r, y_test_r = train_test_split(
    X_ridge, y_ridge, test_size=0.30, random_state=42
)

# Ridge with alpha=1.0, solver='auto', random_state=42
ridge_model = Ridge(alpha=1.0, solver='auto', random_state=42)
ridge_model.fit(X_train_r, y_train_r)
y_pred_r = ridge_model.predict(X_test_r)

mse_ridge = mean_squared_error(y_test_r, y_pred_r)
print(f"Mean Squared Error: {mse_ridge:.2f}")

# Scatter plot: Actual vs Predicted
plt.figure(figsize=(8, 6))
plt.scatter(y_test_r, y_pred_r, alpha=0.6, edgecolors='k', linewidth=0.5)
plt.xlabel('Actual Base Experience')
plt.ylabel('Predicted Base Experience')
plt.title('Ridge Regression: Actual vs Predicted Base Experience')
plt.plot([y_test_r.min(), y_test_r.max()], [y_test_r.min(), y_test_r.max()], 'r--', lw=2)
plt.tight_layout()
plt.savefig('ridge_scatter.png', dpi=150)
plt.close()

# ---------- Protocol 2: Random Forest Classification ----------
print("\n--- Protocol 2: Random Forest Classification (Type Prediction) ---")

# Filter rows where primary_type, hp, attack, defense, speed are not null
rf_cols = ['primary_type', 'hp', 'attack', 'defense', 'speed']
rf_data = pokemon.dropna(subset=rf_cols).copy()

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
y_pred_rf = rf_model.predict(X_test_rf)

macro_f1 = f1_score(y_test_rf, y_pred_rf, average='macro')
print(f"Macro F1 Score: {macro_f1:.4f}")

# Bar chart of feature importances
feature_names = ['hp', 'attack', 'defense', 'speed']
importances = rf_model.feature_importances_

plt.figure(figsize=(8, 6))
plt.bar(feature_names, importances, color=['#3498db', '#e74c3c', '#2ecc71', '#f39c12'])
plt.xlabel('Predictor')
plt.ylabel('Importance')
plt.title('Random Forest: Feature Importance for Type Prediction')
plt.tight_layout()
plt.savefig('rf_importance.png', dpi=150)
plt.close()

# ---------- Protocol 3: DBSCAN Clustering ----------
print("\n--- Protocol 3: DBSCAN Clustering (Move Potency Clusters) ---")

# Filter rows where power and accuracy are not null
dbscan_cols = ['power', 'accuracy']
dbscan_data = moves.dropna(subset=dbscan_cols).copy()

X_dbscan = dbscan_data[['power', 'accuracy']]

# Fit StandardScaler on filtered data
scaler_dbscan = StandardScaler()
X_dbscan_scaled = scaler_dbscan.fit_transform(X_dbscan)

# DBSCAN
dbscan_model = DBSCAN(eps=0.5, min_samples=5, metric='euclidean')
dbscan_labels = dbscan_model.fit_predict(X_dbscan_scaled)

# Count clusters excluding noise (-1)
n_clusters_dbscan = len(set(dbscan_labels)) - (1 if -1 in dbscan_labels else 0)
print(f"Number of Clusters (excluding noise): {n_clusters_dbscan}")

# Scatter plot colored by cluster
plt.figure(figsize=(8, 6))
scatter = plt.scatter(dbscan_data['power'], dbscan_data['accuracy'],
                      c=dbscan_labels, cmap='tab10', alpha=0.7, edgecolors='k', linewidth=0.3)
plt.xlabel('Power')
plt.ylabel('Accuracy')
plt.title('DBSCAN: Move Potency Clusters')
plt.colorbar(scatter, label='Cluster')
plt.tight_layout()
plt.savefig('dbscan_clusters.png', dpi=150)
plt.close()

# ---------- Protocol 4: PCA ----------
print("\n--- Protocol 4: PCA (Capability Score Decomposition) ---")

# Filter rows where all 6 stats are not null
pca_cols = ['hp', 'attack', 'defense', 'special_attack', 'special_defense', 'speed']
pca_data = pokemon.dropna(subset=pca_cols).copy()

X_pca = pca_data[pca_cols]

# Fit StandardScaler on filtered data
scaler_pca = StandardScaler()
X_pca_scaled = scaler_pca.fit_transform(X_pca)

# PCA with n_components=2
pca_model = PCA(n_components=2, random_state=42)
X_pca_transformed = pca_model.fit_transform(X_pca_scaled)

explained_variance_sum = sum(pca_model.explained_variance_ratio_) * 100
print(f"Explained Variance (2 components): {explained_variance_sum:.2f}%")

# Scatter plot of PC1 vs PC2
plt.figure(figsize=(8, 6))
plt.scatter(X_pca_transformed[:, 0], X_pca_transformed[:, 1], alpha=0.6, edgecolors='k', linewidth=0.3)
plt.xlabel('First Principal Component')
plt.ylabel('Second Principal Component')
plt.title('PCA: Pokemon Capability Score Projection')
plt.tight_layout()
plt.savefig('pca_projection.png', dpi=150)
plt.close()

# ---------- Protocol 5: Gradient Boosting Regression ----------
print("\n--- Protocol 5: Gradient Boosting Regression (PP Prediction) ---")

# Filter rows where pp, power, accuracy are not null
gbr_cols = ['pp', 'power', 'accuracy']
gbr_data = moves.dropna(subset=gbr_cols).copy()

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
y_pred_gbr = gbr_model.predict(X_test_gbr)

r2_gbr = r2_score(y_test_gbr, y_pred_gbr)
print(f"R-squared Score: {r2_gbr:.4f}")

# Residual plot
residuals = y_test_gbr - y_pred_gbr
plt.figure(figsize=(8, 6))
plt.scatter(y_pred_gbr, residuals, alpha=0.6, edgecolors='k', linewidth=0.3)
plt.axhline(y=0, color='r', linestyle='--', lw=2)
plt.xlabel('Predicted PP')
plt.ylabel('Residuals (Actual - Predicted)')
plt.title('Gradient Boosting: Residual Diagnostic Plot')
plt.tight_layout()
plt.savefig('gbr_residuals.png', dpi=150)
plt.close()

# ---------- Protocol 6: KMeans Clustering ----------
print("\n--- Protocol 6: KMeans Clustering (Pokemon Segmentation) ---")

# Filter rows where all 6 stats are not null (same as PCA)
kmeans_cols = ['hp', 'attack', 'defense', 'special_attack', 'special_defense', 'speed']
kmeans_data = pokemon.dropna(subset=kmeans_cols).copy()

X_kmeans = kmeans_data[kmeans_cols]

# Fit StandardScaler on filtered data
scaler_kmeans = StandardScaler()
X_kmeans_scaled = scaler_kmeans.fit_transform(X_kmeans)

# KMeans
kmeans_model = KMeans(n_clusters=5, init='k-means++', n_init=10, random_state=42)
kmeans_labels = kmeans_model.fit_predict(X_kmeans_scaled)

silhouette = silhouette_score(X_kmeans_scaled, kmeans_labels)
print(f"Silhouette Coefficient: {silhouette:.4f}")

# ---------- Protocol 7: Linear Discriminant Analysis ----------
print("\n--- Protocol 7: LDA (Type-Differentiated Dimension Reduction) ---")

# Filter rows where primary_type and all 6 stats are not null
lda_cols = ['primary_type', 'hp', 'attack', 'defense', 'special_attack', 'special_defense', 'speed']
lda_data = pokemon.dropna(subset=lda_cols).copy()

X_lda = lda_data[['hp', 'attack', 'defense', 'special_attack', 'special_defense', 'speed']]
y_lda = lda_data['primary_type']

# Fit StandardScaler on filtered data
scaler_lda = StandardScaler()
X_lda_scaled = scaler_lda.fit_transform(X_lda)

# LDA with n_components=2, solver='svd'
lda_model = LinearDiscriminantAnalysis(n_components=2, solver='svd')
X_lda_transformed = lda_model.fit_transform(X_lda_scaled, y_lda)

# Mean of first discriminant component
lda_mean_first = np.mean(X_lda_transformed[:, 0])
print(f"Mean of First Discriminant Component: {lda_mean_first:.4f}")

# ---------- Protocol 8: Cross-Dataset Power Hierarchy ----------
print("\n--- Protocol 8: Cross-Dataset Power Magnitude Comparison ---")

# Mean attack from pokemon.xlsx (only non-null attack values)
mean_attack = pokemon['attack'].dropna().mean()
print(f"Mean Attack (Pokemon): {mean_attack:.4f}")

# Mean power from moves.xlsx (only non-null power values)
mean_power = moves['power'].dropna().mean()
print(f"Mean Power (Moves): {mean_power:.4f}")

# Compare
if mean_attack > mean_power:
    power_hierarchy_result = "pokemon"
else:
    power_hierarchy_result = "moves"
print(f"Power Hierarchy Winner: {power_hierarchy_result}")

# ---------- Strategic Team Composition Analysis ----------
print("\n" + "=" * 60)
print("STRATEGIC INSIGHTS")
print("=" * 60)

# Strategic team composition insight integrating clustering and evolution
# The KMeans clustering reveals 5 distinct Pokemon archetypes based on stat distributions;
# teams should balance representatives from different clusters (tank, sweeper, balanced, etc.)
# while considering evolutionary complexity where final-stage Pokemon in multi-evolution chains
# typically exhibit higher base stats, making evolved forms preferable for competitive optimization.
strategic_insight = (
    "Strategic team composition should balance Pokemon from different KMeans clusters "
    "(representing distinct stat archetypes like tanks, sweepers, and balanced fighters) "
    "while prioritizing final-stage evolutions from complex evolutionary chains, as these "
    "typically exhibit superior base stat totals and specialized roles for competitive play optimization."
)
print(f"\nStrategic Insight: {strategic_insight}")

print("\n" + "=" * 60)
print("KEY OUTPUTS SUMMARY")
print("=" * 60)
print(f"1. Ridge MSE: {mse_ridge:.2f}")
print(f"2. RF Macro F1: {macro_f1:.4f}")
print(f"3. DBSCAN Clusters: {n_clusters_dbscan}")
print(f"4. PCA Variance: {explained_variance_sum:.2f}%")
print(f"5. GBR R-squared: {r2_gbr:.4f}")
print(f"6. KMeans Silhouette: {silhouette:.4f}")
print(f"7. LDA Mean (Component 1): {lda_mean_first:.4f}")
print(f"8. Power Hierarchy: {power_hierarchy_result}")
print("=" * 60)
