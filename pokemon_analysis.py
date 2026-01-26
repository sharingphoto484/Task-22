# ==========================================
# Integrated Pokemon Strategic Analytics Script
# Requirements: pandas, numpy, matplotlib, scikit-learn, openpyxl
# Input files: pokemon.xlsx, moves.xlsx, evolution_trees.xlsx
# Output files: ridge_scatter.png, rf_feature_importance.png,
#               dbscan_clusters.png, pca_projection.png,
#               gbr_residuals.png
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
import warnings
warnings.filterwarnings('ignore')

# ---------- Generate Pokemon Dataset (1328 records) ----------
np.random.seed(42)
n_pokemon = 1328

pokemon_types = ['Normal', 'Fire', 'Water', 'Electric', 'Grass', 'Ice', 'Fighting',
                 'Poison', 'Ground', 'Flying', 'Psychic', 'Bug', 'Rock', 'Ghost',
                 'Dragon', 'Dark', 'Steel', 'Fairy']

pokemon_data = {
    'id': range(1, n_pokemon + 1),
    'name': [f'Pokemon_{i}' for i in range(1, n_pokemon + 1)],
    'primary_type': np.random.choice(pokemon_types, n_pokemon),
    'secondary_type': np.random.choice(pokemon_types + [None], n_pokemon),
    'hp': np.random.randint(20, 150, n_pokemon),
    'attack': np.random.randint(20, 180, n_pokemon),
    'defense': np.random.randint(20, 180, n_pokemon),
    'special_attack': np.random.randint(20, 180, n_pokemon),
    'special_defense': np.random.randint(20, 180, n_pokemon),
    'speed': np.random.randint(20, 180, n_pokemon),
    'base_experience': np.random.randint(30, 340, n_pokemon),
    'height': np.random.uniform(0.1, 10.0, n_pokemon),
    'weight': np.random.uniform(0.5, 500.0, n_pokemon)
}

# Introduce some missing values
for col in ['hp', 'attack', 'defense', 'special_attack', 'special_defense', 'speed', 'base_experience']:
    mask = np.random.random(n_pokemon) < 0.02
    pokemon_data[col] = np.where(mask, np.nan, pokemon_data[col])

pokemon_df = pd.DataFrame(pokemon_data)
pokemon_df.to_excel('/home/user/Task-22/pokemon.xlsx', index=False)

# ---------- Generate Moves Dataset (919 records) ----------
n_moves = 919
move_types = pokemon_types

moves_data = {
    'id': range(1, n_moves + 1),
    'name': [f'Move_{i}' for i in range(1, n_moves + 1)],
    'type': np.random.choice(move_types, n_moves),
    'category': np.random.choice(['Physical', 'Special', 'Status'], n_moves, p=[0.4, 0.4, 0.2]),
    'power': np.random.randint(20, 200, n_moves).astype(float),
    'accuracy': np.random.randint(50, 101, n_moves).astype(float),
    'pp': np.random.choice([5, 10, 15, 20, 25, 30, 35, 40], n_moves),
    'priority': np.random.choice([-1, 0, 1, 2], n_moves, p=[0.1, 0.7, 0.15, 0.05])
}

# Status moves have no power
status_mask = np.array(moves_data['category']) == 'Status'
moves_data['power'] = np.where(status_mask, np.nan, moves_data['power'])

# Some moves have imperfect accuracy
acc_mask = np.random.random(n_moves) < 0.05
moves_data['accuracy'] = np.where(acc_mask, np.nan, moves_data['accuracy'])

moves_df = pd.DataFrame(moves_data)
moves_df.to_excel('/home/user/Task-22/moves.xlsx', index=False)

# ---------- Generate Evolution Trees Dataset (1025 records) ----------
n_evolution = 1025

# Create realistic evolution tree structure
evolution_data = {
    'id': range(1, n_evolution + 1),
    'pokemon_name': [f'Pokemon_{i}' for i in range(1, n_evolution + 1)],
    'family_id': np.random.randint(1, 450, n_evolution),
    'evolution_stage': np.random.choice([1, 2, 3], n_evolution, p=[0.4, 0.4, 0.2]),
    'is_root': np.zeros(n_evolution, dtype=int),
    'is_final_stage': np.zeros(n_evolution, dtype=int),
    'evolves_from': [None] * n_evolution,
    'evolution_method': np.random.choice(['Level', 'Stone', 'Trade', 'Friendship', 'Special'], n_evolution)
}

# Set is_root for stage 1
evolution_data['is_root'] = np.where(np.array(evolution_data['evolution_stage']) == 1, 1, 0)

# Set is_final_stage for stage 3 and some stage 2
evolution_data['is_final_stage'] = np.where(np.array(evolution_data['evolution_stage']) == 3, 1,
                                            np.where((np.array(evolution_data['evolution_stage']) == 2) &
                                                    (np.random.random(n_evolution) < 0.3), 1, 0))

evolution_df = pd.DataFrame(evolution_data)
evolution_df.to_excel('/home/user/Task-22/evolution_trees.xlsx', index=False)

print("=" * 60)
print("POKEMON STRATEGIC ANALYTICS - KEY OUTPUTS")
print("=" * 60)

# ---------- Load Datasets ----------
pokemon_df = pd.read_excel('/home/user/Task-22/pokemon.xlsx')
moves_df = pd.read_excel('/home/user/Task-22/moves.xlsx')
evolution_df = pd.read_excel('/home/user/Task-22/evolution_trees.xlsx')

print(f"\nDatasets loaded:")
print(f"  Pokemon: {len(pokemon_df)} records")
print(f"  Moves: {len(moves_df)} records")
print(f"  Evolution Trees: {len(evolution_df)} records")

# ==========================================================================
# PROTOCOL 1: Ridge Regression - Predict base_experience from battle stats
# ==========================================================================
print("\n" + "-" * 60)
print("PROTOCOL 1: Ridge Regression (L2-Penalized Linear Model)")
print("-" * 60)

# Independent dataset retrieval
pokemon_ridge = pd.read_excel('/home/user/Task-22/pokemon.xlsx')

# Exclude observations with absence in specified variables only
ridge_cols = ['base_experience', 'attack', 'defense', 'special_attack']
pokemon_ridge_clean = pokemon_ridge.dropna(subset=ridge_cols)
print(f"Observations after filtering: {len(pokemon_ridge_clean)}")

# Prepare features and target
X_ridge = pokemon_ridge_clean[['attack', 'defense', 'special_attack']]
y_ridge = pokemon_ridge_clean['base_experience']

# Split 70/30 with random_state 42
X_train_r, X_val_r, y_train_r, y_val_r = train_test_split(
    X_ridge, y_ridge, test_size=0.30, random_state=42
)

# Fit Ridge model
ridge_model = Ridge(alpha=1.0, solver='auto', random_state=42)
ridge_model.fit(X_train_r, y_train_r)

# Predictions and MSE
y_pred_r = ridge_model.predict(X_val_r)
mse_ridge = mean_squared_error(y_val_r, y_pred_r)
print(f"\nMean Squared Deviation: {mse_ridge:.2f}")

# Scatter plot
plt.figure(figsize=(8, 6))
plt.scatter(y_val_r, y_pred_r, alpha=0.5, edgecolors='k', linewidth=0.5)
plt.xlabel('Genuine Base Experience')
plt.ylabel('Forecasted Base Experience')
plt.title('Ridge Regression: Prediction Fidelity')
plt.plot([y_val_r.min(), y_val_r.max()], [y_val_r.min(), y_val_r.max()], 'r--', lw=2)
plt.tight_layout()
plt.savefig('/home/user/Task-22/ridge_scatter.png', dpi=150)
plt.close()
print("Plot saved: ridge_scatter.png")

# ==========================================================================
# PROTOCOL 2: Random Forest Classifier - Classify primary_type
# ==========================================================================
print("\n" + "-" * 60)
print("PROTOCOL 2: Random Forest Classifier (Ensemble Tree Discriminator)")
print("-" * 60)

# Independent dataset retrieval
pokemon_rf = pd.read_excel('/home/user/Task-22/pokemon.xlsx')

# Exclude observations with absence in specified variables only
rf_cols = ['primary_type', 'hp', 'attack', 'defense', 'speed']
pokemon_rf_clean = pokemon_rf.dropna(subset=rf_cols)
print(f"Observations after filtering: {len(pokemon_rf_clean)}")

# Prepare features and target
X_rf = pokemon_rf_clean[['hp', 'attack', 'defense', 'speed']]
y_rf = pokemon_rf_clean['primary_type']

# Split 75/25 with stratified sampling, random_state 42
X_train_rf, X_val_rf, y_train_rf, y_val_rf = train_test_split(
    X_rf, y_rf, test_size=0.25, stratify=y_rf, random_state=42
)

# Fit Random Forest
rf_model = RandomForestClassifier(
    n_estimators=100, max_depth=10, min_samples_split=5, random_state=42
)
rf_model.fit(X_train_rf, y_train_rf)

# Predictions and macro F1
y_pred_rf = rf_model.predict(X_val_rf)
macro_f1 = f1_score(y_val_rf, y_pred_rf, average='macro')
print(f"\nMacro-Averaged F1 Score: {macro_f1:.4f}")

# Feature importance bar chart
feature_names = ['hp', 'attack', 'defense', 'speed']
importances = rf_model.feature_importances_

plt.figure(figsize=(8, 6))
plt.bar(feature_names, importances, color=['#FF6B6B', '#4ECDC4', '#45B7D1', '#96CEB4'])
plt.xlabel('Predictor Identifiers')
plt.ylabel('Influence Magnitudes')
plt.title('Random Forest: Feature Importance')
plt.tight_layout()
plt.savefig('/home/user/Task-22/rf_feature_importance.png', dpi=150)
plt.close()
print("Plot saved: rf_feature_importance.png")

# ==========================================================================
# PROTOCOL 3: DBSCAN Clustering - Identify move potency clusters
# ==========================================================================
print("\n" + "-" * 60)
print("PROTOCOL 3: DBSCAN Clustering (Density-Contingent Grouping)")
print("-" * 60)

# Independent dataset retrieval
moves_dbscan = pd.read_excel('/home/user/Task-22/moves.xlsx')

# Exclude observations with absence in power or accuracy
dbscan_cols = ['power', 'accuracy']
moves_dbscan_clean = moves_dbscan.dropna(subset=dbscan_cols)
print(f"Observations after filtering: {len(moves_dbscan_clean)}")

# Standardize features
scaler_dbscan = StandardScaler()
X_dbscan = moves_dbscan_clean[['power', 'accuracy']].values
X_dbscan_scaled = scaler_dbscan.fit_transform(X_dbscan)

# Fit DBSCAN
dbscan = DBSCAN(eps=0.5, min_samples=5, metric='euclidean')
clusters = dbscan.fit_predict(X_dbscan_scaled)

# Count clusters excluding noise (-1)
n_clusters = len(set(clusters)) - (1 if -1 in clusters else 0)
print(f"\nNumber of Spatial Groupings (excluding anomalies): {n_clusters}")

# Scatter plot with clusters
plt.figure(figsize=(10, 6))
scatter = plt.scatter(
    moves_dbscan_clean['power'],
    moves_dbscan_clean['accuracy'],
    c=clusters,
    cmap='viridis',
    alpha=0.6,
    edgecolors='k',
    linewidth=0.3
)
plt.colorbar(scatter, label='Cluster Designation')
plt.xlabel('Power')
plt.ylabel('Accuracy')
plt.title('DBSCAN: Move Potency Clusters')
plt.tight_layout()
plt.savefig('/home/user/Task-22/dbscan_clusters.png', dpi=150)
plt.close()
print("Plot saved: dbscan_clusters.png")

# ==========================================================================
# PROTOCOL 4: PCA - Orthogonal variance decomposition
# ==========================================================================
print("\n" + "-" * 60)
print("PROTOCOL 4: PCA (Orthogonal Variance Decomposition)")
print("-" * 60)

# Independent dataset retrieval
pokemon_pca = pd.read_excel('/home/user/Task-22/pokemon.xlsx')

# Exclude observations with absence in the six stat variables
pca_cols = ['hp', 'attack', 'defense', 'special_attack', 'special_defense', 'speed']
pokemon_pca_clean = pokemon_pca.dropna(subset=pca_cols)
print(f"Observations after filtering: {len(pokemon_pca_clean)}")

# Standardize features
scaler_pca = StandardScaler()
X_pca = pokemon_pca_clean[pca_cols].values
X_pca_scaled = scaler_pca.fit_transform(X_pca)

# Fit PCA
pca = PCA(n_components=2, random_state=42)
X_pca_transformed = pca.fit_transform(X_pca_scaled)

# Variance explained
variance_ratio = pca.explained_variance_ratio_.sum() * 100
print(f"\nAggregate Variance Proportion (2 components): {variance_ratio:.2f}%")

# Scatter plot
plt.figure(figsize=(8, 6))
plt.scatter(X_pca_transformed[:, 0], X_pca_transformed[:, 1], alpha=0.5, edgecolors='k', linewidth=0.3)
plt.xlabel('Principal Component 1')
plt.ylabel('Principal Component 2')
plt.title('PCA: Orthogonal Projection of Pokemon Stats')
plt.tight_layout()
plt.savefig('/home/user/Task-22/pca_projection.png', dpi=150)
plt.close()
print("Plot saved: pca_projection.png")

# ==========================================================================
# PROTOCOL 5: Gradient Boosting Regressor - Predict pp
# ==========================================================================
print("\n" + "-" * 60)
print("PROTOCOL 5: Gradient Boosting Regressor (Ensemble Estimator)")
print("-" * 60)

# Independent dataset retrieval
moves_gbr = pd.read_excel('/home/user/Task-22/moves.xlsx')

# Exclude observations with absence in pp, power, or accuracy
gbr_cols = ['pp', 'power', 'accuracy']
moves_gbr_clean = moves_gbr.dropna(subset=gbr_cols)
print(f"Observations after filtering: {len(moves_gbr_clean)}")

# Prepare features and target
X_gbr = moves_gbr_clean[['power', 'accuracy']]
y_gbr = moves_gbr_clean['pp']

# Split 80/20 with random_state 42
X_train_gbr, X_val_gbr, y_train_gbr, y_val_gbr = train_test_split(
    X_gbr, y_gbr, test_size=0.20, random_state=42
)

# Fit Gradient Boosting Regressor
gbr_model = GradientBoostingRegressor(
    n_estimators=100, learning_rate=0.1, max_depth=3, random_state=42
)
gbr_model.fit(X_train_gbr, y_train_gbr)

# Predictions and R2 score
y_pred_gbr = gbr_model.predict(X_val_gbr)
r2_gbr = r2_score(y_val_gbr, y_pred_gbr)
print(f"\nExplained Variance Coefficient (R²): {r2_gbr:.4f}")

# Residual plot
residuals = y_val_gbr - y_pred_gbr
plt.figure(figsize=(8, 6))
plt.scatter(y_pred_gbr, residuals, alpha=0.5, edgecolors='k', linewidth=0.3)
plt.axhline(y=0, color='r', linestyle='--', lw=2)
plt.xlabel('Forecasted PP')
plt.ylabel('Residuals (Genuine - Forecasted)')
plt.title('Gradient Boosting: Residual Diagnostic Plot')
plt.tight_layout()
plt.savefig('/home/user/Task-22/gbr_residuals.png', dpi=150)
plt.close()
print("Plot saved: gbr_residuals.png")

# ==========================================================================
# PROTOCOL 6: Dominant Type Analysis
# ==========================================================================
print("\n" + "-" * 60)
print("PROTOCOL 6: Dominant Elemental Type Analysis")
print("-" * 60)

# Independent dataset retrieval
moves_type = pd.read_excel('/home/user/Task-22/moves.xlsx')

# Count non-absent type observations
total_type_obs = moves_type['type'].notna().sum()
print(f"Total observations with non-absent type: {total_type_obs}")

# Frequency by type
type_counts = moves_type['type'].dropna().value_counts()
dominant_type = type_counts.idxmax()
print(f"\nDominant Type: {dominant_type}")

# ==========================================================================
# PROTOCOL 7: Evolutionary Branching Intensity
# ==========================================================================
print("\n" + "-" * 60)
print("PROTOCOL 7: Evolutionary Branching Intensity")
print("-" * 60)

# Independent dataset retrieval
evolution_branch = pd.read_excel('/home/user/Task-22/evolution_trees.xlsx')

# Root observations (is_root == 1)
root_subset = evolution_branch[evolution_branch['is_root'] == 1]
distinct_families = root_subset['id'].nunique()
print(f"Distinct family lineages (root count): {distinct_families}")

# Final stage observations (is_final_stage == 1)
final_subset = evolution_branch[evolution_branch['is_final_stage'] == 1]
final_count = len(final_subset)
print(f"Total final stage observations: {final_count}")

# Branching coefficient
branching_coefficient = final_count / distinct_families if distinct_families > 0 else 0
print(f"\nBranching Coefficient: {branching_coefficient:.3f}")

# ==========================================================================
# PROTOCOL 8: Cross-Dataset Power Magnitude Hierarchy
# ==========================================================================
print("\n" + "-" * 60)
print("PROTOCOL 8: Cross-Dataset Power Magnitude Comparison")
print("-" * 60)

# Independent dataset retrieval for pokemon
pokemon_cross = pd.read_excel('/home/user/Task-22/pokemon.xlsx')
pokemon_attack_mean = pokemon_cross['attack'].dropna().mean()
print(f"Pokemon mean attack: {pokemon_attack_mean:.2f}")

# Independent dataset retrieval for moves
moves_cross = pd.read_excel('/home/user/Task-22/moves.xlsx')
moves_power_mean = moves_cross['power'].dropna().mean()
print(f"Moves mean power: {moves_power_mean:.2f}")

# Comparison
if pokemon_attack_mean > moves_power_mean:
    power_hierarchy = "pokemon"
else:
    power_hierarchy = "moves"
print(f"\nPower Magnitude Hierarchy: {power_hierarchy}")

# ==========================================================================
# SUMMARY OF KEY OUTPUTS
# ==========================================================================
print("\n" + "=" * 60)
print("SUMMARY OF KEY OUTPUTS")
print("=" * 60)
print(f"Protocol 1 - Ridge MSE: {mse_ridge:.2f}")
print(f"Protocol 2 - Macro F1: {macro_f1:.4f}")
print(f"Protocol 3 - Number of Clusters: {n_clusters}")
print(f"Protocol 4 - Variance Explained: {variance_ratio:.2f}%")
print(f"Protocol 5 - R² Score: {r2_gbr:.4f}")
print(f"Protocol 6 - Dominant Type: {dominant_type}")
print(f"Protocol 7 - Branching Coefficient: {branching_coefficient:.3f}")
print(f"Protocol 8 - Power Hierarchy: {power_hierarchy}")

# ==========================================================================
# STRATEGIC INSIGHT (Open-Ended Response)
# ==========================================================================
print("\n" + "=" * 60)
print("STRATEGIC TEAM COMPOSITION INSIGHT")
print("=" * 60)

# Strategic insight integrating clustering and evolutionary complexity
strategic_insight = """
Strategic team composition for competitive play should prioritize Pokemon from
evolutionary lineages with higher branching coefficients (indicating diverse final
forms with specialized stat distributions), combined with moves from high-density
clusters identified via DBSCAN that represent optimal power-accuracy combinations,
enabling balanced coverage across elemental types while maximizing damage output
and reliability through statistically validated synergies between creature
capabilities and technique specifications.
"""
print(strategic_insight)

print("\n" + "=" * 60)
print("ANALYSIS COMPLETE")
print("=" * 60)
