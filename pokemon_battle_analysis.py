# ==========================================
# Integrated Pokemon Battle Strategy Analysis Script
# Requirements: pandas, numpy, matplotlib, scipy, scikit-learn, xgboost
# Input files: pokemon.xlsx, moves.xlsx, evolution_trees.xlsx
# Output files: elasticnet_scatter.png, xgboost_bar.png, dendrogram.png,
#               tsne_scatter.png, svr_residuals.png
# ==========================================

import pandas as pd
import numpy as np
import matplotlib.pyplot as plt
from scipy import stats
from scipy.cluster.hierarchy import dendrogram, linkage
from sklearn.model_selection import train_test_split
from sklearn.preprocessing import StandardScaler, LabelEncoder
from sklearn.linear_model import ElasticNet
from sklearn.svm import SVR
from sklearn.cluster import AgglomerativeClustering
from sklearn.manifold import TSNE
from sklearn.metrics import root_mean_squared_error, f1_score, davies_bouldin_score, mean_absolute_error
from xgboost import XGBClassifier
import warnings
warnings.filterwarnings('ignore')

# ---------- Load Data ----------
print("Loading datasets...")
pokemon = pd.read_excel('pokemon.xlsx')
moves = pd.read_excel('moves.xlsx')
evolution = pd.read_excel('evolution_trees.xlsx')
print(f"Pokemon: {pokemon.shape[0]} records, Moves: {moves.shape[0]} records, Evolution: {evolution.shape[0]} records")

# ==========================================
# PROCEDURE 1: ElasticNet Regression
# Predict base_experience from combat stats
# ==========================================
print("\n" + "="*50)
print("PROCEDURE 1: ElasticNet Regression")
print("="*50)

# Designated variables: base_experience, hp, attack, defense, special_attack, special_defense, speed
designated_cols_1 = ['base_experience', 'hp', 'attack', 'defense', 'special_attack', 'special_defense', 'speed']
df1 = pokemon.copy()

# Eliminate records where ANY designated variable is absent (logical disjunction)
df1_clean = df1.dropna(subset=designated_cols_1)
print(f"Records after elimination: {len(df1_clean)} (removed {len(df1) - len(df1_clean)} with missing designated values)")

# Define features and target
X1 = df1_clean[['hp', 'attack', 'defense', 'special_attack', 'special_defense', 'speed']]
y1 = df1_clean['base_experience']

# Partition: 65% training, 35% validation
X1_train, X1_val, y1_train, y1_val = train_test_split(X1, y1, test_size=0.35, random_state=42)
print(f"Training samples: {len(X1_train)}, Validation samples: {len(X1_val)}")

# Configure ElasticNet
elasticnet = ElasticNet(alpha=0.5, l1_ratio=0.5, max_iter=5000, random_state=42)
elasticnet.fit(X1_train, y1_train)

# Predict and calculate RMSE
y1_pred = elasticnet.predict(X1_val)
rmse = root_mean_squared_error(y1_val, y1_pred)
print(f"\nRoot Mean Squared Error: {rmse:.2f}")

# Generate scatter visualization
plt.figure(figsize=(10, 8))
plt.scatter(y1_val, y1_pred, alpha=0.5, edgecolors='k', linewidths=0.5)
plt.plot([y1_val.min(), y1_val.max()], [y1_val.min(), y1_val.max()], 'r--', lw=2, label='Perfect Prediction')
plt.xlabel('Actual Base Experience')
plt.ylabel('Predicted Base Experience')
plt.title('ElasticNet: Actual vs Predicted Base Experience')
plt.legend()
plt.tight_layout()
plt.savefig('elasticnet_scatter.png', dpi=150)
plt.close()
print("Saved: elasticnet_scatter.png")

# ==========================================
# PROCEDURE 2: XGBoost Multiclass Classification
# Predict primary_type from morphological attributes
# ==========================================
print("\n" + "="*50)
print("PROCEDURE 2: XGBoost Multiclass Classification")
print("="*50)

# Designated variables: primary_type, height, weight, hp, attack, defense, speed
designated_cols_2 = ['primary_type', 'height', 'weight', 'hp', 'attack', 'defense', 'speed']
df2 = pokemon.copy()

# Eliminate records where ANY designated variable is absent
df2_clean = df2.dropna(subset=designated_cols_2)
print(f"Records after elimination: {len(df2_clean)} (removed {len(df2) - len(df2_clean)} with missing designated values)")

# Define features and target
X2 = df2_clean[['height', 'weight', 'hp', 'attack', 'defense', 'speed']]
y2 = df2_clean['primary_type']

# Encode labels
le = LabelEncoder()
y2_encoded = le.fit_transform(y2)

# Partition: 70% training, 30% validation with stratified sampling
X2_train, X2_val, y2_train, y2_val = train_test_split(X2, y2_encoded, test_size=0.30, random_state=42, stratify=y2_encoded)
print(f"Training samples: {len(X2_train)}, Validation samples: {len(X2_val)}")
print(f"Number of classes: {len(le.classes_)}")

# Configure XGBClassifier
xgb_clf = XGBClassifier(n_estimators=150, max_depth=6, learning_rate=0.1, subsample=0.8, random_state=42,
                        use_label_encoder=False, eval_metric='mlogloss')
xgb_clf.fit(X2_train, y2_train)

# Predict and calculate weighted F1 score
y2_pred = xgb_clf.predict(X2_val)
weighted_f1 = f1_score(y2_val, y2_pred, average='weighted')
print(f"\nWeighted F1 Score: {weighted_f1:.4f}")

# Generate bar visualization - prediction counts per category
y2_pred_labels = le.inverse_transform(y2_pred)
pred_counts = pd.Series(y2_pred_labels).value_counts().sort_index()

plt.figure(figsize=(14, 8))
plt.bar(range(len(pred_counts)), pred_counts.values, color='steelblue', edgecolor='black')
plt.xticks(range(len(pred_counts)), pred_counts.index, rotation=45, ha='right')
plt.xlabel('Elemental Type Categories')
plt.ylabel('Prediction Counts')
plt.title('XGBoost: Prediction Counts per Elemental Type Category')
plt.tight_layout()
plt.savefig('xgboost_bar.png', dpi=150)
plt.close()
print("Saved: xgboost_bar.png")

# ==========================================
# PROCEDURE 3: Agglomerative Hierarchical Clustering
# Cluster move techniques by effectiveness characteristics
# ==========================================
print("\n" + "="*50)
print("PROCEDURE 3: Agglomerative Hierarchical Clustering")
print("="*50)

# Designated variables: power, pp, accuracy
designated_cols_3 = ['power', 'pp', 'accuracy']
df3 = moves.copy()

# Eliminate records where ANY designated variable is absent
df3_clean = df3.dropna(subset=designated_cols_3)
print(f"Records after elimination: {len(df3_clean)} (removed {len(df3) - len(df3_clean)} with missing designated values)")

# Fit StandardScaler and transform
scaler3 = StandardScaler()
X3_scaled = scaler3.fit_transform(df3_clean[designated_cols_3])
print(f"Scaled data shape: {X3_scaled.shape}")

# Execute AgglomerativeClustering
agg_cluster = AgglomerativeClustering(n_clusters=4, linkage='ward', metric='euclidean')
cluster_labels = agg_cluster.fit_predict(X3_scaled)

# Calculate Davies-Bouldin index
db_index = davies_bouldin_score(X3_scaled, cluster_labels)
print(f"\nDavies-Bouldin Index: {db_index:.4f}")

# Generate dendrogram visualization
# Use linkage for dendrogram (limit samples for visibility)
sample_size = min(100, len(X3_scaled))
np.random.seed(42)
sample_indices = np.random.choice(len(X3_scaled), sample_size, replace=False)
X3_sample = X3_scaled[sample_indices]

linkage_matrix = linkage(X3_sample, method='ward', metric='euclidean')

plt.figure(figsize=(14, 8))
dendrogram(linkage_matrix, truncate_mode='level', p=5, leaf_rotation=90, leaf_font_size=8)
plt.xlabel('Sample Indices')
plt.ylabel('Linkage Distance')
plt.title('Agglomerative Clustering: Dendrogram of Move Techniques')
plt.tight_layout()
plt.savefig('dendrogram.png', dpi=150)
plt.close()
print("Saved: dendrogram.png")

# ==========================================
# PROCEDURE 4: Manifold Learning (t-SNE) Dimensionality Reduction
# Project pokemon stats into 2D space
# ==========================================
print("\n" + "="*50)
print("PROCEDURE 4: t-SNE Dimensionality Reduction")
print("="*50)

# Designated variables: hp, attack, defense, special_attack, special_defense, speed
designated_cols_4 = ['hp', 'attack', 'defense', 'special_attack', 'special_defense', 'speed']
df4 = pokemon.copy()

# Eliminate records where ANY designated variable is absent
df4_clean = df4.dropna(subset=designated_cols_4)
print(f"Records after elimination: {len(df4_clean)} (removed {len(df4) - len(df4_clean)} with missing designated values)")

# Fit StandardScaler and transform
scaler4 = StandardScaler()
X4_scaled = scaler4.fit_transform(df4_clean[designated_cols_4])
print(f"Scaled data shape: {X4_scaled.shape}")

# Configure and run t-SNE
tsne = TSNE(n_components=2, perplexity=30, learning_rate=200, random_state=42)
X4_embedded = tsne.fit_transform(X4_scaled)

# Get KL divergence
kl_divergence = tsne.kl_divergence_
print(f"\nKullback-Leibler Divergence: {kl_divergence:.4f}")

# Generate scatter visualization
plt.figure(figsize=(12, 10))
scatter = plt.scatter(X4_embedded[:, 0], X4_embedded[:, 1], alpha=0.6, c=range(len(X4_embedded)),
                      cmap='viridis', edgecolors='k', linewidths=0.3)
plt.xlabel('First Embedding Dimension')
plt.ylabel('Second Embedding Dimension')
plt.title('t-SNE: Pokemon Combat Stats 2D Projection')
plt.colorbar(scatter, label='Record Index')
plt.tight_layout()
plt.savefig('tsne_scatter.png', dpi=150)
plt.close()
print("Saved: tsne_scatter.png")

# ==========================================
# PROCEDURE 5: Support Vector Regression (SVR)
# Estimate pp from power and accuracy
# ==========================================
print("\n" + "="*50)
print("PROCEDURE 5: Support Vector Regression")
print("="*50)

# Designated variables: pp, power, accuracy
designated_cols_5 = ['pp', 'power', 'accuracy']
df5 = moves.copy()

# Eliminate records where ANY designated variable is absent
df5_clean = df5.dropna(subset=designated_cols_5)
print(f"Records after elimination: {len(df5_clean)} (removed {len(df5) - len(df5_clean)} with missing designated values)")

# Define features and target
X5 = df5_clean[['power', 'accuracy']]
y5 = df5_clean['pp']

# Partition: 75% training, 25% validation
X5_train, X5_val, y5_train, y5_val = train_test_split(X5, y5, test_size=0.25, random_state=42)
print(f"Training samples: {len(X5_train)}, Validation samples: {len(X5_val)}")

# Configure SVR
svr = SVR(kernel='rbf', C=10.0, epsilon=0.1, gamma='scale')
svr.fit(X5_train, y5_train)

# Predict and calculate MAE
y5_pred = svr.predict(X5_val)
mae = mean_absolute_error(y5_val, y5_pred)
print(f"\nMean Absolute Error: {mae:.3f}")

# Generate residual scatter visualization
absolute_residuals = np.abs(y5_val - y5_pred)

plt.figure(figsize=(10, 8))
plt.scatter(y5_pred, absolute_residuals, alpha=0.6, edgecolors='k', linewidths=0.5)
plt.axhline(y=mae, color='r', linestyle='--', label=f'MAE = {mae:.3f}')
plt.xlabel('Predicted PP Values')
plt.ylabel('Absolute Residuals')
plt.title('SVR: Residual Analysis for PP Prediction')
plt.legend()
plt.tight_layout()
plt.savefig('svr_residuals.png', dpi=150)
plt.close()
print("Saved: svr_residuals.png")

# ==========================================
# PROCEDURE 6: One-Way ANOVA
# Test if attack differs across primary_type categories
# ==========================================
print("\n" + "="*50)
print("PROCEDURE 6: One-Way ANOVA F-Test")
print("="*50)

# Designated variables: primary_type, attack
designated_cols_6 = ['primary_type', 'attack']
df6 = pokemon.copy()

# Eliminate records where ANY designated variable is absent
df6_clean = df6.dropna(subset=designated_cols_6)
print(f"Records after elimination: {len(df6_clean)} (removed {len(df6) - len(df6_clean)} with missing designated values)")

# Group attack values by primary_type
groups = [group['attack'].values for name, group in df6_clean.groupby('primary_type')]
print(f"Number of type categories: {len(groups)}")

# Conduct one-way ANOVA F-test
f_statistic, p_value = stats.f_oneway(*groups)
print(f"\nF-Statistic: {f_statistic:.2f}")
print(f"P-Value: {p_value:.2e}")

# ==========================================
# PROCEDURE 7: Gini Impurity Coefficient
# Measure move type diversity concentration
# ==========================================
print("\n" + "="*50)
print("PROCEDURE 7: Gini Impurity Coefficient")
print("="*50)

# Designated variable: type
df7 = moves.copy()

# Eliminate records where type is absent
df7_clean = df7.dropna(subset=['type'])
print(f"Records after elimination: {len(df7_clean)} (removed {len(df7) - len(df7_clean)} with missing designated values)")

# Calculate proportions for each type category
type_counts = df7_clean['type'].value_counts()
total_count = len(df7_clean)
proportions = type_counts / total_count
print(f"Number of unique types: {len(type_counts)}")

# Calculate Gini impurity: 1 - sum(p_i^2)
gini_impurity = 1 - np.sum(proportions ** 2)
print(f"\nGini Impurity Coefficient: {gini_impurity:.4f}")

# ==========================================
# PROCEDURE 8: Cross-Dataset Pearson Correlation
# Correlate pokemon weight with move power
# ==========================================
print("\n" + "="*50)
print("PROCEDURE 8: Cross-Dataset Pearson Correlation")
print("="*50)

# From pokemon.xlsx: extract weight, eliminate missing
pokemon_weight = pokemon.copy()
pokemon_weight_clean = pokemon_weight.dropna(subset=['weight'])['weight'].reset_index(drop=True)
print(f"Pokemon weight records after elimination: {len(pokemon_weight_clean)}")

# From moves.xlsx: extract power, eliminate missing
moves_power = moves.copy()
moves_power_clean = moves_power.dropna(subset=['power'])['power'].reset_index(drop=True)
print(f"Moves power records after elimination: {len(moves_power_clean)}")

# Use minimum length for paired samples
min_len = min(len(pokemon_weight_clean), len(moves_power_clean))
weight_paired = pokemon_weight_clean.iloc[:min_len]
power_paired = moves_power_clean.iloc[:min_len]
print(f"Paired samples for correlation: {min_len}")

# Calculate Pearson correlation coefficient
pearson_corr, _ = stats.pearsonr(weight_paired, power_paired)
print(f"\nPearson Correlation Coefficient: {pearson_corr:.4f}")

# ==========================================
# OPEN-ENDED QUESTION ANSWER
# ==========================================
print("\n" + "="*50)
print("OPEN-ENDED QUESTION: Why t-SNE Preserves Local Structure Better")
print("="*50)

# Answer to: Why manifold learning dimensionality reduction preserves local neighborhood
# structure superior to linear projection methods
answer = """
t-SNE preserves local neighborhood structure superior to linear methods like PCA because it
explicitly optimizes a probability-based similarity metric that focuses on preserving the
relative distances between nearby points in the high-dimensional space, using Student's
t-distribution in the low-dimensional space to prevent crowding, whereas linear methods
only maximize variance along orthogonal axes without considering local geometric relationships.
"""
print(answer.strip())

# ==========================================
# SUMMARY OF KEY RESULTS
# ==========================================
print("\n" + "="*50)
print("KEY RESULTS SUMMARY")
print("="*50)
print(f"1. ElasticNet RMSE: {rmse:.2f}")
print(f"2. XGBoost Weighted F1: {weighted_f1:.4f}")
print(f"3. Davies-Bouldin Index: {db_index:.4f}")
print(f"4. t-SNE KL Divergence: {kl_divergence:.4f}")
print(f"5. SVR MAE: {mae:.3f}")
print(f"6. ANOVA F-Statistic: {f_statistic:.2f}")
print(f"7. Gini Impurity: {gini_impurity:.4f}")
print(f"8. Pearson Correlation: {pearson_corr:.4f}")
