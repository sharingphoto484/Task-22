# ==========================================
# Athletics Performance Analysis Script
# Requirements: pandas, numpy, matplotlib, scipy, scikit-learn
# Input files: Discus Throw Men.xlsx, Shot Put Men.xlsx, Javelin Throw Men.xlsx
# Output files: gradient_boosting_scatter.png, kmeans_hexbin.png,
#               rf_nationality_bar.png, pca_biplot.png, dendrogram_comparison.png
# ==========================================

import pandas as pd
import numpy as np
import matplotlib.pyplot as plt
from scipy import stats
from scipy.cluster.hierarchy import dendrogram, linkage
from sklearn.ensemble import GradientBoostingRegressor, RandomForestClassifier
from sklearn.cluster import KMeans, AgglomerativeClustering
from sklearn.decomposition import PCA
from sklearn.preprocessing import StandardScaler
from sklearn.model_selection import train_test_split
from sklearn.metrics import root_mean_squared_error, silhouette_score, f1_score, davies_bouldin_score
import warnings
warnings.filterwarnings('ignore')

# ---------- Load Datasets ----------
discus_df = pd.read_excel('Discus Throw Men.xlsx')
shotput_df = pd.read_excel('Shot Put Men.xlsx')
javelin_df = pd.read_excel('Javelin Throw Men.xlsx')

print("=" * 60)
print("ATHLETICS PERFORMANCE ANALYSIS - KEY OUTPUTS")
print("=" * 60)

# ---------- Protocol 1: Gradient Boosting Regression ----------
print("\n--- Protocol 1: Gradient Boosting Regression ---")

# Filter each dataset for missing Result Score OR Mark
discus_p1 = discus_df.dropna(subset=['Result Score', 'Mark'])
shotput_p1 = shotput_df.dropna(subset=['Result Score', 'Mark'])
javelin_p1 = javelin_df.dropna(subset=['Result Score', 'Mark'])

# Add discipline indicator column
discus_p1 = discus_p1.copy()
shotput_p1 = shotput_p1.copy()
javelin_p1 = javelin_p1.copy()
discus_p1['Discipline'] = 'Discus'
shotput_p1['Discipline'] = 'ShotPut'
javelin_p1['Discipline'] = 'Javelin'

# Concatenate datasets
combined_p1 = pd.concat([discus_p1, shotput_p1, javelin_p1], ignore_index=True)

# Prepare features and target
X_p1 = combined_p1[['Mark']]
y_p1 = combined_p1['Result Score']

# Split 75/25
X_train_p1, X_val_p1, y_train_p1, y_val_p1 = train_test_split(
    X_p1, y_p1, test_size=0.25, random_state=42
)

# Configure and fit GradientBoostingRegressor
gbr = GradientBoostingRegressor(
    n_estimators=120,
    max_depth=4,
    learning_rate=0.08,
    min_samples_split=5,
    random_state=42
)
gbr.fit(X_train_p1, y_train_p1)

# Predict and calculate RMSE
y_pred_p1 = gbr.predict(X_val_p1)
rmse_p1 = root_mean_squared_error(y_val_p1, y_pred_p1)
print(f"Root Mean Squared Error: {rmse_p1:.3f}")

# Generate scatter plot
plt.figure(figsize=(8, 6))
plt.scatter(y_val_p1, y_pred_p1, alpha=0.6, edgecolors='k', linewidth=0.5)
plt.plot([y_val_p1.min(), y_val_p1.max()], [y_val_p1.min(), y_val_p1.max()], 'r--', lw=2)
plt.xlabel('Actual Result Score')
plt.ylabel('Predicted Result Score')
plt.title('Gradient Boosting Regression: Actual vs Predicted')
plt.tight_layout()
plt.savefig('gradient_boosting_scatter.png', dpi=150)
plt.close()

# ---------- Protocol 2: K-Means Clustering (Shot Put) ----------
print("\n--- Protocol 2: K-Means Clustering ---")

# Filter for missing Mark OR Result Score
shotput_p2 = shotput_df.dropna(subset=['Mark', 'Result Score']).copy()

# Fit StandardScaler and transform
scaler_p2 = StandardScaler()
scaled_data_p2 = scaler_p2.fit_transform(shotput_p2[['Mark', 'Result Score']])

# Configure KMeans
kmeans = KMeans(
    n_clusters=4,
    init='k-means++',
    n_init=15,
    max_iter=400,
    random_state=42
)
cluster_labels_p2 = kmeans.fit_predict(scaled_data_p2)

# Calculate silhouette coefficient
silhouette_p2 = silhouette_score(scaled_data_p2, cluster_labels_p2)
print(f"Silhouette Coefficient: {silhouette_p2:.3f}")

# Generate hexbin density diagram
plt.figure(figsize=(8, 6))
hb = plt.hexbin(scaled_data_p2[:, 0], scaled_data_p2[:, 1],
                C=cluster_labels_p2, gridsize=15, cmap='viridis', mincnt=1)
plt.colorbar(hb, label='Cluster Assignment')
plt.scatter(scaled_data_p2[:, 0], scaled_data_p2[:, 1], c=cluster_labels_p2,
            cmap='viridis', edgecolors='white', linewidth=0.5, s=50, alpha=0.8)
plt.xlabel('Standardized Mark')
plt.ylabel('Standardized Result Score')
plt.title('K-Means Clustering: Shot Put Performance Profiles')
plt.tight_layout()
plt.savefig('kmeans_hexbin.png', dpi=150)
plt.close()

# ---------- Protocol 3: Random Forest Multiclass Classification (Javelin) ----------
print("\n--- Protocol 3: Random Forest Multiclass Classification ---")

# Filter for missing Nationality OR Mark OR Result Score
javelin_p3 = javelin_df.dropna(subset=['Nationality', 'Mark', 'Result Score']).copy()

# Filter out classes with fewer than 2 members (required for stratified split)
class_counts = javelin_p3['Nationality'].value_counts()
valid_classes = class_counts[class_counts >= 2].index
javelin_p3 = javelin_p3[javelin_p3['Nationality'].isin(valid_classes)]

# Prepare features and target
X_p3 = javelin_p3[['Mark', 'Result Score']]
y_p3 = javelin_p3['Nationality']

# Stratified split 70/30
X_train_p3, X_val_p3, y_train_p3, y_val_p3 = train_test_split(
    X_p3, y_p3, test_size=0.30, stratify=y_p3, random_state=42
)

# Configure RandomForestClassifier
rf = RandomForestClassifier(
    n_estimators=100,
    max_depth=5,
    min_samples_split=4,
    min_samples_leaf=2,
    random_state=42
)
rf.fit(X_train_p3, y_train_p3)

# Predict and calculate macro-averaged F1 score
y_pred_p3 = rf.predict(X_val_p3)
f1_macro_p3 = f1_score(y_val_p3, y_pred_p3, average='macro')
print(f"Macro-averaged F1 Score: {f1_macro_p3:.4f}")

# Generate bar diagram for prediction frequency
pred_counts = pd.Series(y_pred_p3).value_counts().sort_index()
plt.figure(figsize=(10, 6))
plt.bar(pred_counts.index, pred_counts.values, color='steelblue', edgecolor='black')
plt.xlabel('Nationality')
plt.ylabel('Prediction Frequency Count')
plt.title('Random Forest: Nationality Prediction Distribution')
plt.xticks(rotation=45, ha='right')
plt.tight_layout()
plt.savefig('rf_nationality_bar.png', dpi=150)
plt.close()

# ---------- Protocol 4: Principal Component Analysis (Discus) ----------
print("\n--- Protocol 4: Principal Component Analysis ---")

# Filter for missing Mark OR Result Score
discus_p4 = discus_df.dropna(subset=['Mark', 'Result Score']).copy()

# Fit StandardScaler and transform
scaler_p4 = StandardScaler()
scaled_data_p4 = scaler_p4.fit_transform(discus_p4[['Mark', 'Result Score']])

# Configure PCA
pca = PCA(n_components=2, random_state=42)
pca_scores = pca.fit_transform(scaled_data_p4)

# Get explained variance ratio for first component
evr_pc1 = pca.explained_variance_ratio_[0]
print(f"Explained Variance Ratio (PC1): {evr_pc1:.4f}")

# Generate biplot
plt.figure(figsize=(8, 6))
plt.scatter(pca_scores[:, 0], pca_scores[:, 1], alpha=0.6, edgecolors='k', linewidth=0.5)

# Add loading vectors
feature_names = ['Mark', 'Result Score']
loadings = pca.components_.T * np.sqrt(pca.explained_variance_)
scale_factor = 3
for i, feature in enumerate(feature_names):
    plt.arrow(0, 0, loadings[i, 0] * scale_factor, loadings[i, 1] * scale_factor,
              head_width=0.1, head_length=0.05, fc='red', ec='red')
    plt.text(loadings[i, 0] * scale_factor * 1.15, loadings[i, 1] * scale_factor * 1.15,
             feature, color='red', fontsize=10, ha='center')

plt.xlabel('First Principal Component')
plt.ylabel('Second Principal Component')
plt.title('PCA Biplot: Discus Throw Performance')
plt.axhline(y=0, color='gray', linestyle='--', linewidth=0.5)
plt.axvline(x=0, color='gray', linestyle='--', linewidth=0.5)
plt.tight_layout()
plt.savefig('pca_biplot.png', dpi=150)
plt.close()

# ---------- Protocol 5: Hierarchical Clustering Comparison (Discus) ----------
print("\n--- Protocol 5: Hierarchical Clustering Comparison ---")

# Filter for missing Mark OR Result Score
discus_p5 = discus_df.dropna(subset=['Mark', 'Result Score']).copy()

# Fit StandardScaler and transform
scaler_p5 = StandardScaler()
scaled_data_p5 = scaler_p5.fit_transform(discus_p5[['Mark', 'Result Score']])

# Configure first AgglomerativeClustering (Ward linkage)
agg_ward = AgglomerativeClustering(n_clusters=3, linkage='ward', metric='euclidean')
labels_ward = agg_ward.fit_predict(scaled_data_p5)
db_ward = davies_bouldin_score(scaled_data_p5, labels_ward)

# Configure second AgglomerativeClustering (Single linkage)
agg_single = AgglomerativeClustering(n_clusters=3, linkage='single', metric='euclidean')
labels_single = agg_single.fit_predict(scaled_data_p5)
db_single = davies_bouldin_score(scaled_data_p5, labels_single)

# Compare and report best linkage
if db_ward <= db_single:
    best_linkage = 'ward'
else:
    best_linkage = 'single'
print(f"Best Linkage Method: {best_linkage}")

# Generate paired dendrogram
fig, axes = plt.subplots(1, 2, figsize=(14, 6))

# Ward linkage dendrogram
linkage_ward = linkage(scaled_data_p5, method='ward')
dendrogram(linkage_ward, ax=axes[0], leaf_rotation=90, leaf_font_size=8)
axes[0].set_xlabel('Athlete Observation Index')
axes[0].set_ylabel('Linkage Distance')
axes[0].set_title(f'Ward Linkage (DB: {db_ward:.3f})')

# Single linkage dendrogram
linkage_single = linkage(scaled_data_p5, method='single')
dendrogram(linkage_single, ax=axes[1], leaf_rotation=90, leaf_font_size=8)
axes[1].set_xlabel('Athlete Observation Index')
axes[1].set_ylabel('Linkage Distance')
axes[1].set_title(f'Single Linkage (DB: {db_single:.3f})')

plt.tight_layout()
plt.savefig('dendrogram_comparison.png', dpi=150)
plt.close()

# ---------- Protocol 6: Dominant Nationality per Discipline ----------
print("\n--- Protocol 6: Dominant Nationality per Discipline ---")

# Filter each dataset for missing Nationality
discus_p6 = discus_df.dropna(subset=['Nationality'])
shotput_p6 = shotput_df.dropna(subset=['Nationality'])
javelin_p6 = javelin_df.dropna(subset=['Nationality'])

# Compute mode for each (alphabetically first if tie)
def get_mode(series):
    value_counts = series.value_counts()
    max_count = value_counts.max()
    modes = value_counts[value_counts == max_count].index.tolist()
    modes.sort()  # Alphabetically first
    return modes[0]

mode_discus = get_mode(discus_p6['Nationality'])
mode_shotput = get_mode(shotput_p6['Nationality'])
mode_javelin = get_mode(javelin_p6['Nationality'])

# Find nationality appearing as mode in exactly two datasets
modes_list = [mode_discus, mode_shotput, mode_javelin]
nationality_counts = {}
for nat in modes_list:
    nationality_counts[nat] = nationality_counts.get(nat, 0) + 1

nationality_two = None
for nat, count in nationality_counts.items():
    if count == 2:
        nationality_two = nat.upper()
        break

print(f"Nationality appearing as mode in exactly two datasets: {nationality_two}")

# ---------- Protocol 7: Spearman Rank Correlation ----------
print("\n--- Protocol 7: Spearman Rank Correlation ---")

# Filter each dataset for missing Mark OR Result Score
discus_p7 = discus_df.dropna(subset=['Mark', 'Result Score'])
shotput_p7 = shotput_df.dropna(subset=['Mark', 'Result Score'])
javelin_p7 = javelin_df.dropna(subset=['Mark', 'Result Score'])

# Concatenate filtered datasets
combined_p7 = pd.concat([discus_p7, shotput_p7, javelin_p7], ignore_index=True)

# Compute Spearman correlation
spearman_corr, _ = stats.spearmanr(combined_p7['Mark'], combined_p7['Result Score'])
print(f"Spearman Correlation Coefficient: {spearman_corr:.4f}")

# ---------- Protocol 8: Coefficient of Variation ----------
print("\n--- Protocol 8: Coefficient of Variation ---")

# Filter each dataset for missing Mark
discus_p8 = discus_df.dropna(subset=['Mark'])
shotput_p8 = shotput_df.dropna(subset=['Mark'])
javelin_p8 = javelin_df.dropna(subset=['Mark'])

# Calculate CV (population std / mean * 100)
def calc_cv(series):
    mean_val = series.mean()
    # Population standard deviation (ddof=0)
    std_val = series.std(ddof=0)
    return (std_val / mean_val) * 100

cv_discus = calc_cv(discus_p8['Mark'])
cv_shotput = calc_cv(shotput_p8['Mark'])
cv_javelin = calc_cv(javelin_p8['Mark'])

# Find minimum CV discipline
cv_dict = {'discus': cv_discus, 'shot put': cv_shotput, 'javelin': cv_javelin}
min_cv_discipline = min(cv_dict, key=cv_dict.get)
print(f"Discipline with minimum CV: {min_cv_discipline}")

# ---------- Protocol 9: Kruskal-Wallis H-test ----------
print("\n--- Protocol 9: Kruskal-Wallis H-test ---")

# Filter each dataset for missing Result Score
discus_p9 = discus_df.dropna(subset=['Result Score'])
shotput_p9 = shotput_df.dropna(subset=['Result Score'])
javelin_p9 = javelin_df.dropna(subset=['Result Score'])

# Extract Result Score values
scores_discus = discus_p9['Result Score'].values
scores_shotput = shotput_p9['Result Score'].values
scores_javelin = javelin_p9['Result Score'].values

# Compute Kruskal-Wallis H statistic
h_stat, p_value = stats.kruskal(scores_discus, scores_shotput, scores_javelin)
print(f"Kruskal-Wallis H Statistic: {h_stat:.3f}")

# ---------- Protocol 10: Maximum Efficiency Competitor (Javelin) ----------
print("\n--- Protocol 10: Maximum Efficiency Competitor ---")

# Filter for missing Competitor OR Mark OR Result Score
javelin_p10 = javelin_df.dropna(subset=['Competitor', 'Mark', 'Result Score']).copy()

# Compute efficiency ratio (Result Score / Mark)
javelin_p10['Efficiency'] = javelin_p10['Result Score'] / javelin_p10['Mark']

# Find maximum efficiency
max_efficiency = javelin_p10['Efficiency'].max()
max_eff_rows = javelin_p10[javelin_p10['Efficiency'] == max_efficiency]

# Select alphabetically first if tie
max_eff_competitor = sorted(max_eff_rows['Competitor'].tolist())[0]
print(f"Competitor with Maximum Efficiency: {max_eff_competitor}")

# ---------- Open-Ended Question Answer ----------
print("\n--- Open-Ended Question Answer ---")
# Variations in Mark-Result Score relationship across disciplines can be explained by:
# different scoring formulas used per event, biomechanical differences in throwing techniques,
# equipment specifications (discus/shot/javelin weights), and competitive field depth per discipline.
answer = ("Variations in the relationship between throwing distance (Mark) and performance scoring "
          "(Result Score) across disciplines can be explained by different scoring conversion formulas "
          "specific to each event, biomechanical differences in throwing techniques, varying equipment "
          "specifications, and differences in competitive depth across national programs.")
print(f"Answer: {answer}")

print("\n" + "=" * 60)
print("ANALYSIS COMPLETE")
print("=" * 60)
