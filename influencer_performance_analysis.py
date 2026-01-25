# ==========================================
# Influencer Marketing Performance Analysis Script
# Requirements: pandas, numpy, matplotlib, scipy, scikit-learn, catboost, xgboost, lightgbm
# Input files: social media influencers-instagram june 2022 - june 2022.xlsx,
#              social media influencers-youtube june 2022 - june 2022.xlsx,
#              social media influencers-tiktok june 2022 - june 2022.xlsx
# Output files: catboost_feature_importance.png, xgboost_scatter.png, spectral_clusters.png,
#               lightgbm_roc.png, bagging_residuals.png, stacking_confusion.png,
#               agglomerative_dendrogram.png
# ==========================================

import pandas as pd
import numpy as np
import matplotlib.pyplot as plt
import warnings
warnings.filterwarnings('ignore')

from sklearn.model_selection import train_test_split
from sklearn.preprocessing import StandardScaler, LabelEncoder
from sklearn.metrics import (accuracy_score, mean_absolute_error, mean_squared_error,
                             silhouette_score, roc_auc_score, f1_score, confusion_matrix,
                             roc_curve, davies_bouldin_score)
from sklearn.cluster import SpectralClustering, AgglomerativeClustering
from sklearn.ensemble import BaggingRegressor, StackingClassifier, RandomForestClassifier
from sklearn.linear_model import LogisticRegression
from sklearn.neighbors import KNeighborsClassifier
from scipy.cluster.hierarchy import dendrogram, linkage
from scipy.stats import mannwhitneyu
from catboost import CatBoostClassifier
from xgboost import XGBRegressor
from lightgbm import LGBMClassifier

# ---------- Utility Function: Convert M/K Suffix to Numeric ----------
def convert_mk_to_numeric(value):
    """Convert values with M (millions) or K (thousands) suffix to numeric."""
    if pd.isna(value):
        return np.nan
    if isinstance(value, (int, float)):
        return float(value)
    value_str = str(value).strip().replace(',', '')
    if value_str in ["N/A'", "N/A", "-", "", "NaN"]:
        return np.nan
    try:
        if value_str.upper().endswith('M'):
            return float(value_str[:-1]) * 1000000
        elif value_str.upper().endswith('K'):
            return float(value_str[:-1]) * 1000
        else:
            return float(value_str)
    except ValueError:
        return np.nan

# ---------- Load Datasets ----------
print("=" * 60)
print("LOADING DATASETS")
print("=" * 60)

df_instagram = pd.read_excel("social media influencers-instagram june 2022 - june 2022.xlsx")
df_youtube = pd.read_excel("social media influencers-youtube june 2022 - june 2022.xlsx")
df_tiktok = pd.read_excel("social media influencers-tiktok june 2022 - june 2022.xlsx")

print(f"Instagram: {df_instagram.shape[0]} creators")
print(f"YouTube: {df_youtube.shape[0]} channels")
print(f"TikTok: {df_tiktok.shape[0]} personalities")

# ---------- Convert M/K Suffixes to Numeric ----------
print("\n" + "=" * 60)
print("CONVERTING M/K SUFFIXES TO NUMERIC VALUES")
print("=" * 60)

# Instagram columns to convert
ig_numeric_cols = ['Subscribers count', 'Views avg.', 'Likes avg', 'Comments avg.']
for col in ig_numeric_cols:
    if col in df_instagram.columns:
        df_instagram[col] = df_instagram[col].apply(convert_mk_to_numeric)

# YouTube columns to convert
yt_numeric_cols = ['Subscribers count', 'Views avg.', 'Likes avg', 'Comments avg.']
for col in yt_numeric_cols:
    if col in df_youtube.columns:
        df_youtube[col] = df_youtube[col].apply(convert_mk_to_numeric)

# TikTok columns to convert
tt_numeric_cols = ['Subscribers count', 'Views avg.', 'Likes avg', 'Comments avg.', 'Shares avg']
for col in tt_numeric_cols:
    if col in df_tiktok.columns:
        df_tiktok[col] = df_tiktok[col].apply(convert_mk_to_numeric)

print("Numeric conversion completed for all three datasets.")

# ==========================================================
# ANALYSIS 1: CatBoost Classification - Instagram High Performers
# ==========================================================
print("\n" + "=" * 60)
print("ANALYSIS 1: CatBoost Classification - Instagram High Performers")
print("=" * 60)

# Prepare data
df_ig_catboost = df_instagram[['Subscribers count', 'Likes avg']].copy()
df_ig_catboost = df_ig_catboost.dropna()

# Engineer binary outcome: high_performer = 1 if Likes avg > 300000, else 0
df_ig_catboost['high_performer'] = (df_ig_catboost['Likes avg'] > 300000).astype(int)

# Prepare features and target
X_catboost = df_ig_catboost[['Subscribers count']]
y_catboost = df_ig_catboost['high_performer']

# Split data 70/30
X_train_cb, X_test_cb, y_train_cb, y_test_cb = train_test_split(
    X_catboost, y_catboost, test_size=0.30, random_state=42
)

# Train CatBoost model
catboost_model = CatBoostClassifier(
    iterations=100, depth=6, learning_rate=0.1, random_state=42, verbose=0
)
catboost_model.fit(X_train_cb, y_train_cb)

# Predict and evaluate
y_pred_cb = catboost_model.predict(X_test_cb)
catboost_accuracy = accuracy_score(y_test_cb, y_pred_cb) * 100

print(f"CatBoost Classification Accuracy: {catboost_accuracy:.2f}%")

# Feature importance plot
feature_importance = catboost_model.get_feature_importance()
feature_names = ['Subscribers count']

plt.figure(figsize=(8, 6))
plt.bar(feature_names, feature_importance, color='steelblue', edgecolor='black')
plt.xlabel('Feature Names')
plt.ylabel('Importance Scores')
plt.title('CatBoost Feature Importance - Instagram High Performers')
plt.tight_layout()
plt.savefig('catboost_feature_importance.png', dpi=150)
plt.close()
print("Saved: catboost_feature_importance.png")

# ==========================================================
# ANALYSIS 2: XGBoost Regression - YouTube Views Prediction
# ==========================================================
print("\n" + "=" * 60)
print("ANALYSIS 2: XGBoost Regression - YouTube Views Prediction")
print("=" * 60)

# Prepare data
df_yt_xgb = df_youtube[['Subscribers count', 'Views avg.']].copy()
df_yt_xgb = df_yt_xgb.dropna()

# Prepare features and target
X_xgb = df_yt_xgb[['Subscribers count']]
y_xgb = df_yt_xgb['Views avg.']

# Split data 75/25
X_train_xgb, X_test_xgb, y_train_xgb, y_test_xgb = train_test_split(
    X_xgb, y_xgb, test_size=0.25, random_state=42
)

# Train XGBoost model
xgb_model = XGBRegressor(
    n_estimators=100, max_depth=8, learning_rate=0.1, random_state=42
)
xgb_model.fit(X_train_xgb, y_train_xgb)

# Predict and evaluate
y_pred_xgb = xgb_model.predict(X_test_xgb)
xgb_mae = mean_absolute_error(y_test_xgb, y_pred_xgb)

print(f"XGBoost Mean Absolute Error: {xgb_mae:.2f}")

# Scatter plot: Actual vs Predicted
plt.figure(figsize=(8, 6))
plt.scatter(y_test_xgb, y_pred_xgb, alpha=0.6, edgecolor='k', s=50)
plt.xlabel('Actual View Counts')
plt.ylabel('Predicted View Counts')
plt.title('XGBoost: Actual vs Predicted YouTube Views')
min_val = min(y_test_xgb.min(), y_pred_xgb.min())
max_val = max(y_test_xgb.max(), y_pred_xgb.max())
plt.plot([min_val, max_val], [min_val, max_val], 'r--', linewidth=2, label='Perfect Prediction')
plt.legend()
plt.tight_layout()
plt.savefig('xgboost_scatter.png', dpi=150)
plt.close()
print("Saved: xgboost_scatter.png")

# ==========================================================
# ANALYSIS 3: Spectral Clustering - TikTok Creators
# ==========================================================
print("\n" + "=" * 60)
print("ANALYSIS 3: Spectral Clustering - TikTok Creators")
print("=" * 60)

# Prepare data
df_tt_spectral = df_tiktok[['Subscribers count', 'Likes avg', 'Comments avg.']].copy()
df_tt_spectral = df_tt_spectral.dropna()

# Compile features for clustering
X_spectral = df_tt_spectral[['Subscribers count', 'Likes avg', 'Comments avg.']].values

# Standardize features
scaler_spectral = StandardScaler()
X_spectral_scaled = scaler_spectral.fit_transform(X_spectral)

# Apply Spectral Clustering
spectral_model = SpectralClustering(
    n_clusters=5, affinity='nearest_neighbors', random_state=42
)
spectral_labels = spectral_model.fit_predict(X_spectral_scaled)

# Compute silhouette score
spectral_silhouette = silhouette_score(X_spectral_scaled, spectral_labels)

print(f"Spectral Clustering Silhouette Score: {spectral_silhouette:.4f}")

# Scatter plot: Subscribers vs Likes colored by cluster
plt.figure(figsize=(10, 7))
scatter = plt.scatter(
    df_tt_spectral['Subscribers count'], df_tt_spectral['Likes avg'],
    c=spectral_labels, cmap='viridis', alpha=0.7, edgecolor='k', s=50
)
plt.colorbar(scatter, label='Cluster Assignment')
plt.xlabel('Subscribers Count')
plt.ylabel('Likes Avg')
plt.title('Spectral Clustering: TikTok Creators')
plt.tight_layout()
plt.savefig('spectral_clusters.png', dpi=150)
plt.close()
print("Saved: spectral_clusters.png")

# ==========================================================
# ANALYSIS 4: LightGBM Classification - YouTube Category
# ==========================================================
print("\n" + "=" * 60)
print("ANALYSIS 4: LightGBM Classification - YouTube Category")
print("=" * 60)

# Prepare data
df_yt_lgbm = df_youtube[['Subscribers count', 'Views avg.', 'Likes avg', 'Category']].copy()
df_yt_lgbm = df_yt_lgbm.dropna()

# Filter out categories with fewer than 2 samples to enable proper train/test split
category_counts = df_yt_lgbm['Category'].value_counts()
valid_categories = category_counts[category_counts >= 2].index
df_yt_lgbm = df_yt_lgbm[df_yt_lgbm['Category'].isin(valid_categories)]

# Encode category labels
label_encoder = LabelEncoder()
df_yt_lgbm['Category_encoded'] = label_encoder.fit_transform(df_yt_lgbm['Category'])

# Prepare features and target
X_lgbm = df_yt_lgbm[['Subscribers count', 'Views avg.', 'Likes avg']]
y_lgbm = df_yt_lgbm['Category_encoded']

# Split data 80/20
X_train_lgbm, X_test_lgbm, y_train_lgbm, y_test_lgbm = train_test_split(
    X_lgbm, y_lgbm, test_size=0.20, random_state=42
)

# Train LightGBM model
lgbm_model = LGBMClassifier(
    n_estimators=100, max_depth=7, learning_rate=0.1, random_state=42, verbose=-1
)
lgbm_model.fit(X_train_lgbm, y_train_lgbm)

# Predict probabilities for ROC AUC
y_proba_lgbm = lgbm_model.predict_proba(X_test_lgbm)
n_classes = len(label_encoder.classes_)

# Calculate multi-class ROC AUC using One-vs-Rest approach manually
from sklearn.preprocessing import label_binarize
all_classes = list(range(n_classes))
y_test_binarized = label_binarize(y_test_lgbm, classes=all_classes)

# Compute ROC AUC for each class and average (handling single-class edge case)
auc_scores = []
for i in range(n_classes):
    if len(np.unique(y_test_binarized[:, i])) > 1:  # Check if both classes exist
        try:
            class_auc = roc_auc_score(y_test_binarized[:, i], y_proba_lgbm[:, i])
            auc_scores.append(class_auc)
        except:
            pass

lgbm_roc_auc = np.mean(auc_scores) if auc_scores else 0.0

print(f"LightGBM Multi-class ROC AUC: {lgbm_roc_auc:.4f}")

# ROC Curve Plot
plt.figure(figsize=(10, 8))
colors = plt.cm.tab10(np.linspace(0, 1, 10))

# Get classes with samples in test set for plotting
classes_in_test = np.unique(y_test_lgbm)
plot_count = 0

for idx, cls_idx in enumerate(classes_in_test[:5]):  # Plot first 5 classes
    if len(np.unique(y_test_binarized[:, cls_idx])) > 1:
        fpr, tpr, _ = roc_curve(y_test_binarized[:, cls_idx], y_proba_lgbm[:, cls_idx])
        class_name = label_encoder.classes_[cls_idx][:20]
        plt.plot(fpr, tpr, color=colors[plot_count % 10], alpha=0.7, label=f'{class_name}')
        plot_count += 1

plt.plot([0, 1], [0, 1], 'k--', linewidth=2, label='Random Classifier')
plt.xlabel('False Positive Rate')
plt.ylabel('True Positive Rate')
plt.title(f'LightGBM ROC Curves - YouTube Categories (Avg AUC: {lgbm_roc_auc:.4f})')
plt.legend(loc='lower right', fontsize=8)
plt.tight_layout()
plt.savefig('lightgbm_roc.png', dpi=150)
plt.close()
print("Saved: lightgbm_roc.png")

# ==========================================================
# ANALYSIS 5: Bagging Regressor - Instagram Engagement
# ==========================================================
print("\n" + "=" * 60)
print("ANALYSIS 5: Bagging Regressor - Instagram Engagement")
print("=" * 60)

# Prepare data
df_ig_bagging = df_instagram[['Subscribers count', 'Comments avg.']].copy()
df_ig_bagging = df_ig_bagging.dropna()

# Prepare features and target
X_bagging = df_ig_bagging[['Subscribers count']]
y_bagging = df_ig_bagging['Comments avg.']

# Split data 70/30
X_train_bag, X_test_bag, y_train_bag, y_test_bag = train_test_split(
    X_bagging, y_bagging, test_size=0.30, random_state=42
)

# Train Bagging Regressor
bagging_model = BaggingRegressor(
    n_estimators=50, max_samples=0.8, max_features=1.0, random_state=42
)
bagging_model.fit(X_train_bag, y_train_bag)

# Predict and evaluate
y_pred_bag = bagging_model.predict(X_test_bag)
bagging_rmse = np.sqrt(mean_squared_error(y_test_bag, y_pred_bag))

print(f"Bagging Regressor RMSE: {bagging_rmse:.2f}")

# Residual plot
residuals = y_test_bag.values - y_pred_bag

plt.figure(figsize=(8, 6))
plt.scatter(y_pred_bag, residuals, alpha=0.6, edgecolor='k', s=50)
plt.axhline(y=0, color='r', linestyle='--', linewidth=2)
plt.xlabel('Predicted Comments Avg')
plt.ylabel('Residuals (Actual - Predicted)')
plt.title('Bagging Regressor: Residual Plot - Instagram Engagement')
plt.tight_layout()
plt.savefig('bagging_residuals.png', dpi=150)
plt.close()
print("Saved: bagging_residuals.png")

# ==========================================================
# ANALYSIS 6: Stacking Classifier - TikTok Viral Prediction
# ==========================================================
print("\n" + "=" * 60)
print("ANALYSIS 6: Stacking Classifier - TikTok Viral Prediction")
print("=" * 60)

# Prepare data
df_tt_stacking = df_tiktok[['Subscribers count', 'Likes avg']].copy()
df_tt_stacking = df_tt_stacking.dropna()

# Define binary target: viral = 1 if Likes avg > 1000000, else 0
df_tt_stacking['viral'] = (df_tt_stacking['Likes avg'] > 1000000).astype(int)

# Prepare features and target
X_stacking = df_tt_stacking[['Subscribers count', 'Likes avg']]
y_stacking = df_tt_stacking['viral']

# Split data 75/25
X_train_stack, X_test_stack, y_train_stack, y_test_stack = train_test_split(
    X_stacking, y_stacking, test_size=0.25, random_state=42
)

# Define base estimators
base_estimators = [
    ('lr', LogisticRegression(max_iter=1000, random_state=42)),
    ('rf', RandomForestClassifier(n_estimators=50, random_state=42)),
    ('knn', KNeighborsClassifier(n_neighbors=5))
]

# Define stacking classifier
stacking_model = StackingClassifier(
    estimators=base_estimators,
    final_estimator=LogisticRegression(max_iter=1000, random_state=42)
)
stacking_model.fit(X_train_stack, y_train_stack)

# Predict and evaluate
y_pred_stack = stacking_model.predict(X_test_stack)
stacking_f1 = f1_score(y_test_stack, y_pred_stack, average='macro')

print(f"Stacking Classifier Macro F1 Score: {stacking_f1:.4f}")

# Confusion matrix heatmap
cm = confusion_matrix(y_test_stack, y_pred_stack)

plt.figure(figsize=(8, 6))
plt.imshow(cm, interpolation='nearest', cmap='Blues')
plt.colorbar()
plt.xlabel('Predicted Classes')
plt.ylabel('Actual Classes')
plt.title('Stacking Classifier: Confusion Matrix - TikTok Viral')
plt.xticks([0, 1], ['Not Viral (0)', 'Viral (1)'])
plt.yticks([0, 1], ['Not Viral (0)', 'Viral (1)'])
for i in range(2):
    for j in range(2):
        plt.text(j, i, str(cm[i, j]), ha='center', va='center', fontsize=14,
                 color='white' if cm[i, j] > cm.max()/2 else 'black')
plt.tight_layout()
plt.savefig('stacking_confusion.png', dpi=150)
plt.close()
print("Saved: stacking_confusion.png")

# ==========================================================
# ANALYSIS 7: Agglomerative Clustering - Instagram Segmentation
# ==========================================================
print("\n" + "=" * 60)
print("ANALYSIS 7: Agglomerative Clustering - Instagram Segmentation")
print("=" * 60)

# Prepare data
df_ig_agglo = df_instagram[['Subscribers count', 'Views avg.']].copy()
df_ig_agglo = df_ig_agglo.dropna()

# Compile features for clustering
X_agglo = df_ig_agglo[['Subscribers count', 'Views avg.']].values

# Standardize features
scaler_agglo = StandardScaler()
X_agglo_scaled = scaler_agglo.fit_transform(X_agglo)

# Apply Agglomerative Clustering
agglo_model = AgglomerativeClustering(
    n_clusters=4, linkage='ward', metric='euclidean'
)
agglo_labels = agglo_model.fit_predict(X_agglo_scaled)

# Compute Davies-Bouldin index
agglo_db_score = davies_bouldin_score(X_agglo_scaled, agglo_labels)

print(f"Agglomerative Clustering Davies-Bouldin Score: {agglo_db_score:.3f}")

# Dendrogram plot
Z = linkage(X_agglo_scaled, method='ward')

plt.figure(figsize=(14, 8))
dendrogram(Z, truncate_mode='lastp', p=50, leaf_rotation=90, leaf_font_size=8,
           show_contracted=True)
plt.xlabel('Observation Indices')
plt.ylabel('Linkage Distances')
plt.title('Agglomerative Clustering: Dendrogram - Instagram Influencers')
plt.tight_layout()
plt.savefig('agglomerative_dendrogram.png', dpi=150)
plt.close()
print("Saved: agglomerative_dendrogram.png")

# ==========================================================
# ANALYSIS 8: Mann-Whitney U Test - YouTube vs TikTok
# ==========================================================
print("\n" + "=" * 60)
print("ANALYSIS 8: Mann-Whitney U Test - YouTube vs TikTok Engagement")
print("=" * 60)

# Extract Likes avg from both platforms
youtube_likes = df_youtube['Likes avg'].dropna()
tiktok_likes = df_tiktok['Likes avg'].dropna()

# Perform Mann-Whitney U test
u_statistic, p_value = mannwhitneyu(youtube_likes, tiktok_likes, alternative='two-sided')

print(f"Mann-Whitney U-Statistic: {u_statistic:.2f}")
print(f"P-value: {p_value:.6f}")

# ==========================================================
# STRATEGIC INSIGHTS: CatBoost vs XGBoost Comparison
# ==========================================================
print("\n" + "=" * 60)
print("STRATEGIC INSIGHTS: CatBoost vs XGBoost Comparison")
print("=" * 60)

# One sentence answer for the open-ended question
strategic_insight = """
Strategic Insight: CatBoost demonstrated strong classification capability with {:.2f}% accuracy
for identifying high-performing Instagram influencers based solely on subscriber count, while
XGBoost's regression approach for YouTube views prediction yielded a MAE of {:.2f}, suggesting
that gradient boosting classifiers may be better suited for binary engagement segmentation tasks
whereas regression models face challenges with high-variance viewership metrics that exhibit
non-linear relationships with subscriber bases.
""".format(catboost_accuracy, xgb_mae)

print(strategic_insight)

# ==========================================================
# KEY OUTPUT SUMMARY
# ==========================================================
print("\n" + "=" * 60)
print("KEY OUTPUT SUMMARY")
print("=" * 60)
print(f"1. CatBoost Classification Accuracy: {catboost_accuracy:.2f}%")
print(f"2. XGBoost Mean Absolute Error: {xgb_mae:.2f}")
print(f"3. Spectral Clustering Silhouette Score: {spectral_silhouette:.4f}")
print(f"4. LightGBM Multi-class ROC AUC: {lgbm_roc_auc:.4f}")
print(f"5. Bagging Regressor RMSE: {bagging_rmse:.2f}")
print(f"6. Stacking Classifier Macro F1 Score: {stacking_f1:.4f}")
print(f"7. Agglomerative Clustering Davies-Bouldin Score: {agglo_db_score:.3f}")
print(f"8. Mann-Whitney U-Statistic: {u_statistic:.2f}")
print("=" * 60)
