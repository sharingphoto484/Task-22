# ==========================================
# Property Market Strategist Analysis Script
# Requirements: pandas, numpy, matplotlib, scipy, scikit-learn
# Input files: berlin_weekdays.xlsx, lisbon_weekdays.xlsx, budapest_weekdays.xlsx
# Output files: lasso_coefficient_trajectory.png, roc_curve_superhost.png,
#               dendrogram_hierarchy.png, elasticnet_prediction_scatter.png,
#               confusion_matrix_roomtype.png, lda_discriminant_projection.png,
#               polynomial_fitted_curve.png
# Actionable Insight: Lasso achieves superior parsimony via aggressive feature elimination making it ideal for identifying core pricing drivers, while Elastic Net balances feature retention and regularization yielding more stable predictions when features exhibit multicollinearity, thus Lasso guides strategic feature focus whereas Elastic Net optimizes operational forecasting accuracy.
# ==========================================

import pandas as pd
import numpy as np
import matplotlib.pyplot as plt
import seaborn as sns
from sklearn.linear_model import Lasso, LogisticRegression, ElasticNet, LinearRegression
from sklearn.cluster import AgglomerativeClustering
from sklearn.naive_bayes import GaussianNB
from sklearn.discriminant_analysis import LinearDiscriminantAnalysis
from sklearn.model_selection import train_test_split
from sklearn.preprocessing import StandardScaler
from sklearn.metrics import (mean_squared_error, roc_auc_score, roc_curve,
                            silhouette_score, f1_score, confusion_matrix, accuracy_score)
from sklearn.preprocessing import PolynomialFeatures
from scipy.stats import kruskal
from scipy.cluster.hierarchy import dendrogram, linkage
import warnings
warnings.filterwarnings('ignore')

# ---------- Load Datasets ----------
print("Loading European property market datasets...")
berlin_df = pd.read_excel('berlin_weekdays.xlsx')
lisbon_df = pd.read_excel('lisbon_weekdays.xlsx')
budapest_df = pd.read_excel('budapest_weekdays.xlsx')

print(f"Berlin properties: {len(berlin_df)}")
print(f"Lisbon properties: {len(lisbon_df)}")
print(f"Budapest properties: {len(budapest_df)}")

# ==========================================
# METHODOLOGY 1: LASSO REGRESSION - LISBON PRICE PREDICTION
# ==========================================
print("\n" + "="*60)
print("METHODOLOGY 1: Lasso Regression - Lisbon Price Prediction")
print("="*60)

# ---------- Prepare Lisbon Data for Lasso ----------
lasso_features = ['person_capacity', 'bedrooms', 'dist', 'metro_dist',
                 'cleanliness_rating', 'guest_satisfaction_overall',
                 'attr_index_norm', 'rest_index_norm']
lasso_target = 'realSum'

lisbon_lasso = lisbon_df[lasso_features + [lasso_target]].dropna()
X_lasso = lisbon_lasso[lasso_features]
y_lasso = lisbon_lasso[lasso_target]

# ---------- Train-Test Split ----------
X_lasso_train, X_lasso_test, y_lasso_train, y_lasso_test = train_test_split(
    X_lasso, y_lasso, test_size=0.2, random_state=42
)

# ---------- Fit Lasso Model ----------
lasso_model = Lasso(alpha=1.0, random_state=42)
lasso_model.fit(X_lasso_train, y_lasso_train)

# ---------- Calculate MSE ----------
y_lasso_pred = lasso_model.predict(X_lasso_test)
lasso_mse = mean_squared_error(y_lasso_test, y_lasso_pred)
print(f"\nLasso MSE: {lasso_mse:.2f}")

# ---------- Generate Coefficient Trajectory Plot ----------
alphas = np.logspace(-2, 1, 100)
coefficients = []

for alpha in alphas:
    lasso_temp = Lasso(alpha=alpha, random_state=42, max_iter=10000)
    lasso_temp.fit(X_lasso_train, y_lasso_train)
    coefficients.append(lasso_temp.coef_)

coefficients = np.array(coefficients)

plt.figure(figsize=(12, 7))
for i, feature in enumerate(lasso_features):
    plt.plot(alphas, coefficients[:, i], label=feature, linewidth=2)
plt.xscale('log')
plt.xlabel('Alpha (Regularization Strength)', fontsize=12)
plt.ylabel('Coefficient Magnitude', fontsize=12)
plt.title('Lasso Coefficient Trajectory: Feature Sparsification Path', fontsize=14, fontweight='bold')
plt.legend(loc='best', fontsize=9)
plt.grid(True, alpha=0.3)
plt.tight_layout()
plt.savefig('lasso_coefficient_trajectory.png', dpi=300, bbox_inches='tight')
plt.close()
print("Saved: lasso_coefficient_trajectory.png")

# ==========================================
# METHODOLOGY 2: LOGISTIC REGRESSION - BERLIN SUPERHOST CLASSIFICATION
# ==========================================
print("\n" + "="*60)
print("METHODOLOGY 2: Logistic Regression - Berlin Superhost Classification")
print("="*60)

# ---------- Prepare Berlin Data for Logistic Regression ----------
logistic_features = ['realSum', 'person_capacity', 'bedrooms', 'cleanliness_rating',
                    'guest_satisfaction_overall', 'metro_dist', 'attr_index_norm']
logistic_target = 'host_is_superhost'

berlin_logistic = berlin_df[logistic_features + [logistic_target]].dropna()
X_logistic = berlin_logistic[logistic_features]
y_logistic = berlin_logistic[logistic_target]

# ---------- Train-Test Split ----------
X_log_train, X_log_test, y_log_train, y_log_test = train_test_split(
    X_logistic, y_logistic, test_size=0.25, random_state=42
)

# ---------- Fit Logistic Regression ----------
log_model = LogisticRegression(penalty='l2', C=1.0, solver='lbfgs',
                               max_iter=1000, random_state=42)
log_model.fit(X_log_train, y_log_train)

# ---------- Calculate AUC ----------
y_log_proba = log_model.predict_proba(X_log_test)[:, 1]
log_auc = roc_auc_score(y_log_test, y_log_proba)
print(f"\nLogistic Regression AUC: {log_auc:.4f}")

# ---------- Generate ROC Curve ----------
fpr, tpr, thresholds = roc_curve(y_log_test, y_log_proba)

plt.figure(figsize=(10, 8))
plt.plot(fpr, tpr, linewidth=3, label=f'ROC Curve (AUC = {log_auc:.4f})', color='#2E86AB')
plt.plot([0, 1], [0, 1], 'k--', linewidth=2, label='Chance Baseline', alpha=0.6)
plt.xlabel('False Positive Rate', fontsize=12)
plt.ylabel('True Positive Rate', fontsize=12)
plt.title('ROC Curve: Superhost Classification Performance', fontsize=14, fontweight='bold')
plt.legend(loc='lower right', fontsize=11)
plt.grid(True, alpha=0.3)
plt.tight_layout()
plt.savefig('roc_curve_superhost.png', dpi=300, bbox_inches='tight')
plt.close()
print("Saved: roc_curve_superhost.png")

# ==========================================
# METHODOLOGY 3: HIERARCHICAL CLUSTERING - BUDAPEST PROPERTIES
# ==========================================
print("\n" + "="*60)
print("METHODOLOGY 3: Hierarchical Clustering - Budapest Properties")
print("="*60)

# ---------- Prepare Budapest Data for Clustering ----------
cluster_features = ['realSum', 'dist', 'metro_dist', 'attr_index_norm']
budapest_cluster = budapest_df[cluster_features].dropna()

# ---------- Normalize Features ----------
scaler_cluster = StandardScaler()
X_cluster_scaled = scaler_cluster.fit_transform(budapest_cluster)

# ---------- Apply Agglomerative Clustering ----------
agg_cluster = AgglomerativeClustering(n_clusters=4, linkage='ward', metric='euclidean')
cluster_labels = agg_cluster.fit_predict(X_cluster_scaled)

# ---------- Calculate Silhouette Score ----------
silhouette_avg = silhouette_score(X_cluster_scaled, cluster_labels)
print(f"\nSilhouette Score: {silhouette_avg:.4f}")

# ---------- Generate Dendrogram ----------
linkage_matrix = linkage(X_cluster_scaled, method='ward', metric='euclidean')

plt.figure(figsize=(14, 8))
dendrogram(linkage_matrix, no_labels=True, color_threshold=None,
          above_threshold_color='#404040')
plt.xlabel('Property Index', fontsize=12)
plt.ylabel('Euclidean Linkage Distance', fontsize=12)
plt.title('Hierarchical Dendrogram: Budapest Property Segmentation', fontsize=14, fontweight='bold')
plt.tight_layout()
plt.savefig('dendrogram_hierarchy.png', dpi=300, bbox_inches='tight')
plt.close()
print("Saved: dendrogram_hierarchy.png")

# ==========================================
# METHODOLOGY 4: ELASTIC NET REGRESSION - LISBON SATISFACTION PREDICTION
# ==========================================
print("\n" + "="*60)
print("METHODOLOGY 4: Elastic Net - Lisbon Satisfaction Prediction")
print("="*60)

# ---------- Prepare Lisbon Data for Elastic Net ----------
elastic_features = ['realSum', 'person_capacity', 'bedrooms', 'cleanliness_rating',
                   'dist', 'attr_index_norm']
elastic_target = 'guest_satisfaction_overall'

lisbon_elastic = lisbon_df[elastic_features + [elastic_target]].dropna()
X_elastic = lisbon_elastic[elastic_features]
y_elastic = lisbon_elastic[elastic_target]

# ---------- Train-Test Split ----------
X_elastic_train, X_elastic_test, y_elastic_train, y_elastic_test = train_test_split(
    X_elastic, y_elastic, test_size=0.3, random_state=42
)

# ---------- Fit Elastic Net Model ----------
elastic_model = ElasticNet(alpha=0.5, l1_ratio=0.5, random_state=42)
elastic_model.fit(X_elastic_train, y_elastic_train)

# ---------- Calculate MAPE ----------
y_elastic_pred = elastic_model.predict(X_elastic_test)
mape_elastic = np.mean(np.abs((y_elastic_test - y_elastic_pred) / y_elastic_test)) * 100
print(f"\nElastic Net MAPE: {mape_elastic:.3f}%")

# ---------- Generate Prediction Scatter Plot ----------
residuals = np.abs(y_elastic_test - y_elastic_pred)

plt.figure(figsize=(10, 8))
scatter = plt.scatter(y_elastic_test, y_elastic_pred, c=residuals, cmap='coolwarm',
                     alpha=0.6, s=50, edgecolors='black', linewidth=0.5)
plt.plot([y_elastic_test.min(), y_elastic_test.max()],
         [y_elastic_test.min(), y_elastic_test.max()],
         'k--', linewidth=2, label='Perfect Prediction', alpha=0.7)
plt.xlabel('Observed Satisfaction Rating', fontsize=12)
plt.ylabel('Predicted Satisfaction Rating', fontsize=12)
plt.title('Elastic Net: Observed vs Predicted Guest Satisfaction', fontsize=14, fontweight='bold')
plt.colorbar(scatter, label='Absolute Deviation')
plt.legend(loc='upper left', fontsize=10)
plt.grid(True, alpha=0.3)
plt.tight_layout()
plt.savefig('elasticnet_prediction_scatter.png', dpi=300, bbox_inches='tight')
plt.close()
print("Saved: elasticnet_prediction_scatter.png")

# ==========================================
# METHODOLOGY 5: GAUSSIAN NAIVE BAYES - BERLIN ROOM TYPE CLASSIFICATION
# ==========================================
print("\n" + "="*60)
print("METHODOLOGY 5: Gaussian Naive Bayes - Berlin Room Type Classification")
print("="*60)

# ---------- Prepare Berlin Data for Naive Bayes ----------
nb_features = ['person_capacity', 'bedrooms', 'cleanliness_rating',
              'guest_satisfaction_overall', 'realSum']
nb_target = 'room_type'

berlin_nb = berlin_df[nb_features + [nb_target]].dropna()
X_nb = berlin_nb[nb_features]
y_nb = berlin_nb[nb_target]

# ---------- Train-Test Split ----------
X_nb_train, X_nb_test, y_nb_train, y_nb_test = train_test_split(
    X_nb, y_nb, test_size=0.2, random_state=42
)

# ---------- Fit Gaussian Naive Bayes ----------
nb_model = GaussianNB()
nb_model.fit(X_nb_train, y_nb_train)

# ---------- Calculate Macro F1 Score ----------
y_nb_pred = nb_model.predict(X_nb_test)
nb_f1 = f1_score(y_nb_test, y_nb_pred, average='macro')
print(f"\nGaussian NB Macro F1: {nb_f1:.4f}")

# ---------- Generate Confusion Matrix ----------
cm = confusion_matrix(y_nb_test, y_nb_pred)
room_types = sorted(y_nb.unique())

plt.figure(figsize=(10, 8))
sns.heatmap(cm, annot=True, fmt='d', cmap='YlOrRd', cbar_kws={'label': 'Frequency'},
           xticklabels=room_types, yticklabels=room_types, linewidths=1, linecolor='black')
plt.xlabel('Predicted Room Type', fontsize=12)
plt.ylabel('Actual Room Type', fontsize=12)
plt.title('Confusion Matrix: Room Type Classification', fontsize=14, fontweight='bold')
plt.tight_layout()
plt.savefig('confusion_matrix_roomtype.png', dpi=300, bbox_inches='tight')
plt.close()
print("Saved: confusion_matrix_roomtype.png")

# ==========================================
# METHODOLOGY 6: LINEAR DISCRIMINANT ANALYSIS - BUDAPEST HOST TYPE
# ==========================================
print("\n" + "="*60)
print("METHODOLOGY 6: Linear Discriminant Analysis - Budapest Host Type")
print("="*60)

# ---------- Prepare Budapest Data for LDA ----------
lda_features = ['realSum', 'person_capacity', 'bedrooms', 'guest_satisfaction_overall',
               'cleanliness_rating', 'dist']
lda_target = 'multi'

budapest_lda = budapest_df[lda_features + [lda_target]].dropna()
X_lda = budapest_lda[lda_features]
y_lda = budapest_lda[lda_target]

# ---------- Train-Test Split ----------
X_lda_train, X_lda_test, y_lda_train, y_lda_test = train_test_split(
    X_lda, y_lda, test_size=0.25, random_state=42
)

# ---------- Fit Linear Discriminant Analysis ----------
n_classes_lda = len(y_lda_train.unique())
n_components_lda = min(2, n_classes_lda - 1)
lda_model = LinearDiscriminantAnalysis(n_components=n_components_lda)
X_lda_transformed = lda_model.fit_transform(X_lda_train, y_lda_train)
X_lda_test_transformed = lda_model.transform(X_lda_test)

# ---------- Calculate Accuracy ----------
y_lda_pred = lda_model.predict(X_lda_test)
lda_accuracy = accuracy_score(y_lda_test, y_lda_pred) * 100
print(f"\nLDA Accuracy: {lda_accuracy:.2f}%")

# ---------- Generate LDA Projection Plot ----------
plt.figure(figsize=(12, 8))
colors = ['#E63946', '#457B9D']
markers = ['o', 's']
multi_values = sorted(y_lda_train.unique())

if n_components_lda == 2:
    for i, multi_val in enumerate(multi_values):
        mask = y_lda_train == multi_val
        plt.scatter(X_lda_transformed[mask, 0], X_lda_transformed[mask, 1],
                   c=colors[i], marker=markers[i], s=60, alpha=0.6,
                   edgecolors='black', linewidth=0.5,
                   label=f'Multi={multi_val}')
    plt.xlabel('First Discriminant Axis (LD1)', fontsize=12)
    plt.ylabel('Second Discriminant Axis (LD2)', fontsize=12)
else:
    # For binary classification, plot LD1 vs index
    for i, multi_val in enumerate(multi_values):
        mask = y_lda_train == multi_val
        indices = np.where(mask)[0]
        plt.scatter(X_lda_transformed[mask, 0], indices,
                   c=colors[i], marker=markers[i], s=60, alpha=0.6,
                   edgecolors='black', linewidth=0.5,
                   label=f'Multi={multi_val}')
    plt.xlabel('First Discriminant Axis (LD1)', fontsize=12)
    plt.ylabel('Sample Index', fontsize=12)

plt.title('LDA Projection: Host Type Separation', fontsize=14, fontweight='bold')
plt.legend(loc='best', fontsize=11)
plt.grid(True, alpha=0.3)
plt.tight_layout()
plt.savefig('lda_discriminant_projection.png', dpi=300, bbox_inches='tight')
plt.close()
print("Saved: lda_discriminant_projection.png")

# ==========================================
# METHODOLOGY 7: POLYNOMIAL REGRESSION DEGREE 3 - BERLIN PRICE-DISTANCE
# ==========================================
print("\n" + "="*60)
print("METHODOLOGY 7: Polynomial Regression (Degree 3) - Berlin Price-Distance")
print("="*60)

# ---------- Prepare Berlin Data for Polynomial Regression ----------
poly_feature = ['dist']
poly_target = 'realSum'

berlin_poly = berlin_df[poly_feature + [poly_target]].dropna()
X_poly = berlin_poly[poly_feature].values
y_poly = berlin_poly[poly_target].values

# ---------- Train-Test Split ----------
X_poly_train, X_poly_test, y_poly_train, y_poly_test = train_test_split(
    X_poly, y_poly, test_size=0.3, random_state=42
)

# ---------- Create Polynomial Features ----------
poly_transformer = PolynomialFeatures(degree=3, include_bias=False)
X_poly_train_transformed = poly_transformer.fit_transform(X_poly_train)
X_poly_test_transformed = poly_transformer.transform(X_poly_test)

# ---------- Fit Linear Regression on Polynomial Features ----------
poly_model = LinearRegression()
poly_model.fit(X_poly_train_transformed, y_poly_train)

# ---------- Calculate R-squared ----------
y_poly_pred = poly_model.predict(X_poly_test_transformed)
poly_r2 = poly_model.score(X_poly_test_transformed, y_poly_test)
print(f"\nPolynomial Regression R²: {poly_r2:.4f}")

# ---------- Generate Polynomial Fitted Curve ----------
X_range = np.linspace(X_poly_test.min(), X_poly_test.max(), 300).reshape(-1, 1)
X_range_transformed = poly_transformer.transform(X_range)
y_range_pred = poly_model.predict(X_range_transformed)

plt.figure(figsize=(12, 8))
plt.scatter(X_poly_test, y_poly_test, alpha=0.4, s=40, c='#2A9D8F',
           edgecolors='black', linewidth=0.3, label='Test Observations')
plt.plot(X_range, y_range_pred, color='#E76F51', linewidth=3,
        label='Polynomial Degree 3 Fit')
plt.xlabel('Distance from City Center (dist)', fontsize=12)
plt.ylabel('Nightly Rental Price (realSum)', fontsize=12)
plt.title('Polynomial Regression: Non-Linear Price-Distance Relationship',
         fontsize=14, fontweight='bold')
plt.legend(loc='best', fontsize=11)
plt.grid(True, alpha=0.3)
plt.tight_layout()
plt.savefig('polynomial_fitted_curve.png', dpi=300, bbox_inches='tight')
plt.close()
print("Saved: polynomial_fitted_curve.png")

# ==========================================
# METHODOLOGY 8: KRUSKAL-WALLIS TEST - CROSS-CITY PRICE COMPARISON
# ==========================================
print("\n" + "="*60)
print("METHODOLOGY 8: Kruskal-Wallis Test - Cross-City Price Comparison")
print("="*60)

# ---------- Extract Price Data from All Cities ----------
berlin_prices = berlin_df['realSum'].dropna().values
lisbon_prices = lisbon_df['realSum'].dropna().values
budapest_prices = budapest_df['realSum'].dropna().values

# ---------- Perform Kruskal-Wallis Test ----------
h_stat, p_value = kruskal(berlin_prices, lisbon_prices, budapest_prices)
print(f"\nKruskal-Wallis H-Statistic: {h_stat:.3f}")
print(f"P-value: {p_value:.6f}")

# ==========================================
# SUMMARY OF KEY RESULTS
# ==========================================
print("\n" + "="*60)
print("SUMMARY: KEY PERFORMANCE METRICS")
print("="*60)
print(f"1. Lasso Price Prediction MSE: {lasso_mse:.2f}")
print(f"2. Logistic Superhost AUC: {log_auc:.4f}")
print(f"3. Hierarchical Clustering Silhouette: {silhouette_avg:.4f}")
print(f"4. Elastic Net Satisfaction MAPE: {mape_elastic:.3f}%")
print(f"5. Gaussian NB Room Type Macro F1: {nb_f1:.4f}")
print(f"6. LDA Host Type Accuracy: {lda_accuracy:.2f}%")
print(f"7. Polynomial Regression R²: {poly_r2:.4f}")
print(f"8. Kruskal-Wallis H-Statistic: {h_stat:.3f}")
print("="*60)

print("\n✓ Analysis Complete: All 8 methodologies executed successfully")
print("✓ Generated 7 visualization outputs")
