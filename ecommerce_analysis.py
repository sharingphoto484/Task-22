# ==========================================
# Integrated E-commerce Product & Seller Analysis Script
# ==========================================
# Requirements: pandas, numpy, matplotlib, scipy, scikit-learn, statsmodels
# Input files: Products.xlsx, Categories.xlsx, Sellers.xlsx
# Output files: quantile_scatter.png, kmeans_scatter.png,
#               elasticnet_path.png, poisson_bar.png, pca_biplot.png
# ==========================================

import pandas as pd
import numpy as np
import matplotlib.pyplot as plt
from sklearn.preprocessing import StandardScaler
from sklearn.cluster import KMeans
from sklearn.linear_model import ElasticNetCV, PoissonRegressor
from sklearn.decomposition import PCA
from scipy.stats import pearsonr
import statsmodels.api as sm
import statsmodels.formula.api as smf
from statsmodels.regression.quantile_regression import QuantReg
import warnings
warnings.filterwarnings('ignore')

# ---------- Load Excel Files Robustly ----------
print("Loading datasets...")
products = pd.read_excel('Products.xlsx')
categories = pd.read_excel('Categories.xlsx')
sellers = pd.read_excel('Sellers.xlsx')

print(f"Products: {products.shape[0]} rows")
print(f"Categories: {categories.shape[0]} rows")
print(f"Sellers: {sellers.shape[0]} rows\n")

# ==========================================
# ANALYSIS 1: Quantile Regression (90th Percentile)
# ==========================================
print("=" * 60)
print("ANALYSIS 1: Quantile Regression - 90th Percentile Weight")
print("=" * 60)

# ---------- Filter Complete Cases ----------
qr_data = products[['product_weight_g', 'product_length_cm',
                     'product_height_cm', 'product_width_cm']].dropna()
print(f"Complete cases for quantile regression: {len(qr_data)}")

# ---------- Fit Quantile Regression Model ----------
X_qr = qr_data[['product_length_cm', 'product_height_cm', 'product_width_cm']]
X_qr = sm.add_constant(X_qr)
y_qr = qr_data['product_weight_g']

qr_model = QuantReg(y_qr, X_qr)
qr_results = qr_model.fit(q=0.90)

length_coef = qr_results.params['product_length_cm']
print(f"✓ 90th Percentile Coefficient for product_length_cm: {length_coef:.4f}")

# ---------- Generate Scatter Plot ----------
plt.figure(figsize=(10, 6))
plt.scatter(qr_data['product_length_cm'], qr_data['product_weight_g'],
            alpha=0.3, s=10, color='steelblue')
plt.xlabel('Product Length (cm)', fontsize=12)
plt.ylabel('Product Weight (g)', fontsize=12)
plt.title('Product Weight vs Length - Quantile Regression', fontsize=14, fontweight='bold')
plt.grid(True, alpha=0.3)
plt.tight_layout()
plt.savefig('quantile_scatter.png', dpi=150, bbox_inches='tight')
plt.close()
print("✓ Saved: quantile_scatter.png\n")

# ==========================================
# ANALYSIS 2: K-Means Clustering (k=4)
# ==========================================
print("=" * 60)
print("ANALYSIS 2: K-Means Clustering - Product Segmentation")
print("=" * 60)

# ---------- Select Complete Cases ----------
kmeans_data = products[['product_weight_g', 'product_length_cm',
                         'product_height_cm', 'product_width_cm']].dropna()
print(f"Complete cases for k-means: {len(kmeans_data)}")

# ---------- Standardize Features ----------
scaler_kmeans = StandardScaler()
X_kmeans = scaler_kmeans.fit_transform(kmeans_data)

# ---------- Fit K-Means with k=4 ----------
kmeans = KMeans(n_clusters=4, random_state=42, n_init=10)
clusters = kmeans.fit_predict(X_kmeans)

# ---------- Identify Largest Cluster ----------
cluster_counts = pd.Series(clusters).value_counts().sort_index()
largest_cluster = cluster_counts.idxmax()
print(f"Cluster sizes: {dict(cluster_counts)}")
print(f"✓ Largest cluster identifier: {largest_cluster}")

# ---------- Generate Scatter Plot ----------
plt.figure(figsize=(10, 6))
plt.scatter(kmeans_data['product_weight_g'], kmeans_data['product_length_cm'],
            c=clusters, cmap='viridis', alpha=0.5, s=10)
plt.xlabel('Product Weight (g)', fontsize=12)
plt.ylabel('Product Length (cm)', fontsize=12)
plt.title('K-Means Clustering (k=4) - Physical Characteristics', fontsize=14, fontweight='bold')
plt.colorbar(label='Cluster ID')
plt.grid(True, alpha=0.3)
plt.tight_layout()
plt.savefig('kmeans_scatter.png', dpi=150, bbox_inches='tight')
plt.close()
print("✓ Saved: kmeans_scatter.png\n")

# ==========================================
# ANALYSIS 3: Elastic Net Regression
# ==========================================
print("=" * 60)
print("ANALYSIS 3: Elastic Net - Description Length Prediction")
print("=" * 60)

# ---------- Select Complete Cases ----------
enet_data = products[['product_description_lenght', 'product_name_lenght',
                       'product_photos_qty', 'product_weight_g',
                       'product_length_cm']].dropna()
print(f"Complete cases for elastic net: {len(enet_data)}")

# ---------- Prepare Features and Target ----------
X_enet = enet_data[['product_name_lenght', 'product_photos_qty',
                     'product_weight_g', 'product_length_cm']]
y_enet = enet_data['product_description_lenght']

# ---------- Standardize Predictors ----------
scaler_enet = StandardScaler()
X_enet_scaled = scaler_enet.fit_transform(X_enet)
X_enet_df = pd.DataFrame(X_enet_scaled, columns=X_enet.columns)

# ---------- Fit Elastic Net with CV ----------
enet_model = ElasticNetCV(l1_ratio=0.5, cv=5, random_state=42, max_iter=10000)
enet_model.fit(X_enet_scaled, y_enet)

# ---------- Extract Coefficient for product_name_lenght ----------
feature_names = X_enet.columns.tolist()
name_length_idx = feature_names.index('product_name_lenght')
name_length_coef = enet_model.coef_[name_length_idx]
print(f"✓ Standardized coefficient for product_name_lenght: {name_length_coef:.4f}")

# ---------- Generate Regularization Path Plot ----------
# For visualization, we'll plot coefficients across different alpha values
from sklearn.linear_model import enet_path
alphas_enet, coefs_enet, _ = enet_path(X_enet_scaled, y_enet, l1_ratio=0.5, eps=1e-4)

plt.figure(figsize=(10, 6))
for i, feature in enumerate(feature_names):
    plt.plot(np.log(alphas_enet), coefs_enet[i], label=feature)
plt.xlabel('Log(Lambda Penalty)', fontsize=12)
plt.ylabel('Standardized Coefficient', fontsize=12)
plt.title('Elastic Net Regularization Path', fontsize=14, fontweight='bold')
plt.legend(loc='best', fontsize=9)
plt.grid(True, alpha=0.3)
plt.tight_layout()
plt.savefig('elasticnet_path.png', dpi=150, bbox_inches='tight')
plt.close()
print("✓ Saved: elasticnet_path.png\n")

# ==========================================
# ANALYSIS 4: Poisson Regression - Photo Count
# ==========================================
print("=" * 60)
print("ANALYSIS 4: Poisson Regression - Photo Count Modeling")
print("=" * 60)

# ---------- Select Complete Cases ----------
poisson_data = products[['product_photos_qty', 'product_weight_g',
                          'product_description_lenght']].dropna()
print(f"Complete cases for Poisson regression: {len(poisson_data)}")

# ---------- Create Binary Predictors ----------
median_weight = poisson_data['product_weight_g'].median()
median_desc = poisson_data['product_description_lenght'].median()

poisson_data = poisson_data.copy()
poisson_data['heavy_product'] = (poisson_data['product_weight_g'] >= median_weight).astype(int)
poisson_data['long_description'] = (poisson_data['product_description_lenght'] >= median_desc).astype(int)

# ---------- Fit Poisson Regression ----------
X_poisson = poisson_data[['heavy_product', 'long_description']]
y_poisson = poisson_data['product_photos_qty']

poisson_model = PoissonRegressor(max_iter=1000)
poisson_model.fit(X_poisson, y_poisson)

# ---------- Calculate Dispersion Parameter ----------
y_pred_poisson = poisson_model.predict(X_poisson)
deviance = 2 * np.sum(y_poisson * np.log((y_poisson + 1e-10) / (y_pred_poisson + 1e-10)) -
                      (y_poisson - y_pred_poisson))
df_residual = len(y_poisson) - X_poisson.shape[1] - 1
dispersion = deviance / df_residual
print(f"✓ Dispersion parameter: {dispersion:.4f}")

# ---------- Generate Bar Chart ----------
# Create four combinations
combinations = [
    (0, 0, 'Light/Short'),
    (0, 1, 'Light/Long'),
    (1, 0, 'Heavy/Short'),
    (1, 1, 'Heavy/Long')
]

mean_predictions = []
for heavy, long_d, label in combinations:
    pred = poisson_model.predict([[heavy, long_d]])[0]
    mean_predictions.append(pred)

plt.figure(figsize=(10, 6))
plt.bar(range(4), mean_predictions, color=['#1f77b4', '#ff7f0e', '#2ca02c', '#d62728'])
plt.xlabel('Product Category', fontsize=12)
plt.ylabel('Mean Predicted Photo Count', fontsize=12)
plt.title('Poisson Model - Photo Count by Product Characteristics', fontsize=14, fontweight='bold')
plt.xticks(range(4), [c[2] for c in combinations], rotation=15)
plt.grid(True, alpha=0.3, axis='y')
plt.tight_layout()
plt.savefig('poisson_bar.png', dpi=150, bbox_inches='tight')
plt.close()
print("✓ Saved: poisson_bar.png\n")

# ==========================================
# ANALYSIS 5: Principal Component Analysis
# ==========================================
print("=" * 60)
print("ANALYSIS 5: PCA - Dimensionality Reduction")
print("=" * 60)

# ---------- Select Complete Cases ----------
pca_features = ['product_weight_g', 'product_length_cm', 'product_height_cm',
                'product_width_cm', 'product_description_lenght']
pca_data = products[pca_features].dropna()
print(f"Complete cases for PCA: {len(pca_data)}")

# ---------- Standardize Features ----------
scaler_pca = StandardScaler()
X_pca = scaler_pca.fit_transform(pca_data)

# ---------- Fit PCA ----------
pca = PCA()
pca_transformed = pca.fit_transform(X_pca)

# ---------- Calculate Variance Explained by PC1 ----------
pc1_variance = pca.explained_variance_ratio_[0] * 100
print(f"✓ Variance explained by PC1: {pc1_variance:.2f}%")

# ---------- Generate Biplot ----------
plt.figure(figsize=(10, 8))
plt.scatter(pca_transformed[:, 0], pca_transformed[:, 1], alpha=0.3, s=10, color='steelblue')

# Plot loading vectors
loadings = pca.components_[:2, :].T * np.sqrt(pca.explained_variance_[:2])
for i, feature in enumerate(pca_features):
    plt.arrow(0, 0, loadings[i, 0]*3, loadings[i, 1]*3,
              head_width=0.3, head_length=0.3, fc='red', ec='red', alpha=0.7)
    plt.text(loadings[i, 0]*3.2, loadings[i, 1]*3.2, feature,
             fontsize=9, ha='center', va='center', color='darkred', fontweight='bold')

plt.xlabel(f'PC1 ({pca.explained_variance_ratio_[0]*100:.1f}% variance)', fontsize=12)
plt.ylabel(f'PC2 ({pca.explained_variance_ratio_[1]*100:.1f}% variance)', fontsize=12)
plt.title('PCA Biplot - Product Features', fontsize=14, fontweight='bold')
plt.grid(True, alpha=0.3)
plt.axhline(y=0, color='k', linewidth=0.5)
plt.axvline(x=0, color='k', linewidth=0.5)
plt.tight_layout()
plt.savefig('pca_biplot.png', dpi=150, bbox_inches='tight')
plt.close()
print("✓ Saved: pca_biplot.png\n")

# ==========================================
# ANALYSIS 6: Weighted Average Weight by Category
# ==========================================
print("=" * 60)
print("ANALYSIS 6: Weighted Average Product Weight by Category")
print("=" * 60)

# ---------- Merge Products with Categories ----------
merged = products.merge(categories, on='product_category_name', how='inner')
print(f"Merged records: {len(merged)}")

# ---------- Group by Category ----------
category_stats = merged.groupby('product_category_name_english').agg({
    'product_id': 'count',
    'product_weight_g': 'mean'
}).rename(columns={'product_id': 'count', 'product_weight_g': 'mean_weight'})

# ---------- Calculate Weighted Average ----------
category_stats = category_stats.dropna()
total_products = category_stats['count'].sum()
weighted_avg = (category_stats['count'] * category_stats['mean_weight']).sum() / total_products
print(f"✓ Weighted average weight across all categories: {weighted_avg:.2f} grams\n")

# ==========================================
# ANALYSIS 7: Seller Density by State
# ==========================================
print("=" * 60)
print("ANALYSIS 7: Seller Density Analysis by State")
print("=" * 60)

# ---------- Group by State ----------
state_counts = sellers.groupby('seller_state').size().sort_values(ascending=False)
max_state = state_counts.idxmax()
max_count = state_counts.max()
total_sellers = len(sellers)
max_percentage = (max_count / total_sellers) * 100

print(f"State with maximum sellers: {max_state}")
print(f"✓ Percentage of total sellers in {max_state}: {max_percentage:.2f}%\n")

# ==========================================
# ANALYSIS 8: Herfindahl-Hirschman Index
# ==========================================
print("=" * 60)
print("ANALYSIS 8: HHI - State Diversity Measurement")
print("=" * 60)

# ---------- Calculate State Proportions ----------
state_proportions = state_counts / total_sellers

# ---------- Calculate HHI ----------
hhi = (state_proportions ** 2).sum() * 10000
print(f"✓ HHI Index (state concentration): {int(round(hhi))}\n")

# ==========================================
# ANALYSIS 9: Correlation - Weight vs Volume
# ==========================================
print("=" * 60)
print("ANALYSIS 9: Correlation between Weight and Volume")
print("=" * 60)

# ---------- Select Complete Cases ----------
corr_data = products[['product_weight_g', 'product_length_cm',
                       'product_height_cm', 'product_width_cm']].dropna()

# ---------- Calculate Volume ----------
corr_data = corr_data.copy()
corr_data['volume'] = (corr_data['product_length_cm'] *
                       corr_data['product_height_cm'] *
                       corr_data['product_width_cm'])

# ---------- Calculate Pearson Correlation ----------
correlation, _ = pearsonr(corr_data['product_weight_g'], corr_data['volume'])
print(f"✓ Pearson correlation (weight vs volume): {correlation:.4f}\n")

# ==========================================
# ANALYSIS 10: Category Size Distribution
# ==========================================
print("=" * 60)
print("ANALYSIS 10: Category Size Distribution Analysis")
print("=" * 60)

# ---------- Merge and Group ----------
merged_full = products.merge(categories, on='product_category_name', how='inner')
category_sizes = merged_full.groupby('product_category_name_english').size()

# ---------- Calculate Size Ratio ----------
max_size = category_sizes.max()
min_size = category_sizes.min()
size_ratio = max_size / min_size
print(f"Maximum category size: {max_size}")
print(f"Minimum category size: {min_size}")
print(f"✓ Size ratio (max/min): {size_ratio:.2f}\n")

# ==========================================
# ANALYSIS 11: Zip Code Concentration
# ==========================================
print("=" * 60)
print("ANALYSIS 11: Zip Code Concentration Metric")
print("=" * 60)

# ---------- Group by Zip Code ----------
zip_counts = sellers.groupby('seller_zip_code_prefix').size().sort_values(ascending=False)

# ---------- Top 5 Zip Codes ----------
top5_count = zip_counts.head(5).sum()
top5_percentage = (top5_count / total_sellers) * 100
print(f"Top 5 zip codes seller count: {top5_count}")
print(f"✓ Top-five concentration percentage: {top5_percentage:.2f}%\n")

# ==========================================
# ANALYSIS 12: Category Weight Comparison
# ==========================================
print("=" * 60)
print("ANALYSIS 12: Category Weight Comparison")
print("=" * 60)

# ---------- Calculate Mean Weights by Category ----------
merged_weight = products.merge(categories, on='product_category_name', how='inner')
category_weights = merged_weight.groupby('product_category_name_english')['product_weight_g'].mean()

# ---------- Find Max and Min ----------
category_weights = category_weights.dropna()
max_weight_category = category_weights.idxmax()
min_weight_category = category_weights.idxmin()
max_weight = category_weights.max()
min_weight = category_weights.min()
weight_difference = max_weight - min_weight

print(f"Heaviest category: {max_weight_category} ({max_weight:.2f}g)")
print(f"Lightest category: {min_weight_category} ({min_weight:.2f}g)")
print(f"✓ Weight difference: {weight_difference:.2f} grams\n")

# ==========================================
# STRATEGIC RECOMMENDATION
# ==========================================
print("=" * 60)
print("STRATEGIC RECOMMENDATION")
print("=" * 60)
print("\n📊 Product Segmentation Strategy:")
print("The marketplace should prioritize a dimension-based segmentation strategy")
print("focusing on the largest cluster's physical characteristics for inventory")
print("optimization, while leveraging PCA insights to streamline product attribute")
print("collection by focusing on the primary variance drivers identified in PC1.")
print("\n" + "=" * 60)

# ==========================================
# SUMMARY OF KEY OUTPUTS
# ==========================================
print("\n" + "=" * 60)
print("SUMMARY - KEY OUTPUTS")
print("=" * 60)
print(f"1. Quantile Regression - Length Coefficient (90th %ile): {length_coef:.4f}")
print(f"2. K-Means - Largest Cluster ID: {largest_cluster}")
print(f"3. Elastic Net - Name Length Coefficient: {name_length_coef:.4f}")
print(f"4. Poisson - Dispersion Parameter: {dispersion:.4f}")
print(f"5. PCA - Variance Explained by PC1: {pc1_variance:.2f}%")
print(f"6. Weighted Average Weight: {weighted_avg:.2f}g")
print(f"7. Seller Density - {max_state} State: {max_percentage:.2f}%")
print(f"8. HHI Index: {int(round(hhi))}")
print(f"9. Weight-Volume Correlation: {correlation:.4f}")
print(f"10. Category Size Ratio: {size_ratio:.2f}")
print(f"11. Top-5 Zip Code Concentration: {top5_percentage:.2f}%")
print(f"12. Category Weight Difference: {weight_difference:.2f}g")
print("=" * 60)
