# ==========================================
# Integrated Rental Pricing & Analytics Script
# ==========================================
# Requirements: pandas, numpy, matplotlib, scipy, scikit-learn
# Input files: airnb.xlsx, airnb_luxe.xlsx, airnb_desert.xlsx
# Output files: scatter_beds_price.png, scatter_distance_price.png,
#               scatter_price_rating.png, biplot_pca.png, dendrogram.png
# ==========================================

import pandas as pd
import numpy as np
import matplotlib.pyplot as plt
from sklearn.preprocessing import StandardScaler
from sklearn.linear_model import LinearRegression
from sklearn.cluster import KMeans
from sklearn.decomposition import PCA
from sklearn.metrics import r2_score, silhouette_score
from sklearn.preprocessing import PolynomialFeatures
from scipy.cluster.hierarchy import dendrogram, linkage
from scipy.stats import spearmanr
import re
import warnings
warnings.filterwarnings('ignore')

# ---------- Load Excel Files ----------
print("Loading datasets...")
df_airnb = pd.read_excel('airnb.xlsx')
df_luxe = pd.read_excel('airnb_luxe.xlsx')
df_desert = pd.read_excel('airnb_desert.xlsx')

print(f"Loaded {len(df_airnb)} regular listings")
print(f"Loaded {len(df_luxe)} luxury listings")
print(f"Loaded {len(df_desert)} desert listings\n")

# ==========================================
# ANALYSIS 1: Multiple Linear Regression (airnb.xlsx)
# ==========================================
print("=" * 50)
print("ANALYSIS 1: Multiple Linear Regression")
print("=" * 50)

# ---------- Extract Numeric Price ----------
def extract_price(price_str):
    """Remove commas and convert to float"""
    try:
        if pd.isna(price_str):
            return np.nan
        price_clean = str(price_str).replace(',', '').replace('$', '').strip()
        return float(price_clean)
    except:
        return np.nan

df_airnb['price_numeric'] = df_airnb['Price(in dollar)'].apply(extract_price)

# ---------- Extract Numeric Rating ----------
def extract_rating(rating_str):
    """Parse first number from rating string"""
    try:
        if pd.isna(rating_str):
            return np.nan
        # Find first decimal number in string
        match = re.search(r'(\d+\.?\d*)', str(rating_str))
        if match:
            return float(match.group(1))
        return np.nan
    except:
        return np.nan

df_airnb['rating_numeric'] = df_airnb['Review and rating'].apply(extract_rating)

# ---------- Extract Bed Count ----------
def extract_bed_count(bed_str):
    """Parse integer from bed count string"""
    try:
        if pd.isna(bed_str):
            return np.nan
        bed_text = str(bed_str).lower()

        # Map word numbers to integers
        word_to_num = {
            'one': 1, 'two': 2, 'three': 3, 'four': 4, 'five': 5,
            'six': 6, 'seven': 7, 'eight': 8, 'nine': 9, 'ten': 10
        }

        for word, num in word_to_num.items():
            if word in bed_text:
                return num

        # Try to extract numeric value
        match = re.search(r'(\d+)', bed_text)
        if match:
            return int(match.group(1))
        return np.nan
    except:
        return np.nan

df_airnb['bed_count'] = df_airnb['Number of bed'].apply(extract_bed_count)

# ---------- Select Complete Records ----------
df_reg = df_airnb[['price_numeric', 'rating_numeric', 'bed_count']].dropna()
print(f"Complete records for regression: {len(df_reg)}")

# ---------- Standardize Features ----------
scaler_reg = StandardScaler()
X_reg = scaler_reg.fit_transform(df_reg[['bed_count', 'rating_numeric']])
y_reg = df_reg['price_numeric'].values

# ---------- Fit Multiple Linear Regression ----------
lr_model = LinearRegression()
lr_model.fit(X_reg, y_reg)

bed_coefficient = lr_model.coef_[0]
print(f"\nStandardized coefficient for bed count: {bed_coefficient:.4f}")

# ---------- Generate Scatter Plot (Beds vs Price) ----------
plt.figure(figsize=(10, 6))
plt.scatter(df_reg['bed_count'], df_reg['price_numeric'], alpha=0.5, s=30)
plt.xlabel('Number of Beds', fontsize=12)
plt.ylabel('Price (Dollars)', fontsize=12)
plt.title('Price vs Number of Beds - Regular Listings', fontsize=14, fontweight='bold')
plt.grid(True, alpha=0.3)
plt.tight_layout()
plt.savefig('scatter_beds_price.png', dpi=300, bbox_inches='tight')
plt.close()
print("Generated: scatter_beds_price.png")

# ==========================================
# ANALYSIS 2: K-Means Clustering (airnb_luxe.xlsx)
# ==========================================
print("\n" + "=" * 50)
print("ANALYSIS 2: K-Means Clustering")
print("=" * 50)

# ---------- Extract Numeric Price (Luxe) ----------
df_luxe['price_numeric'] = df_luxe['Price(In dollar)'].apply(extract_price)

# ---------- Extract Numeric Distance ----------
def extract_distance(dist_str):
    """Parse kilometers value from distance string"""
    try:
        if pd.isna(dist_str):
            return np.nan
        # Remove commas and extract number before 'kilometers'
        dist_clean = str(dist_str).replace(',', '')
        match = re.search(r'(\d+\.?\d*)', dist_clean)
        if match:
            return float(match.group(1))
        return np.nan
    except:
        return np.nan

df_luxe['distance_numeric'] = df_luxe['Distance'].apply(extract_distance)

# ---------- Select Complete Records ----------
df_kmeans = df_luxe[['price_numeric', 'distance_numeric']].dropna()
print(f"Complete records for clustering: {len(df_kmeans)}")

# ---------- Standardize Features ----------
scaler_kmeans = StandardScaler()
X_kmeans = scaler_kmeans.fit_transform(df_kmeans)

# ---------- Fit K-Means with k=3 ----------
kmeans = KMeans(n_clusters=3, random_state=42, n_init=10)
clusters = kmeans.fit_predict(X_kmeans)

# ---------- Count Observations per Cluster ----------
cluster_counts = pd.Series(clusters).value_counts().sort_index()
print(f"\nCluster counts:")
for cluster_id, count in cluster_counts.items():
    print(f"  Cluster {cluster_id}: {count} observations")

largest_cluster = cluster_counts.idxmax()
print(f"\nLargest cluster identifier: {largest_cluster}")

# ---------- Generate Scatter Plot (Distance vs Price) ----------
plt.figure(figsize=(10, 6))
scatter = plt.scatter(df_kmeans['distance_numeric'], df_kmeans['price_numeric'],
                     c=clusters, cmap='viridis', alpha=0.6, s=50, edgecolors='black', linewidth=0.5)
plt.xlabel('Distance (Kilometers)', fontsize=12)
plt.ylabel('Price (Dollars)', fontsize=12)
plt.title('K-Means Clustering - Luxury Properties', fontsize=14, fontweight='bold')
plt.colorbar(scatter, label='Cluster')
plt.grid(True, alpha=0.3)
plt.tight_layout()
plt.savefig('scatter_distance_price.png', dpi=300, bbox_inches='tight')
plt.close()
print("Generated: scatter_distance_price.png")

# ==========================================
# ANALYSIS 3: Polynomial Regression (airnb_desert.xlsx)
# ==========================================
print("\n" + "=" * 50)
print("ANALYSIS 3: Polynomial Regression")
print("=" * 50)

# ---------- Extract Numeric Price (Desert) ----------
df_desert['price_numeric'] = df_desert['Price(In dollar)'].apply(extract_price)

# ---------- Select Complete Records ----------
df_poly = df_desert[['price_numeric', 'Rating']].dropna()
print(f"Complete records for polynomial regression: {len(df_poly)}")

# ---------- Create Polynomial Features (Degree 2) ----------
poly = PolynomialFeatures(degree=2, include_bias=False)
price_poly = poly.fit_transform(df_poly[['price_numeric']])

# ---------- Standardize Polynomial Features ----------
scaler_poly = StandardScaler()
X_poly = scaler_poly.fit_transform(price_poly)
y_poly = df_poly['Rating'].values

# ---------- Fit Polynomial Regression ----------
poly_model = LinearRegression()
poly_model.fit(X_poly, y_poly)
y_pred_poly = poly_model.predict(X_poly)

# ---------- Calculate R-squared ----------
r2_poly = r2_score(y_poly, y_pred_poly)
print(f"\nR-squared for polynomial regression: {r2_poly:.4f}")

# ---------- Generate Scatter Plot (Price vs Rating) ----------
plt.figure(figsize=(10, 6))
plt.scatter(df_poly['price_numeric'], df_poly['Rating'], alpha=0.5, s=30, label='Actual')

# Sort for smooth line
sort_idx = np.argsort(df_poly['price_numeric'].values)
plt.plot(df_poly['price_numeric'].values[sort_idx], y_pred_poly[sort_idx],
         color='red', linewidth=2, label='Polynomial Fit')

plt.xlabel('Price (Dollars)', fontsize=12)
plt.ylabel('Rating Score', fontsize=12)
plt.title('Polynomial Regression - Desert Properties', fontsize=14, fontweight='bold')
plt.legend()
plt.grid(True, alpha=0.3)
plt.tight_layout()
plt.savefig('scatter_price_rating.png', dpi=300, bbox_inches='tight')
plt.close()
print("Generated: scatter_price_rating.png")

# ==========================================
# ANALYSIS 4: Principal Component Analysis (airnb.xlsx)
# ==========================================
print("\n" + "=" * 50)
print("ANALYSIS 4: Principal Component Analysis")
print("=" * 50)

# ---------- Select Complete Records (Already extracted) ----------
df_pca = df_airnb[['price_numeric', 'rating_numeric', 'bed_count']].dropna()
print(f"Complete records for PCA: {len(df_pca)}")

# ---------- Standardize Features ----------
scaler_pca = StandardScaler()
X_pca = scaler_pca.fit_transform(df_pca)

# ---------- Fit PCA ----------
pca = PCA()
pca_transformed = pca.fit_transform(X_pca)

# ---------- Calculate Variance Explained by PC1 ----------
variance_pc1 = pca.explained_variance_ratio_[0] * 100
print(f"\nVariance explained by PC1: {variance_pc1:.2f}%")

# ---------- Generate Biplot ----------
fig, ax = plt.subplots(figsize=(10, 8))

# Plot samples
sample_indices = np.random.choice(len(pca_transformed), size=min(500, len(pca_transformed)), replace=False)
ax.scatter(pca_transformed[sample_indices, 0], pca_transformed[sample_indices, 1],
          alpha=0.3, s=20, color='steelblue')

# Plot feature vectors
feature_names = ['Price', 'Rating', 'Bed Count']
for i, feature in enumerate(feature_names):
    ax.arrow(0, 0, pca.components_[0, i]*3, pca.components_[1, i]*3,
            head_width=0.15, head_length=0.15, fc='red', ec='red', linewidth=2)
    ax.text(pca.components_[0, i]*3.5, pca.components_[1, i]*3.5, feature,
           fontsize=12, fontweight='bold', color='darkred')

ax.set_xlabel(f'PC1 ({pca.explained_variance_ratio_[0]*100:.1f}% variance)', fontsize=12)
ax.set_ylabel(f'PC2 ({pca.explained_variance_ratio_[1]*100:.1f}% variance)', fontsize=12)
ax.set_title('PCA Biplot - Regular Listings', fontsize=14, fontweight='bold')
ax.axhline(y=0, color='k', linewidth=0.5, linestyle='--', alpha=0.3)
ax.axvline(x=0, color='k', linewidth=0.5, linestyle='--', alpha=0.3)
ax.grid(True, alpha=0.3)
plt.tight_layout()
plt.savefig('biplot_pca.png', dpi=300, bbox_inches='tight')
plt.close()
print("Generated: biplot_pca.png")

# ==========================================
# ANALYSIS 5: Hierarchical Clustering (airnb_luxe.xlsx)
# ==========================================
print("\n" + "=" * 50)
print("ANALYSIS 5: Hierarchical Clustering")
print("=" * 50)

# ---------- Use Already Prepared Data ----------
# df_kmeans already has complete price and distance data
print(f"Complete records for hierarchical clustering: {len(df_kmeans)}")

# ---------- Compute Hierarchical Clustering (Ward Linkage) ----------
linkage_matrix = linkage(X_kmeans, method='ward')

# ---------- Compute Silhouette Scores for k=2 to k=5 ----------
silhouette_scores = {}
for k in range(2, 6):
    # Cut dendrogram to get k clusters
    from scipy.cluster.hierarchy import fcluster
    clusters_hc = fcluster(linkage_matrix, k, criterion='maxclust')
    score = silhouette_score(X_kmeans, clusters_hc)
    silhouette_scores[k] = score
    print(f"k={k}: Silhouette Score = {score:.4f}")

# ---------- Identify Optimal Cluster Count ----------
optimal_k = max(silhouette_scores, key=silhouette_scores.get)
print(f"\nOptimal number of clusters: {optimal_k}")

# ---------- Generate Dendrogram ----------
plt.figure(figsize=(12, 6))
dendrogram(linkage_matrix, no_labels=True, color_threshold=0)
plt.xlabel('Sample Index', fontsize=12)
plt.ylabel('Linkage Distance', fontsize=12)
plt.title('Hierarchical Clustering Dendrogram - Luxury Properties', fontsize=14, fontweight='bold')
plt.grid(True, alpha=0.3, axis='y')
plt.tight_layout()
plt.savefig('dendrogram.png', dpi=300, bbox_inches='tight')
plt.close()
print("Generated: dendrogram.png")

# ==========================================
# ANALYSIS 6: Weighted Average Price (airnb.xlsx)
# ==========================================
print("\n" + "=" * 50)
print("ANALYSIS 6: Weighted Average Price")
print("=" * 50)

# ---------- Extract Review Count ----------
def extract_review_count(rating_str):
    """Parse integer from parentheses containing review count"""
    try:
        if pd.isna(rating_str):
            return np.nan
        # Find number in parentheses
        match = re.search(r'\((\d+)\)', str(rating_str))
        if match:
            return int(match.group(1))
        return np.nan
    except:
        return np.nan

df_airnb['review_count'] = df_airnb['Review and rating'].apply(extract_review_count)

# ---------- Select Complete Records ----------
df_weighted = df_airnb[['price_numeric', 'review_count']].dropna()
print(f"Complete records for weighted average: {len(df_weighted)}")

# ---------- Calculate Weighted Average Price ----------
weighted_avg_price = (df_weighted['price_numeric'] * df_weighted['review_count']).sum() / df_weighted['review_count'].sum()
print(f"\nWeighted average price: ${weighted_avg_price:.2f}")

# ==========================================
# ANALYSIS 7: Interquartile Range (airnb_desert.xlsx)
# ==========================================
print("\n" + "=" * 50)
print("ANALYSIS 7: Interquartile Range")
print("=" * 50)

# ---------- Select Non-Missing Ratings ----------
ratings_desert = df_desert['Rating'].dropna()
print(f"Complete ratings: {len(ratings_desert)}")

# ---------- Calculate IQR ----------
Q1 = ratings_desert.quantile(0.25)
Q3 = ratings_desert.quantile(0.75)
IQR = Q3 - Q1
print(f"\nQ1 (25th percentile): {Q1:.4f}")
print(f"Q3 (75th percentile): {Q3:.4f}")
print(f"Interquartile Range (IQR): {IQR:.4f}")

# ==========================================
# ANALYSIS 8: Coefficient of Variation (airnb_luxe.xlsx)
# ==========================================
print("\n" + "=" * 50)
print("ANALYSIS 8: Coefficient of Variation")
print("=" * 50)

# ---------- Select Complete Distance Values ----------
distance_values = df_luxe['distance_numeric'].dropna()
print(f"Complete distance values: {len(distance_values)}")

# ---------- Calculate CV ----------
mean_distance = distance_values.mean()
std_distance = distance_values.std()
cv_distance = (std_distance / mean_distance) * 100

print(f"\nMean distance: {mean_distance:.2f} km")
print(f"Standard deviation: {std_distance:.2f} km")
print(f"Coefficient of Variation: {cv_distance:.2f}%")

# ==========================================
# ANALYSIS 9: Price Comparison (airnb.xlsx + airnb_luxe.xlsx)
# ==========================================
print("\n" + "=" * 50)
print("ANALYSIS 9: Price Comparison")
print("=" * 50)

# ---------- Calculate Mean Prices ----------
mean_regular_price = df_airnb['price_numeric'].dropna().mean()
mean_luxury_price = df_luxe['price_numeric'].dropna().mean()

print(f"Mean regular listing price: ${mean_regular_price:.2f}")
print(f"Mean luxury listing price: ${mean_luxury_price:.2f}")

# ---------- Compute Price Ratio ----------
price_ratio = mean_luxury_price / mean_regular_price
print(f"\nPrice ratio (luxury/regular): {price_ratio:.2f}")

# ==========================================
# ANALYSIS 10: Spearman Correlation (airnb_desert.xlsx)
# ==========================================
print("\n" + "=" * 50)
print("ANALYSIS 10: Spearman Correlation")
print("=" * 50)

# ---------- Select Complete Records ----------
df_spearman = df_desert[['price_numeric', 'Rating']].dropna()
print(f"Complete records for correlation: {len(df_spearman)}")

# ---------- Calculate Spearman Correlation ----------
spearman_corr, p_value = spearmanr(df_spearman['price_numeric'], df_spearman['Rating'])
print(f"\nSpearman correlation coefficient: {spearman_corr:.4f}")
print(f"P-value: {p_value:.4f}")

# ==========================================
# KEY OUTPUTS SUMMARY
# ==========================================
print("\n" + "=" * 60)
print("KEY OUTPUTS SUMMARY")
print("=" * 60)
print(f"1. Multiple Linear Regression - Bed Count Coefficient: {bed_coefficient:.4f}")
print(f"2. K-Means Clustering - Largest Cluster ID: {largest_cluster}")
print(f"3. Polynomial Regression - R-squared: {r2_poly:.4f}")
print(f"4. PCA - Variance Explained by PC1: {variance_pc1:.2f}%")
print(f"5. Hierarchical Clustering - Optimal K: {optimal_k}")
print(f"6. Weighted Average Price: ${weighted_avg_price:.2f}")
print(f"7. IQR for Desert Ratings: {IQR:.4f}")
print(f"8. Coefficient of Variation for Distance: {cv_distance:.2f}%")
print(f"9. Price Ratio (Luxury/Regular): {price_ratio:.2f}")
print(f"10. Spearman Correlation (Price-Rating): {spearman_corr:.4f}")
print("=" * 60)

print("\n✓ Analysis complete! Generated 5 visualization files.")
