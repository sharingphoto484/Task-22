# ==========================================
# Integrated Music Streaming Analytics Script
# ==========================================
# Requirements: pandas, numpy, matplotlib, scipy, scikit-learn, openpyxl
# Input files: Customer.xlsx, Track.xlsx, Invoice.xlsx (in same directory)
# Output files: ridge_regression_scatter.png, hierarchical_dendrogram.png,
#               pca_biplot.png, gradient_boosting_importance.png,
#               dbscan_clusters.png
# ==========================================

import pandas as pd
import numpy as np
import matplotlib.pyplot as plt
from sklearn.linear_model import Ridge
from sklearn.model_selection import train_test_split
from sklearn.metrics import mean_absolute_error, r2_score
from sklearn.preprocessing import StandardScaler
from sklearn.cluster import AgglomerativeClustering, DBSCAN
from sklearn.decomposition import PCA
from sklearn.ensemble import GradientBoostingRegressor
from scipy.cluster.hierarchy import dendrogram, linkage
from scipy.stats import mannwhitneyu
import warnings
warnings.filterwarnings('ignore')

# ---------- Load Excel Files ----------
print("Loading data files...")
customer_df = pd.read_excel('Customer.xlsx')
track_df = pd.read_excel('Track.xlsx')
invoice_df = pd.read_excel('Invoice.xlsx')

print(f"Customer records: {len(customer_df)}")
print(f"Track records: {len(track_df)}")
print(f"Invoice records: {len(invoice_df)}")
print()

# ==========================================
# TASK 1: Ridge Regression on Track Pricing
# ==========================================
print("=" * 60)
print("TASK 1: Ridge Regression for Track Pricing Prediction")
print("=" * 60)

# ---------- Filter Standard-Priced Tracks ----------
standard_tracks = track_df[track_df['UnitPrice'] == 0.99].copy()
print(f"Standard-priced tracks (0.99): {len(standard_tracks)}")

# ---------- Construct Predictor Features ----------
X_ridge = pd.DataFrame()
X_ridge['Milliseconds'] = standard_tracks['Milliseconds']
X_ridge['Bytes'] = standard_tracks['Bytes']
X_ridge['GenreId_1'] = (standard_tracks['GenreId'] == 1).astype(int)
X_ridge['GenreId_2'] = (standard_tracks['GenreId'] == 2).astype(int)
X_ridge['MediaTypeId_1'] = (standard_tracks['MediaTypeId'] == 1).astype(int)
y_ridge = standard_tracks['UnitPrice']

# ---------- Train-Test Split ----------
X_train, X_test, y_train, y_test = train_test_split(
    X_ridge, y_ridge, test_size=0.20, random_state=42
)

# ---------- Train Ridge Regression ----------
ridge_model = Ridge(alpha=1.0)
ridge_model.fit(X_train, y_train)

# ---------- Evaluate on Test Set ----------
y_pred = ridge_model.predict(X_test)
mae_ridge = mean_absolute_error(y_test, y_pred)

print(f"Ridge Regression Mean Absolute Error (Test Set): {mae_ridge:.6f}")
print()

# ---------- Generate Scatter Plot ----------
plt.figure(figsize=(8, 6))
plt.scatter(y_test, y_pred, alpha=0.6, edgecolors='k', s=50)
plt.plot([y_test.min(), y_test.max()], [y_test.min(), y_test.max()],
         'r--', lw=2, label='Perfect Prediction')
plt.xlabel('Actual UnitPrice', fontsize=12)
plt.ylabel('Predicted UnitPrice', fontsize=12)
plt.title('Ridge Regression: Actual vs Predicted UnitPrice', fontsize=14)
plt.legend()
plt.grid(alpha=0.3)
plt.tight_layout()
plt.savefig('ridge_regression_scatter.png', dpi=300)
print("Saved: ridge_regression_scatter.png")
print()

# ==========================================
# TASK 2: Agglomerative Hierarchical Clustering
# ==========================================
print("=" * 60)
print("TASK 2: Agglomerative Hierarchical Clustering")
print("=" * 60)

# ---------- Aggregate Invoice Data by Customer ----------
invoice_agg = invoice_df.groupby('CustomerId').agg(
    invoice_count=('InvoiceId', 'count'),
    total_revenue=('Total', 'sum'),
    avg_transaction=('Total', 'mean')
).reset_index()

# ---------- Merge with Customer Data ----------
customer_invoice = customer_df.merge(invoice_agg, on='CustomerId', how='inner')
print(f"Customers with invoice data: {len(customer_invoice)}")

# ---------- Select and Standardize Features ----------
clustering_features = customer_invoice[['invoice_count', 'total_revenue', 'avg_transaction']]
scaler = StandardScaler()
features_scaled = scaler.fit_transform(clustering_features)

# ---------- Apply Agglomerative Clustering ----------
agg_cluster = AgglomerativeClustering(n_clusters=4, linkage='ward')
customer_invoice['Cluster'] = agg_cluster.fit_predict(features_scaled)

# ---------- Identify Cluster with Highest Revenue ----------
cluster_revenue = customer_invoice.groupby('Cluster')['total_revenue'].mean()
highest_revenue_cluster = cluster_revenue.idxmax()

print(f"Cluster number with highest average total revenue: {highest_revenue_cluster}")
print(f"Average revenue in cluster {highest_revenue_cluster}: ${cluster_revenue[highest_revenue_cluster]:.2f}")
print()

# Answer to open-ended question:
# Hierarchical clustering reveals that high-value customers form a distinct segment (cluster with highest revenue) characterized by elevated purchase frequency and transaction values, indicating targeted marketing strategies should focus on nurturing similar behavioral patterns to optimize revenue streams.

# ---------- Generate Dendrogram ----------
plt.figure(figsize=(12, 6))
linkage_matrix = linkage(features_scaled, method='ward')
dendrogram(linkage_matrix)
plt.axhline(y=linkage_matrix[-3, 2], color='r', linestyle='--', label='4 Clusters')
plt.xlabel('Customer Observations', fontsize=12)
plt.ylabel('Euclidean Distance', fontsize=12)
plt.title('Hierarchical Clustering Dendrogram', fontsize=14)
plt.legend()
plt.tight_layout()
plt.savefig('hierarchical_dendrogram.png', dpi=300)
print("Saved: hierarchical_dendrogram.png")
print()

# ==========================================
# TASK 3: Principal Component Analysis
# ==========================================
print("=" * 60)
print("TASK 3: Principal Component Analysis on Track Features")
print("=" * 60)

# ---------- Extract Numeric Features ----------
pca_features = track_df[['Milliseconds', 'Bytes', 'AlbumId', 'GenreId']].copy()
pca_features = pca_features.dropna()

# ---------- Standardize Features ----------
scaler_pca = StandardScaler()
features_pca_scaled = scaler_pca.fit_transform(pca_features)

# ---------- Apply PCA ----------
pca = PCA()
pca_transformed = pca.fit_transform(features_pca_scaled)

# ---------- Compute Cumulative Variance ----------
cumulative_variance = np.cumsum(pca.explained_variance_ratio_)
cumulative_variance_first_two = cumulative_variance[1]

print(f"Variance explained by PC1: {pca.explained_variance_ratio_[0]:.4f}")
print(f"Variance explained by PC2: {pca.explained_variance_ratio_[1]:.4f}")
print(f"Cumulative variance explained by first two components: {cumulative_variance_first_two:.4f}")
print()

# ---------- Generate Biplot ----------
plt.figure(figsize=(10, 8))
plt.scatter(pca_transformed[:, 0], pca_transformed[:, 1], alpha=0.3, s=10, c='blue')

# Plot loading vectors
feature_names = ['Milliseconds', 'Bytes', 'AlbumId', 'GenreId']
loadings = pca.components_.T * np.sqrt(pca.explained_variance_)
scale_factor = 3
for i, feature in enumerate(feature_names):
    plt.arrow(0, 0, loadings[i, 0] * scale_factor, loadings[i, 1] * scale_factor,
              head_width=0.3, head_length=0.3, fc='red', ec='red', lw=2)
    plt.text(loadings[i, 0] * scale_factor * 1.15, loadings[i, 1] * scale_factor * 1.15,
             feature, fontsize=11, weight='bold', color='darkred')

plt.xlabel(f'PC1 ({pca.explained_variance_ratio_[0]:.2%} variance)', fontsize=12)
plt.ylabel(f'PC2 ({pca.explained_variance_ratio_[1]:.2%} variance)', fontsize=12)
plt.title('PCA Biplot: Track Features', fontsize=14)
plt.grid(alpha=0.3)
plt.axhline(y=0, color='k', linestyle='-', linewidth=0.5)
plt.axvline(x=0, color='k', linestyle='-', linewidth=0.5)
plt.tight_layout()
plt.savefig('pca_biplot.png', dpi=300)
print("Saved: pca_biplot.png")
print()

# ==========================================
# TASK 4: Gradient Boosting Regression
# ==========================================
print("=" * 60)
print("TASK 4: Gradient Boosting Regression for Invoice Prediction")
print("=" * 60)

# ---------- Extract Temporal and Geographic Features ----------
invoice_df['InvoiceDate'] = pd.to_datetime(invoice_df['InvoiceDate'])
invoice_df['Year'] = invoice_df['InvoiceDate'].dt.year
invoice_df['Month'] = invoice_df['InvoiceDate'].dt.month
invoice_df['USA_Indicator'] = (invoice_df['BillingCountry'] == 'USA').astype(int)

# ---------- Prepare Features and Target ----------
X_gb = invoice_df[['Year', 'Month', 'USA_Indicator']]
y_gb = invoice_df['Total']

# ---------- Train-Test Split with Stratification ----------
X_train_gb, X_test_gb, y_train_gb, y_test_gb = train_test_split(
    X_gb, y_gb, test_size=0.25, random_state=33, stratify=invoice_df['Year']
)

# ---------- Train Gradient Boosting Model ----------
gb_model = GradientBoostingRegressor(
    n_estimators=100,
    learning_rate=0.1,
    max_depth=3,
    random_state=33
)
gb_model.fit(X_train_gb, y_train_gb)

# ---------- Evaluate on Test Set ----------
y_pred_gb = gb_model.predict(X_test_gb)
r2_gb = r2_score(y_test_gb, y_pred_gb)

print(f"Gradient Boosting R-squared (Test Set): {r2_gb:.6f}")
print()

# ---------- Generate Feature Importance Chart ----------
feature_importance = gb_model.feature_importances_
features_gb = ['Year', 'Month', 'USA Indicator']

plt.figure(figsize=(8, 6))
plt.barh(features_gb, feature_importance, color='steelblue', edgecolor='black')
plt.xlabel('Importance Score', fontsize=12)
plt.ylabel('Feature Names', fontsize=12)
plt.title('Gradient Boosting: Feature Importance', fontsize=14)
plt.tight_layout()
plt.savefig('gradient_boosting_importance.png', dpi=300)
print("Saved: gradient_boosting_importance.png")
print()

# ==========================================
# TASK 5: DBSCAN Clustering
# ==========================================
print("=" * 60)
print("TASK 5: DBSCAN Density-Based Clustering")
print("=" * 60)

# ---------- Merge and Aggregate ----------
invoice_customer = invoice_df.merge(customer_df, on='CustomerId', how='inner')
customer_spending = invoice_customer.groupby('CustomerId').agg(
    total_spending=('Total', 'sum'),
    Country=('Country', 'first')
).reset_index()

# ---------- Encode Country ----------
def encode_country(country):
    if country == 'USA':
        return 1
    elif country == 'Canada':
        return 2
    else:
        return 3

customer_spending['Country_Encoded'] = customer_spending['Country'].apply(encode_country)

# ---------- Standardize Features ----------
dbscan_features = customer_spending[['total_spending', 'Country_Encoded']]
scaler_dbscan = StandardScaler()
features_dbscan_scaled = scaler_dbscan.fit_transform(dbscan_features)

# ---------- Apply DBSCAN ----------
dbscan = DBSCAN(eps=0.5, min_samples=3)
customer_spending['DBSCAN_Cluster'] = dbscan.fit_predict(features_dbscan_scaled)

# ---------- Count Customers in Cluster 0 ----------
cluster_0_count = (customer_spending['DBSCAN_Cluster'] == 0).sum()

print(f"Number of customers in cluster 0: {cluster_0_count}")
print(f"Cluster distribution:")
print(customer_spending['DBSCAN_Cluster'].value_counts().sort_index())
print()

# ---------- Generate DBSCAN Scatter Plot ----------
plt.figure(figsize=(10, 7))
colors = ['red', 'blue', 'green', 'orange', 'purple']
unique_clusters = sorted(customer_spending['DBSCAN_Cluster'].unique())

for cluster in unique_clusters:
    cluster_data = customer_spending[customer_spending['DBSCAN_Cluster'] == cluster]
    scaled_data = scaler_dbscan.transform(cluster_data[['total_spending', 'Country_Encoded']])
    if cluster == -1:
        plt.scatter(scaled_data[:, 1], scaled_data[:, 0],
                   label=f'Noise (Cluster -1)', marker='x', s=50, c='black', alpha=0.5)
    else:
        color = colors[cluster % len(colors)]
        plt.scatter(scaled_data[:, 1], scaled_data[:, 0],
                   label=f'Cluster {cluster}', s=50, c=color, alpha=0.7, edgecolors='k')

plt.xlabel('Standardized Country Encoding', fontsize=12)
plt.ylabel('Standardized Total Spending', fontsize=12)
plt.title('DBSCAN Clustering: Customer Geographic and Spending Profiles', fontsize=14)
plt.legend()
plt.grid(alpha=0.3)
plt.tight_layout()
plt.savefig('dbscan_clusters.png', dpi=300)
print("Saved: dbscan_clusters.png")
print()

# ==========================================
# TASK 6: Maximum Revenue Billing Country
# ==========================================
print("=" * 60)
print("TASK 6: Billing Country with Maximum Total Revenue")
print("=" * 60)

# ---------- Aggregate Revenue by Country ----------
country_revenue = invoice_df.groupby('BillingCountry')['Total'].sum()
max_revenue_country = country_revenue.idxmax()
max_revenue_amount = country_revenue.max()

print(f"Billing country with maximum total revenue: {max_revenue_country}")
print(f"Total revenue: ${max_revenue_amount:.2f}")
print()

# ==========================================
# TASK 7: GenreId with Most Premium Tracks
# ==========================================
print("=" * 60)
print("TASK 7: GenreId with Most Premium-Priced Tracks")
print("=" * 60)

# ---------- Filter Premium Tracks ----------
premium_tracks = track_df[track_df['UnitPrice'] == 1.99]

# ---------- Count by GenreId ----------
genre_counts = premium_tracks['GenreId'].value_counts()
most_premium_genre = genre_counts.idxmax()
premium_count = genre_counts.max()

print(f"GenreId with most premium tracks (1.99): {most_premium_genre}")
print(f"Number of premium tracks: {premium_count}")
print()

# ==========================================
# TASK 8: Mann-Whitney U Test
# ==========================================
print("=" * 60)
print("TASK 8: Mann-Whitney U Test (USA vs Non-USA)")
print("=" * 60)

# ---------- Merge Invoice with Customer ----------
invoice_country = invoice_df.merge(customer_df[['CustomerId', 'Country']],
                                   on='CustomerId', how='inner')

# ---------- Create Two Samples ----------
usa_totals = invoice_country[invoice_country['Country'] == 'USA']['Total']
non_usa_totals = invoice_country[invoice_country['Country'] != 'USA']['Total']

# ---------- Apply Mann-Whitney U Test ----------
u_statistic, p_value = mannwhitneyu(usa_totals, non_usa_totals, alternative='two-sided')

print(f"Mann-Whitney U Test Results:")
print(f"USA sample size: {len(usa_totals)}")
print(f"Non-USA sample size: {len(non_usa_totals)}")
print(f"U-statistic: {u_statistic:.2f}")
print(f"P-value: {p_value:.6f}")
print()

# ==========================================
# SUMMARY OF KEY OUTPUTS
# ==========================================
print("=" * 60)
print("KEY OUTPUTS SUMMARY")
print("=" * 60)
print(f"1. Ridge Regression MAE: {mae_ridge:.6f}")
print(f"2. Cluster with highest revenue: {highest_revenue_cluster}")
print(f"3. PCA cumulative variance (first 2 PCs): {cumulative_variance_first_two:.4f}")
print(f"4. Gradient Boosting R-squared: {r2_gb:.6f}")
print(f"5. DBSCAN Cluster 0 count: {cluster_0_count}")
print(f"6. Max revenue country: {max_revenue_country}")
print(f"7. Most premium GenreId: {most_premium_genre}")
print(f"8. Mann-Whitney U p-value: {p_value:.6f}")
print()
print("=" * 60)
print("ANALYSIS COMPLETE")
print("=" * 60)
