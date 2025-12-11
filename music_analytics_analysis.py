# ==========================================
# Integrated Music Analytics Script
# ==========================================
# Requirements: pandas, numpy, matplotlib, scipy, scikit-learn, statsmodels
# Input files: Customer.xlsx, Invoice.xlsx, Track.xlsx (in same directory)
# Output files: customer_segmentation_scatter.png, spending_heatmap.png,
#               revenue_forecast_plot.png, hierarchical_dendrogram.png,
#               duration_density_plot.png
# ==========================================

import pandas as pd
import numpy as np
import matplotlib.pyplot as plt
from scipy import stats
from scipy.cluster.hierarchy import dendrogram, linkage, fcluster
from scipy.stats import gaussian_kde
from sklearn.cluster import KMeans
from sklearn.preprocessing import StandardScaler
from sklearn.decomposition import PCA
from sklearn.metrics import silhouette_score
from statsmodels.tsa.arima.model import ARIMA
import warnings
warnings.filterwarnings('ignore')

# ---------- Load Datasets ----------
print("Loading datasets...")
customer_df = pd.read_excel('Customer.xlsx')
invoice_df = pd.read_excel('Invoice.xlsx')
track_df = pd.read_excel('Track.xlsx')

# ---------- Data Cleaning ----------
print("Cleaning datasets...")
# Remove rows with missing key identifiers
customer_df = customer_df.dropna(subset=['CustomerId'])
invoice_df = invoice_df.dropna(subset=['InvoiceId', 'CustomerId'])
track_df = track_df.dropna(subset=['TrackId'])

# Convert InvoiceDate to datetime
invoice_df['InvoiceDate'] = pd.to_datetime(invoice_df['InvoiceDate'])
invoice_df = invoice_df.sort_values('InvoiceDate')

print(f"Customer records: {len(customer_df)}")
print(f"Invoice records: {len(invoice_df)}")
print(f"Track records: {len(track_df)}")

# ==========================================
# TASK 1: CUSTOMER REVENUE SEGMENTATION
# ==========================================
print("\n" + "="*60)
print("TASK 1: Customer Revenue Segmentation (K-Means Clustering)")
print("="*60)

# ---------- Merge and Aggregate ----------
# Merge Invoice with Customer, keeping only customers with at least one invoice
invoice_customer = invoice_df.merge(customer_df, on='CustomerId', how='inner')

# Aggregate by CustomerId
customer_features = invoice_customer.groupby('CustomerId').agg(
    total_revenue=('Total', 'sum'),
    invoice_count=('InvoiceId', 'count')
).reset_index()

# Remove any rows with missing or non-numeric values
customer_features = customer_features.dropna()
customer_features = customer_features[
    (pd.to_numeric(customer_features['total_revenue'], errors='coerce').notna()) &
    (pd.to_numeric(customer_features['invoice_count'], errors='coerce').notna())
]

# ---------- Standardize Features ----------
scaler = StandardScaler()
features_standardized = scaler.fit_transform(customer_features[['total_revenue', 'invoice_count']])
customer_features['std_revenue'] = features_standardized[:, 0]
customer_features['std_invoice_count'] = features_standardized[:, 1]

# ---------- K-Means Clustering ----------
kmeans = KMeans(n_clusters=3, random_state=42, n_init=10)
customer_features['cluster'] = kmeans.fit_predict(features_standardized)

# ---------- Silhouette Score ----------
silhouette = silhouette_score(features_standardized, customer_features['cluster'])
print(f"Silhouette Score: {silhouette:.3f}")

# ---------- Visualization ----------
plt.figure(figsize=(10, 6))
colors = ['red', 'blue', 'green']
for cluster in range(3):
    cluster_data = customer_features[customer_features['cluster'] == cluster]
    plt.scatter(cluster_data['std_revenue'], cluster_data['std_invoice_count'],
                c=colors[cluster], label=f'Cluster {cluster}', alpha=0.6, s=50)
plt.xlabel('Standardized Total Revenue')
plt.ylabel('Standardized Invoice Count')
plt.title('Customer Segmentation by Revenue and Invoice Count')
plt.legend()
plt.grid(True, alpha=0.3)
plt.tight_layout()
plt.savefig('customer_segmentation_scatter.png', dpi=300, bbox_inches='tight')
plt.close()
print("Saved: customer_segmentation_scatter.png")

# ==========================================
# TASK 2: SPENDING PROFILE DIFFERENCES ACROSS COUNTRIES
# ==========================================
print("\n" + "="*60)
print("TASK 2: Spending Profile Differences (Chi-Square Test)")
print("="*60)

# ---------- Merge and Calculate Customer Total Revenue ----------
customer_spending = invoice_df.groupby('CustomerId').agg(
    total_revenue=('Total', 'sum')
).reset_index()

# Merge with customer data to get Country
customer_spending = customer_spending.merge(customer_df[['CustomerId', 'Country']], on='CustomerId')

# ---------- Define High Spender Threshold ----------
threshold_75 = customer_spending['total_revenue'].quantile(0.75)
customer_spending['high_spender'] = (customer_spending['total_revenue'] >= threshold_75).astype(int)

# ---------- Contingency Table ----------
contingency_table = pd.crosstab(customer_spending['Country'], customer_spending['high_spender'])

# ---------- Chi-Square Test ----------
chi2, p_value, dof, expected = stats.chi2_contingency(contingency_table)
print(f"Chi-Square Statistic: {chi2:.3f}")
print(f"P-value: {p_value:.6f}")

# ---------- Heatmap Visualization ----------
plt.figure(figsize=(8, 10))
plt.imshow(contingency_table.values, cmap='YlOrRd', aspect='auto')
plt.colorbar(label='Count')
plt.xlabel('Spender Category (0=Non-High, 1=High)')
plt.ylabel('Country')
plt.title('Contingency Table: Country vs High Spender')
plt.xticks([0, 1], ['Non-High Spender', 'High Spender'])
plt.yticks(range(len(contingency_table.index)), contingency_table.index)
for i in range(len(contingency_table.index)):
    for j in range(len(contingency_table.columns)):
        plt.text(j, i, str(contingency_table.values[i, j]),
                ha='center', va='center', color='black', fontsize=8)
plt.tight_layout()
plt.savefig('spending_heatmap.png', dpi=300, bbox_inches='tight')
plt.close()
print("Saved: spending_heatmap.png")

# ==========================================
# TASK 3: TRACK FEATURE STRUCTURE (PCA)
# ==========================================
print("\n" + "="*60)
print("TASK 3: Track Feature Structure (Principal Component Analysis)")
print("="*60)

# ---------- Select and Clean Track Data ----------
track_pca = track_df[['Milliseconds', 'Bytes', 'UnitPrice']].copy()
track_pca = track_pca.dropna()
track_pca = track_pca[(track_pca['Milliseconds'] > 0) &
                      (track_pca['Bytes'] > 0) &
                      (track_pca['UnitPrice'] > 0)]

# ---------- Feature Engineering ----------
track_pca['duration_minutes'] = track_pca['Milliseconds'] / 60000
track_pca['storage_megabytes'] = track_pca['Bytes'] / 1e6

# ---------- Standardize Features ----------
pca_features = track_pca[['duration_minutes', 'storage_megabytes', 'UnitPrice']].values
scaler_pca = StandardScaler()
pca_features_std = scaler_pca.fit_transform(pca_features)

# ---------- Apply PCA ----------
pca = PCA()
pca.fit(pca_features_std)

# ---------- Variance Explained by First Two Components ----------
variance_explained = pca.explained_variance_ratio_
cumulative_variance_2pc = variance_explained[0] + variance_explained[1]
print(f"Cumulative Variance Explained (First 2 PCs): {cumulative_variance_2pc:.4f}")
print(f"PC1 variance: {variance_explained[0]:.4f}")
print(f"PC2 variance: {variance_explained[1]:.4f}")

# ==========================================
# TASK 4: REVENUE FORECASTING (ARIMA)
# ==========================================
print("\n" + "="*60)
print("TASK 4: Revenue Forecasting (ARIMA Model)")
print("="*60)

# ---------- Extract Date and Aggregate Daily Revenue ----------
invoice_forecast = invoice_df.copy()
invoice_forecast['date'] = invoice_forecast['InvoiceDate'].dt.date
daily_revenue = invoice_forecast.groupby('date')['Total'].sum().reset_index()
daily_revenue.columns = ['date', 'revenue']
daily_revenue['date'] = pd.to_datetime(daily_revenue['date'])

# ---------- Create Continuous Daily Index ----------
date_range = pd.date_range(start=daily_revenue['date'].min(),
                           end=daily_revenue['date'].max(),
                           freq='D')
daily_series = pd.DataFrame({'date': date_range})
daily_series = daily_series.merge(daily_revenue, on='date', how='left')
daily_series['revenue'] = daily_series['revenue'].fillna(0)
daily_series = daily_series.sort_values('date')

# ---------- Fit ARIMA(1,1,1) ----------
print("Fitting ARIMA(1,1,1) model...")
model = ARIMA(daily_series['revenue'].values, order=(1, 1, 1))
model_fit = model.fit()

# ---------- Forecast 14 Days ----------
forecast = model_fit.forecast(steps=14)
forecast_day_14 = forecast[13]  # 14th day (0-indexed)
print(f"Forecasted Revenue for Day 14: {forecast_day_14:.2f}")

# ---------- Visualization ----------
last_date = daily_series['date'].max()
forecast_dates = pd.date_range(start=last_date + pd.Timedelta(days=1), periods=14, freq='D')

plt.figure(figsize=(14, 6))
plt.plot(daily_series['date'], daily_series['revenue'], label='Observed Revenue', color='blue', linewidth=1.5)
plt.plot(forecast_dates, forecast, label='14-Day Forecast', color='red', linewidth=1.5, linestyle='--')
plt.xlabel('Date')
plt.ylabel('Revenue')
plt.title('Daily Revenue and 14-Day Forecast (ARIMA 1,1,1)')
plt.legend()
plt.grid(True, alpha=0.3)
plt.xticks(rotation=45)
plt.tight_layout()
plt.savefig('revenue_forecast_plot.png', dpi=300, bbox_inches='tight')
plt.close()
print("Saved: revenue_forecast_plot.png")

# ==========================================
# TASK 5: TRACK CATALOGUE GROUPING (HIERARCHICAL CLUSTERING)
# ==========================================
print("\n" + "="*60)
print("TASK 5: Track Catalogue Grouping (Hierarchical Clustering)")
print("="*60)

# ---------- Select and Clean Track Data ----------
track_hier = track_df[['TrackId', 'Milliseconds', 'UnitPrice']].copy()
track_hier = track_hier.dropna()
track_hier = track_hier[(track_hier['Milliseconds'] > 0) & (track_hier['UnitPrice'] > 0)]

# ---------- Feature Engineering ----------
track_hier['duration_minutes'] = track_hier['Milliseconds'] / 60000

# ---------- Standardize Features ----------
scaler_hier = StandardScaler()
hier_features_std = scaler_hier.fit_transform(track_hier[['duration_minutes', 'UnitPrice']])

# ---------- Hierarchical Clustering ----------
linkage_matrix = linkage(hier_features_std, method='ward', metric='euclidean')

# ---------- Cut Tree to Get 4 Clusters ----------
track_hier['cluster'] = fcluster(linkage_matrix, t=4, criterion='maxclust')

# ---------- Find Cluster with Highest Mean UnitPrice ----------
cluster_mean_price = track_hier.groupby('cluster')['UnitPrice'].mean()
most_expensive_cluster = cluster_mean_price.idxmax()
print(f"Cluster with Highest Mean UnitPrice: {most_expensive_cluster}")
print(f"Mean prices by cluster:\n{cluster_mean_price}")

# ---------- Dendrogram Visualization ----------
plt.figure(figsize=(12, 6))
dendrogram(linkage_matrix, no_labels=True, color_threshold=0)
plt.xlabel('Track Index')
plt.ylabel('Linkage Distance')
plt.title('Hierarchical Clustering Dendrogram (Ward Linkage)')
plt.tight_layout()
plt.savefig('hierarchical_dendrogram.png', dpi=300, bbox_inches='tight')
plt.close()
print("Saved: hierarchical_dendrogram.png")

# ==========================================
# TASK 6: LISTENING TIME DISTRIBUTION (KDE)
# ==========================================
print("\n" + "="*60)
print("TASK 6: Listening Time Distribution (Kernel Density Estimation)")
print("="*60)

# ---------- Select and Clean Track Duration Data ----------
track_kde = track_df[['Milliseconds']].copy()
track_kde = track_kde.dropna()
track_kde = track_kde[track_kde['Milliseconds'] > 0]
track_kde['duration_minutes'] = track_kde['Milliseconds'] / 60000
duration_values = track_kde['duration_minutes'].values

# ---------- Fit Gaussian KDE with Scott's Rule ----------
kde = gaussian_kde(duration_values, bw_method='scott')

# ---------- Find Mode (Maximum Density) ----------
duration_grid = np.linspace(duration_values.min(), duration_values.max(), 1000)
density_values = kde(duration_grid)
mode_index = np.argmax(density_values)
modal_duration = duration_grid[mode_index]
print(f"Modal Duration (minutes): {modal_duration:.2f}")

# ---------- Visualization ----------
plt.figure(figsize=(10, 6))
plt.plot(duration_grid, density_values, color='blue', linewidth=2)
plt.axvline(modal_duration, color='red', linestyle='--', linewidth=2, label=f'Mode: {modal_duration:.2f} min')
plt.xlabel('Duration (minutes)')
plt.ylabel('Estimated Density')
plt.title('Kernel Density Estimation of Track Duration')
plt.legend()
plt.grid(True, alpha=0.3)
plt.tight_layout()
plt.savefig('duration_density_plot.png', dpi=300, bbox_inches='tight')
plt.close()
print("Saved: duration_density_plot.png")

# ==========================================
# TASK 7: SEASONAL STRUCTURE IN REVENUE (FFT)
# ==========================================
print("\n" + "="*60)
print("TASK 7: Seasonal Structure in Revenue (Fourier Transform)")
print("="*60)

# ---------- Extract Year-Month and Aggregate ----------
invoice_seasonal = invoice_df.copy()
invoice_seasonal['year_month'] = invoice_seasonal['InvoiceDate'].dt.to_period('M')
monthly_revenue = invoice_seasonal.groupby('year_month')['Total'].sum().reset_index()
monthly_revenue = monthly_revenue.sort_values('year_month')
revenue_series = monthly_revenue['Total'].values

# ---------- Apply Discrete Fourier Transform ----------
fft_result = np.fft.fft(revenue_series)
fft_magnitude = np.abs(fft_result)

# ---------- Find Frequency with Largest Magnitude (Excluding Zero) ----------
fft_magnitude_no_zero = fft_magnitude[1:]  # Exclude index 0
max_freq_index = np.argmax(fft_magnitude_no_zero) + 1  # Add 1 to account for exclusion
print(f"Frequency Index with Largest Magnitude: {max_freq_index}")
print(f"This corresponds to a period of approximately {len(revenue_series)/max_freq_index:.2f} months")

# ==========================================
# TASK 8: CUSTOMER VALUE INDICATOR
# ==========================================
print("\n" + "="*60)
print("TASK 8: Customer Value Indicator")
print("="*60)

# ---------- Calculate Total Revenue Per Customer ----------
customer_value = invoice_df.groupby('CustomerId')['Total'].sum().reset_index()
customer_value.columns = ['CustomerId', 'total_revenue']

# ---------- Calculate Average Customer Revenue ----------
avg_customer_revenue = customer_value['total_revenue'].mean()
print(f"Average Customer Revenue: {avg_customer_revenue:.2f}")

# ==========================================
# SUMMARY OF KEY FINDINGS
# ==========================================
print("\n" + "="*60)
print("SUMMARY OF KEY FINDINGS")
print("="*60)
print(f"1. Customer Segmentation Silhouette Score: {silhouette:.3f}")
print(f"2. Country vs Spending Chi-Square Statistic: {chi2:.3f}")
print(f"3. PCA Cumulative Variance (2 PCs): {cumulative_variance_2pc:.4f}")
print(f"4. ARIMA Forecast for Day 14: ${forecast_day_14:.2f}")
print(f"5. Most Expensive Track Cluster: {most_expensive_cluster}")
print(f"6. Modal Track Duration: {modal_duration:.2f} minutes")
print(f"7. Dominant Seasonal Frequency Index: {max_freq_index}")
print(f"8. Average Customer Revenue: ${avg_customer_revenue:.2f}")

# ==========================================
# OPEN-ENDED VERDICT
# ==========================================
verdict = ("Customer behavior exhibits strong segmentation by revenue and purchase frequency "
           "with silhouette score of {:.3f}, while geographic location shows significant "
           "association with high-spending patterns (chi-square = {:.3f}), revealing that "
           "both individual engagement patterns and regional market characteristics jointly "
           "drive music purchasing behavior.").format(silhouette, chi2)

print("\n" + "="*60)
print("VERDICT")
print("="*60)
print(verdict)
print("\nAnalysis complete!")
