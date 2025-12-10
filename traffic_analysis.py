# ==========================================
# Integrated Traffic Volume & Weather Analysis Script
# ==========================================
# Requirements: pandas, numpy, matplotlib, scipy, scikit-learn, openpyxl, statsmodels
# Input files: traffic_volume.xlsx, climate_data.xlsx, weather_data.xlsx (in same directory)
# Output files: Various PNG plots for each analysis
# Key outputs: Printed statistical results and metrics
# ==========================================

import pandas as pd
import numpy as np
import matplotlib.pyplot as plt
from sklearn.linear_model import LinearRegression, LogisticRegression
from sklearn.cluster import KMeans, AgglomerativeClustering
from sklearn.decomposition import PCA
from sklearn.preprocessing import StandardScaler
from sklearn.ensemble import RandomForestRegressor
from sklearn.model_selection import train_test_split
from sklearn.metrics import mean_absolute_error, accuracy_score, confusion_matrix
from scipy.stats import pearsonr, kstest, norm
from statsmodels.tsa.arima.model import ARIMA
from statsmodels.tsa.stattools import grangercausalitytests
from scipy.cluster.hierarchy import dendrogram, linkage
import warnings
warnings.filterwarnings('ignore')

# ---------- Load Excel Files ----------
print("Loading data files...")
traffic_df = pd.read_excel('traffic_volume.xlsx')
climate_df = pd.read_excel('climate_data.xlsx')
weather_df = pd.read_excel('weather_data.xlsx')

print(f"Traffic data shape: {traffic_df.shape}")
print(f"Climate data shape: {climate_df.shape}")
print(f"Weather data shape: {weather_df.shape}")

# ==========================================
# ANALYSIS 1: Multiple Linear Regression
# Traffic Volume vs Climate Conditions
# ==========================================
print("\n" + "="*50)
print("ANALYSIS 1: Multiple Linear Regression")
print("="*50)

# Extract date from timestamp and aggregate daily traffic
traffic_df['date'] = pd.to_datetime(traffic_df['timestamp']).dt.date
daily_traffic = traffic_df.groupby('date')['traffic_volume'].sum().reset_index()
daily_traffic.columns = ['date', 'daily_total_traffic']

# Merge with climate data
climate_df['date'] = pd.to_datetime(climate_df['date']).dt.date
merged_climate = pd.merge(daily_traffic, climate_df, on='date', how='inner')

# Prepare features and target
X_climate = merged_climate[['meantemp', 'humidity', 'wind_speed', 'meanpressure']]
y_climate = merged_climate['daily_total_traffic']

# Fit multiple linear regression
lr_model = LinearRegression()
lr_model.fit(X_climate, y_climate)
r_squared = lr_model.score(X_climate, y_climate)

print(f"\nR-squared value: {r_squared:.6f}")

# Generate scatter plot with regression line
plt.figure(figsize=(10, 6))
plt.scatter(merged_climate['meantemp'], merged_climate['daily_total_traffic'], alpha=0.5)
# Create prediction line for meantemp (holding other variables at mean)
temp_range = np.linspace(merged_climate['meantemp'].min(), merged_climate['meantemp'].max(), 100)
mean_humidity = merged_climate['humidity'].mean()
mean_wind = merged_climate['wind_speed'].mean()
mean_pressure = merged_climate['meanpressure'].mean()
X_pred = pd.DataFrame({
    'meantemp': temp_range,
    'humidity': mean_humidity,
    'wind_speed': mean_wind,
    'meanpressure': mean_pressure
})
y_pred = lr_model.predict(X_pred)
plt.plot(temp_range, y_pred, 'r-', linewidth=2, label='Regression Line')
plt.xlabel('Mean Temperature (degrees)')
plt.ylabel('Daily Total Traffic Volume')
plt.title('Traffic Volume vs Temperature with Regression Line')
plt.legend()
plt.tight_layout()
plt.savefig('regression_traffic_temperature.png', dpi=300)
plt.close()
print("Plot saved: regression_traffic_temperature.png")

# ==========================================
# ANALYSIS 2: K-Means Clustering
# Traffic Pattern Segmentation
# ==========================================
print("\n" + "="*50)
print("ANALYSIS 2: K-Means Clustering")
print("="*50)

# Extract hour and normalize traffic volume
traffic_df['hour'] = pd.to_datetime(traffic_df['timestamp']).dt.hour
traffic_df['normalized_volume'] = traffic_df['traffic_volume'] / traffic_df['traffic_volume'].max()

# Prepare features for clustering
X_kmeans = traffic_df[['hour', 'normalized_volume']].values

# Apply k-means with 5 clusters
kmeans = KMeans(n_clusters=5, random_state=42, n_init=10)
traffic_df['cluster'] = kmeans.fit_predict(X_kmeans)

# Identify cluster with highest average traffic
cluster_avg = traffic_df.groupby('cluster')['traffic_volume'].mean()
highest_cluster = cluster_avg.idxmax()

print(f"\nCluster with highest average traffic volume: {highest_cluster}")
print(f"Average volumes by cluster:\n{cluster_avg}")

# Generate scatter plot
plt.figure(figsize=(10, 6))
colors = ['red', 'blue', 'green', 'orange', 'purple']
for i in range(5):
    cluster_data = traffic_df[traffic_df['cluster'] == i]
    plt.scatter(cluster_data['hour'], cluster_data['normalized_volume'],
                c=colors[i], label=f'Cluster {i}', alpha=0.5, s=10)
plt.xlabel('Hour of Day (0-23)')
plt.ylabel('Normalized Traffic Volume (0-1)')
plt.title('K-Means Clustering of Traffic Patterns')
plt.legend()
plt.grid(True, alpha=0.3)
plt.tight_layout()
plt.savefig('kmeans_traffic_clustering.png', dpi=300)
plt.close()
print("Plot saved: kmeans_traffic_clustering.png")

# ==========================================
# ANALYSIS 3: Principal Component Analysis
# Weather Data Dimensionality Reduction
# ==========================================
print("\n" + "="*50)
print("ANALYSIS 3: Principal Component Analysis")
print("="*50)

# Select features and remove missing values
pca_features = ['air_pollution_index', 'humidity', 'wind_speed', 'visibility_in_miles',
                'dew_point', 'temperature', 'clouds_all']
weather_pca = weather_df[pca_features].dropna()

# Standardize features
scaler_pca = StandardScaler()
X_standardized = scaler_pca.fit_transform(weather_pca)

# Apply PCA
pca = PCA()
pca_scores = pca.fit_transform(X_standardized)

# Calculate cumulative variance explained
cumulative_variance = np.cumsum(pca.explained_variance_ratio_)
variance_3pc = cumulative_variance[2]

print(f"\nCumulative variance explained by first 3 PCs: {variance_3pc:.6f}")
print(f"Individual variance explained: {pca.explained_variance_ratio_[:3]}")

# Generate biplot
plt.figure(figsize=(12, 8))
plt.scatter(pca_scores[:, 0], pca_scores[:, 1], alpha=0.3, s=10)

# Add loading vectors
loadings = pca.components_[:2, :].T * np.sqrt(pca.explained_variance_[:2])
for i, feature in enumerate(pca_features):
    plt.arrow(0, 0, loadings[i, 0]*3, loadings[i, 1]*3,
              head_width=0.3, head_length=0.3, fc='red', ec='red')
    plt.text(loadings[i, 0]*3.2, loadings[i, 1]*3.2, feature,
             fontsize=9, ha='center', va='center')

plt.xlabel(f'PC1 ({pca.explained_variance_ratio_[0]:.2%} variance)')
plt.ylabel(f'PC2 ({pca.explained_variance_ratio_[1]:.2%} variance)')
plt.title('PCA Biplot - Weather Data')
plt.grid(True, alpha=0.3)
plt.tight_layout()
plt.savefig('pca_weather_biplot.png', dpi=300)
plt.close()
print("Plot saved: pca_weather_biplot.png")

# ==========================================
# ANALYSIS 4: ARIMA Time Series Forecasting
# Future Traffic Volume Prediction
# ==========================================
print("\n" + "="*50)
print("ANALYSIS 4: ARIMA Forecasting")
print("="*50)

# Aggregate daily traffic
traffic_df['date'] = pd.to_datetime(traffic_df['timestamp']).dt.date
daily_ts = traffic_df.groupby('date')['traffic_volume'].sum().reset_index()
daily_ts = daily_ts.sort_values('date')
daily_ts.set_index('date', inplace=True)

# Fit ARIMA(2,1,2)
arima_model = ARIMA(daily_ts['traffic_volume'], order=(2, 1, 2))
arima_fit = arima_model.fit()

# Forecast 7 days ahead
forecast = arima_fit.forecast(steps=7)
day_7_forecast = forecast.iloc[-1]

print(f"\nForecasted traffic volume for day 7: {day_7_forecast:.2f}")

# Generate plot with forecast
plt.figure(figsize=(14, 6))
plt.plot(daily_ts.index, daily_ts['traffic_volume'], label='Historical', linewidth=1)

# Create future dates
last_date = daily_ts.index[-1]
future_dates = pd.date_range(start=pd.Timestamp(last_date) + pd.Timedelta(days=1), periods=7, freq='D')
plt.plot(future_dates, forecast.values, 'r--', label='7-Day Forecast', linewidth=2, marker='o')

plt.xlabel('Date')
plt.ylabel('Daily Total Traffic Volume')
plt.title('ARIMA(2,1,2) Traffic Volume Forecast')
plt.legend()
plt.xticks(rotation=45)
plt.grid(True, alpha=0.3)
plt.tight_layout()
plt.savefig('arima_traffic_forecast.png', dpi=300)
plt.close()
print("Plot saved: arima_traffic_forecast.png")

# ==========================================
# ANALYSIS 5: Granger Causality Test
# Air Pollution -> Traffic Volume
# ==========================================
print("\n" + "="*50)
print("ANALYSIS 5: Granger Causality Test")
print("="*50)

# Extract date and aggregate traffic (May-Dec 2017)
traffic_df['date'] = pd.to_datetime(traffic_df['timestamp']).dt.date
traffic_daily = traffic_df.groupby('date')['traffic_volume'].sum().reset_index()
traffic_daily.columns = ['date', 'daily_traffic']

# Extract date and aggregate air pollution
weather_df['date'] = pd.to_datetime(weather_df['date_time']).dt.date
pollution_daily = weather_df.groupby('date')['air_pollution_index'].mean().reset_index()
pollution_daily.columns = ['date', 'daily_pollution']

# Merge both series
granger_data = pd.merge(pollution_daily, traffic_daily, on='date', how='inner')
granger_data = granger_data.sort_values('date')

# Prepare time series for Granger test
granger_ts = granger_data[['daily_pollution', 'daily_traffic']].values

# Perform Granger causality test with lag 4
granger_result = grangercausalitytests(granger_ts, maxlag=4, verbose=False)
p_value_granger = granger_result[4][0]['ssr_ftest'][1]

print(f"\nGranger causality test p-value (lag 4): {p_value_granger:.6f}")

# Generate dual time series plot
fig, ax1 = plt.subplots(figsize=(14, 6))

# Normalize both series to 0-1
pollution_norm = (granger_data['daily_pollution'] - granger_data['daily_pollution'].min()) / \
                 (granger_data['daily_pollution'].max() - granger_data['daily_pollution'].min())
traffic_norm = (granger_data['daily_traffic'] - granger_data['daily_traffic'].min()) / \
               (granger_data['daily_traffic'].max() - granger_data['daily_traffic'].min())

ax1.plot(granger_data['date'], pollution_norm, 'b-', label='Air Pollution Index (normalized)', linewidth=1)
ax1.plot(granger_data['date'], traffic_norm, 'r-', label='Traffic Volume (normalized)', linewidth=1)
ax1.set_xlabel('Date (May - December 2017)')
ax1.set_ylabel('Normalized Value (0-1)')
ax1.set_title('Air Pollution and Traffic Volume Time Series')
ax1.legend()
ax1.grid(True, alpha=0.3)
plt.xticks(rotation=45)
plt.tight_layout()
plt.savefig('granger_pollution_traffic.png', dpi=300)
plt.close()
print("Plot saved: granger_pollution_traffic.png")

# ==========================================
# ANALYSIS 6: Hierarchical Clustering
# Climate Profile Grouping
# ==========================================
print("\n" + "="*50)
print("ANALYSIS 6: Hierarchical Clustering")
print("="*50)

# Select features and standardize
climate_features = ['meantemp', 'humidity', 'wind_speed', 'meanpressure']
X_climate_hc = climate_df[climate_features].dropna()

scaler_hc = StandardScaler()
X_climate_std = scaler_hc.fit_transform(X_climate_hc)

# Perform hierarchical clustering
hc_model = AgglomerativeClustering(n_clusters=6, linkage='ward')
climate_clusters = hc_model.fit_predict(X_climate_std)

# Add cluster assignments
climate_clean = climate_df[climate_features].dropna().copy()
climate_clean['cluster'] = climate_clusters

# Identify cluster with highest average meanpressure
cluster_pressure = climate_clean.groupby('cluster')['meanpressure'].mean()
highest_pressure_cluster = cluster_pressure.idxmax()

print(f"\nCluster with highest average mean pressure: {highest_pressure_cluster}")
print(f"Average mean pressure by cluster:\n{cluster_pressure}")

# Generate dendrogram
plt.figure(figsize=(14, 6))
linkage_matrix = linkage(X_climate_std, method='ward', metric='euclidean')
dendrogram(linkage_matrix, truncate_mode='lastp', p=30)
plt.xlabel('Sample Index (or Cluster Size)')
plt.ylabel('Euclidean Distance')
plt.title('Hierarchical Clustering Dendrogram (6 Clusters)')
plt.axhline(y=linkage_matrix[-5, 2], color='r', linestyle='--', label='6 Cluster Cut')
plt.legend()
plt.tight_layout()
plt.savefig('hierarchical_climate_dendrogram.png', dpi=300)
plt.close()
print("Plot saved: hierarchical_climate_dendrogram.png")

# ==========================================
# ANALYSIS 7: Random Forest Regression
# Traffic Volume Prediction
# ==========================================
print("\n" + "="*50)
print("ANALYSIS 7: Random Forest Regression")
print("="*50)

# Engineer features
traffic_df['hour'] = pd.to_datetime(traffic_df['timestamp']).dt.hour
traffic_df['day_of_week'] = pd.to_datetime(traffic_df['timestamp']).dt.dayofweek
traffic_df['events_binary'] = traffic_df['events'].apply(lambda x: 1 if x == True else 0)

# Prepare features and target
X_rf = traffic_df[['hour', 'day_of_week', 'events_binary']]
y_rf = traffic_df['traffic_volume']

# Split dataset
X_train, X_test, y_train, y_test = train_test_split(X_rf, y_rf, test_size=0.2, random_state=42)

# Train Random Forest
rf_model = RandomForestRegressor(n_estimators=100, random_state=42)
rf_model.fit(X_train, y_train)

# Evaluate on test set
y_pred_rf = rf_model.predict(X_test)
mae_rf = mean_absolute_error(y_test, y_pred_rf)

print(f"\nMean Absolute Error on test set: {mae_rf:.2f}")

# Generate feature importance plot
feature_names = ['Hour of Day', 'Day of Week', 'Events Indicator']
importances = rf_model.feature_importances_

plt.figure(figsize=(10, 6))
plt.bar(feature_names, importances, color=['steelblue', 'coral', 'mediumseagreen'])
plt.xlabel('Features')
plt.ylabel('Importance Score')
plt.title('Random Forest Feature Importance')
plt.tight_layout()
plt.savefig('random_forest_importance.png', dpi=300)
plt.close()
print("Plot saved: random_forest_importance.png")

# ==========================================
# ANALYSIS 8: Logistic Regression
# Holiday Prediction from Weather
# ==========================================
print("\n" + "="*50)
print("ANALYSIS 8: Logistic Regression Classification")
print("="*50)

# Create binary target
weather_df['holiday_binary'] = weather_df['is_holiday'].apply(lambda x: 0 if pd.isna(x) else 1)

# Select features and remove missing values
log_features = ['temperature', 'humidity', 'wind_speed', 'air_pollution_index', 'clouds_all']
weather_log = weather_df[log_features + ['holiday_binary']].dropna()

X_log = weather_log[log_features]
y_log = weather_log['holiday_binary']

# Split dataset with stratification
X_train_log, X_test_log, y_train_log, y_test_log = train_test_split(
    X_log, y_log, test_size=0.25, random_state=42, stratify=y_log)

# Train Logistic Regression
log_model = LogisticRegression(random_state=42, max_iter=1000)
log_model.fit(X_train_log, y_train_log)

# Evaluate on test set
y_pred_log = log_model.predict(X_test_log)
accuracy_log = accuracy_score(y_test_log, y_pred_log)

print(f"\nClassification accuracy on test set: {accuracy_log:.6f}")

# Generate confusion matrix
cm = confusion_matrix(y_test_log, y_pred_log)

plt.figure(figsize=(8, 6))
plt.imshow(cm, interpolation='nearest', cmap='Blues')
plt.title('Confusion Matrix - Holiday Prediction')
plt.colorbar()
tick_marks = [0, 1]
plt.xticks(tick_marks, ['Non-Holiday (0)', 'Holiday (1)'])
plt.yticks(tick_marks, ['Non-Holiday (0)', 'Holiday (1)'])

# Add text annotations
for i in range(2):
    for j in range(2):
        plt.text(j, i, str(cm[i, j]), ha='center', va='center',
                color='white' if cm[i, j] > cm.max()/2 else 'black', fontsize=16)

plt.ylabel('Actual Class')
plt.xlabel('Predicted Class')
plt.tight_layout()
plt.savefig('logistic_confusion_matrix.png', dpi=300)
plt.close()
print("Plot saved: logistic_confusion_matrix.png")

# ==========================================
# ANALYSIS 9: Pearson Correlation
# Visibility vs Traffic Volume
# ==========================================
print("\n" + "="*50)
print("ANALYSIS 9: Pearson Correlation Analysis")
print("="*50)

# Aggregate traffic by date (May-Dec)
traffic_df['date'] = pd.to_datetime(traffic_df['timestamp']).dt.date
traffic_mean = traffic_df.groupby('date')['traffic_volume'].mean().reset_index()
traffic_mean.columns = ['date', 'mean_traffic']

# Aggregate visibility by date
weather_df['date'] = pd.to_datetime(weather_df['date_time']).dt.date
visibility_mean = weather_df.groupby('date')['visibility_in_miles'].mean().reset_index()
visibility_mean.columns = ['date', 'mean_visibility']

# Merge both series
corr_data = pd.merge(traffic_mean, visibility_mean, on='date', how='inner')

# Calculate Pearson correlation
pearson_corr, pearson_p = pearsonr(corr_data['mean_visibility'], corr_data['mean_traffic'])

print(f"\nPearson correlation coefficient: {pearson_corr:.6f}")
print(f"P-value: {pearson_p:.6f}")

# Generate scatter plot
plt.figure(figsize=(10, 6))
plt.scatter(corr_data['mean_visibility'], corr_data['mean_traffic'], alpha=0.5)
plt.xlabel('Daily Mean Visibility (miles)')
plt.ylabel('Daily Mean Traffic Volume')
plt.title(f'Traffic Volume vs Visibility\nPearson r = {pearson_corr:.4f}')
plt.grid(True, alpha=0.3)
plt.tight_layout()
plt.savefig('pearson_visibility_traffic.png', dpi=300)
plt.close()
print("Plot saved: pearson_visibility_traffic.png")

# ==========================================
# ANALYSIS 10: Kolmogorov-Smirnov Test
# Humidity Normality Assessment
# ==========================================
print("\n" + "="*50)
print("ANALYSIS 10: Kolmogorov-Smirnov Test")
print("="*50)

# Extract humidity values
humidity_values = climate_df['humidity'].dropna().values

# Estimate parameters from sample
humidity_mean = np.mean(humidity_values)
humidity_std = np.std(humidity_values, ddof=1)

# Perform KS test
ks_statistic, ks_pvalue = kstest(humidity_values, lambda x: norm.cdf(x, humidity_mean, humidity_std))

print(f"\nKolmogorov-Smirnov test p-value: {ks_pvalue:.6f}")
print(f"KS statistic: {ks_statistic:.6f}")

# ==========================================
# ANALYSIS 11: Peak Traffic Hour
# Resource Allocation Planning
# ==========================================
print("\n" + "="*50)
print("ANALYSIS 11: Peak Traffic Hour Identification")
print("="*50)

# Extract hour and sum traffic across all dates
traffic_df['hour'] = pd.to_datetime(traffic_df['timestamp']).dt.hour
hourly_totals = traffic_df.groupby('hour')['traffic_volume'].sum()
peak_hour = hourly_totals.idxmax()

print(f"\nHour with maximum total traffic volume: {peak_hour}")
print(f"Total traffic at peak hour: {hourly_totals[peak_hour]:.0f}")

print("\n" + "="*50)
print("ALL ANALYSES COMPLETED")
print("="*50)
