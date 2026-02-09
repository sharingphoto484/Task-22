# ==========================================
# Integrated Tesla Stock Market Analysis Script
# Requirements: pandas, numpy, matplotlib, scipy, scikit-learn, statsmodels
# Input files: tsla_2025.xlsx, TSLA-2.xlsx, HistoricalData_1726367135218.xlsx
# Output files: rf_scatter.png, gb_lineplot.png, dbscan_boxplot.png,
#               exp_smoothing_lineplot.png, daily_returns_histogram.png
# ==========================================

import pandas as pd
import numpy as np
import matplotlib
matplotlib.use('Agg')
import matplotlib.pyplot as plt
from sklearn.ensemble import (RandomForestRegressor, GradientBoostingRegressor,
                              ExtraTreesClassifier, IsolationForest)
from sklearn.neighbors import KNeighborsRegressor
from sklearn.cluster import DBSCAN
from sklearn.preprocessing import StandardScaler
from sklearn.metrics import (mean_absolute_error, r2_score,
                             matthews_corrcoef, mean_squared_error)
from statsmodels.tsa.api import VAR
from statsmodels.tsa.holtwinters import ExponentialSmoothing
import warnings
warnings.filterwarnings('ignore')

# ========================================================================
# 1. RANDOM FOREST REGRESSION — tsla_2025.xlsx (Next-Day Close Prediction)
# ========================================================================

# ---------- Load Data ----------
df_rf = pd.read_excel('tsla_2025.xlsx')

# ---------- Remove Observations Where Close/Open/High/Low/Volume Is Missing ----------
df_rf = df_rf.dropna(subset=['Close', 'Open', 'High', 'Low', 'Volume'])

# ---------- Sort by Date Chronologically Ascending ----------
df_rf['Date'] = pd.to_datetime(df_rf['Date'])
df_rf = df_rf.sort_values('Date').reset_index(drop=True)

# ---------- Create Lagged Features ----------
df_rf['lag_close_1'] = df_rf['Close'].shift(1)
df_rf['lag_volume_1'] = df_rf['Volume'].shift(1)
df_rf['lag_high_1'] = df_rf['High'].shift(1)

# ---------- Remove Observations With Missing Lagged Values ----------
df_rf = df_rf.dropna(subset=['lag_close_1', 'lag_volume_1', 'lag_high_1']).reset_index(drop=True)

# ---------- Train/Test Split (75/25, Temporal Order, No Shuffle) ----------
split_rf = int(len(df_rf) * 0.75)
train_rf = df_rf.iloc[:split_rf]
test_rf = df_rf.iloc[split_rf:]

# ---------- Fit Random Forest Regressor ----------
features_rf = ['lag_close_1', 'lag_volume_1', 'lag_high_1', 'Open', 'High', 'Low']
X_train_rf = train_rf[features_rf]
y_train_rf = train_rf['Close']
X_test_rf = test_rf[features_rf]
y_test_rf = test_rf['Close']

rf_model = RandomForestRegressor(
    n_estimators=100, max_depth=10, min_samples_split=5, random_state=42
)
rf_model.fit(X_train_rf, y_train_rf)
rf_pred = rf_model.predict(X_test_rf)

# ---------- Report Testing Set MAE ----------
rf_mae = mean_absolute_error(y_test_rf, rf_pred)
print(f"1. Random Forest Regression — Testing Set MAE: {rf_mae:.4f}")

# ========================================================================
# 2. GRADIENT BOOSTING REGRESSION — TSLA-2.xlsx (Adj Close Forecast)
# ========================================================================

# ---------- Load Data ----------
df_gb = pd.read_excel('TSLA-2.xlsx')

# ---------- Remove Observations Where Adj Close/Open/Volume Is Missing ----------
df_gb = df_gb.dropna(subset=['Adj Close', 'Open', 'Volume'])

# ---------- Sort by Date Chronologically Ascending ----------
df_gb['Date'] = pd.to_datetime(df_gb['Date'])
df_gb = df_gb.sort_values('Date').reset_index(drop=True)

# ---------- Train/Test Split (80/20, Temporal Order, No Shuffle) ----------
split_gb = int(len(df_gb) * 0.80)
train_gb = df_gb.iloc[:split_gb]
test_gb = df_gb.iloc[split_gb:]

# ---------- Fit Gradient Boosting Regressor ----------
features_gb = ['Open', 'Volume']
X_train_gb = train_gb[features_gb]
y_train_gb = train_gb['Adj Close']
X_test_gb = test_gb[features_gb]
y_test_gb = test_gb['Adj Close']

gb_model = GradientBoostingRegressor(
    learning_rate=0.1, n_estimators=150, max_depth=4,
    subsample=0.8, random_state=42
)
gb_model.fit(X_train_gb, y_train_gb)
gb_pred = gb_model.predict(X_test_gb)

# ---------- Report Testing Set R-squared ----------
gb_r2 = r2_score(y_test_gb, gb_pred)
print(f"2. Gradient Boosting Regression — Testing Set R-squared: {gb_r2:.4f}")

# ========================================================================
# 3. EXTRA TREES CLASSIFICATION — HistoricalData (Price Direction)
# ========================================================================

# ---------- Load Data ----------
df_et = pd.read_excel('HistoricalData_1726367135218.xlsx')

# ---------- Remove Observations Where Close/Last or Open Is Missing ----------
df_et = df_et.dropna(subset=['Close/Last', 'Open'])

# ---------- Convert Close/Last and Open From String to Numeric (Remove $) ----------
df_et['Close/Last'] = df_et['Close/Last'].astype(str).str.replace('$', '', regex=False).astype(float)
df_et['Open'] = df_et['Open'].astype(str).str.replace('$', '', regex=False).astype(float)

# ---------- Create Binary Target Direction ----------
df_et['direction'] = (df_et['Close/Last'] > df_et['Open']).astype(int)

# ---------- Sort by Date Chronologically Ascending ----------
df_et['Date'] = pd.to_datetime(df_et['Date'])
df_et = df_et.sort_values('Date').reset_index(drop=True)

# ---------- Train/Test Split (65/35, Temporal Order, No Shuffle) ----------
split_et = int(len(df_et) * 0.65)
train_et = df_et.iloc[:split_et]
test_et = df_et.iloc[split_et:]

# ---------- Fit Extra Trees Classifier ----------
X_train_et = train_et[['Open']]
y_train_et = train_et['direction']
X_test_et = test_et[['Open']]
y_test_et = test_et['direction']

et_model = ExtraTreesClassifier(
    n_estimators=200, max_depth=8, min_samples_leaf=3, random_state=42
)
et_model.fit(X_train_et, y_train_et)
et_pred = et_model.predict(X_test_et)

# ---------- Report Testing Set Matthews Correlation Coefficient ----------
et_mcc = matthews_corrcoef(y_test_et, et_pred)
print(f"3. Extra Trees Classification — Testing Set MCC: {et_mcc:.4f}")

# ========================================================================
# 4. ISOLATION FOREST ANOMALY DETECTION — tsla_2025.xlsx (Volume)
# ========================================================================

# ---------- Load Data ----------
df_if = pd.read_excel('tsla_2025.xlsx')

# ---------- Remove Observations Where Volume Is Missing ----------
df_if = df_if.dropna(subset=['Volume'])

# ---------- Sort by Date Chronologically Ascending ----------
df_if['Date'] = pd.to_datetime(df_if['Date'])
df_if = df_if.sort_values('Date').reset_index(drop=True)

# ---------- Fit Isolation Forest ----------
iso_model = IsolationForest(
    contamination=0.05, n_estimators=100, random_state=42
)
iso_labels = iso_model.fit_predict(df_if[['Volume']])

# ---------- Count Anomalies (Label -1) ----------
anomaly_count = int(np.sum(iso_labels == -1))
print(f"4. Isolation Forest — Anomaly Count: {anomaly_count}")

# ========================================================================
# 5. DBSCAN DENSITY CLUSTERING — TSLA-2.xlsx (Trading Patterns)
# ========================================================================

# ---------- Load Data ----------
df_db = pd.read_excel('TSLA-2.xlsx')

# ---------- Remove Observations Where High/Low/Volume Is Missing ----------
df_db = df_db.dropna(subset=['High', 'Low', 'Volume'])

# ---------- Sort by Date Chronologically Ascending ----------
df_db['Date'] = pd.to_datetime(df_db['Date'])
df_db = df_db.sort_values('Date').reset_index(drop=True)

# ---------- Calculate Daily Range ----------
df_db['range'] = df_db['High'] - df_db['Low']

# ---------- Standardize Range and Volume ----------
scaler = StandardScaler()
df_db[['std_range', 'std_volume']] = scaler.fit_transform(df_db[['range', 'Volume']])

# ---------- Fit DBSCAN ----------
dbscan_model = DBSCAN(eps=0.5, min_samples=10)
df_db['cluster'] = dbscan_model.fit_predict(df_db[['std_range', 'std_volume']])

# ---------- Count Unique Clusters Excluding Noise (-1) ----------
cluster_labels = df_db['cluster'].unique()
cluster_count = int(len([c for c in cluster_labels if c != -1]))
print(f"5. DBSCAN Clustering — Cluster Count: {cluster_count}")

# ========================================================================
# 6. VECTOR AUTOREGRESSION — tsla_2025.xlsx (Multivariate Time Series)
# ========================================================================

# ---------- Load Data ----------
df_var = pd.read_excel('tsla_2025.xlsx')

# ---------- Remove Observations Where Close or Volume Is Missing ----------
df_var = df_var.dropna(subset=['Close', 'Volume'])

# ---------- Sort by Date Chronologically Ascending ----------
df_var['Date'] = pd.to_datetime(df_var['Date'])
df_var = df_var.sort_values('Date').reset_index(drop=True)

# ---------- Extract Most Recent 500 Observations ----------
df_var_subset = df_var.tail(500).reset_index(drop=True)

# ---------- Fit VAR Model With Lag Order 2 ----------
var_data = df_var_subset[['Close', 'Volume']].astype(float)
var_model = VAR(var_data)
var_result = var_model.fit(maxlags=2, ic=None)

# ---------- Extract Coefficient for Close at Lag 1 Predicting Close ----------
# The coefficient matrix for lag 1: var_result.coefs[0]
# Row 0 = Close equation, Column 0 = Close lag 1
var_coef = var_result.coefs[0][0, 0]
print(f"6. VAR Model — Close Lag-1 Coefficient (predicting Close): {var_coef:.4f}")

# ========================================================================
# 7. TRIPLE EXPONENTIAL SMOOTHING — HistoricalData (Holt-Winters)
# ========================================================================

# ---------- Load Data ----------
df_hw = pd.read_excel('HistoricalData_1726367135218.xlsx')

# ---------- Remove Observations Where Close/Last Is Missing ----------
df_hw = df_hw.dropna(subset=['Close/Last'])

# ---------- Convert Close/Last From String to Numeric (Remove $) ----------
df_hw['Close/Last'] = df_hw['Close/Last'].astype(str).str.replace('$', '', regex=False).astype(float)

# ---------- Sort by Date Chronologically Ascending ----------
df_hw['Date'] = pd.to_datetime(df_hw['Date'])
df_hw = df_hw.sort_values('Date').reset_index(drop=True)

# ---------- Fit Holt-Winters Exponential Smoothing ----------
hw_model = ExponentialSmoothing(
    df_hw['Close/Last'],
    trend='add',
    seasonal='add',
    seasonal_periods=7
)
hw_result = hw_model.fit()

# ---------- Generate 7-Step Ahead Forecast ----------
hw_forecast = hw_result.forecast(steps=7)

# ---------- Report Mean of 7 Forecasted Values ----------
hw_mean = hw_forecast.mean()
print(f"7. Holt-Winters Exponential Smoothing — Mean of 7-Step Forecast: {hw_mean:.4f}")

# ========================================================================
# 8. KNN REGRESSION — TSLA-2.xlsx (Close Price Prediction)
# ========================================================================

# ---------- Load Data ----------
df_knn = pd.read_excel('TSLA-2.xlsx')

# ---------- Remove Observations Where Close/Open/High Is Missing ----------
df_knn = df_knn.dropna(subset=['Close', 'Open', 'High'])

# ---------- Sort by Date Chronologically Ascending ----------
df_knn['Date'] = pd.to_datetime(df_knn['Date'])
df_knn = df_knn.sort_values('Date').reset_index(drop=True)

# ---------- Train/Test Split (70/30, Temporal Order, No Shuffle) ----------
split_knn = int(len(df_knn) * 0.70)
train_knn = df_knn.iloc[:split_knn]
test_knn = df_knn.iloc[split_knn:]

# ---------- Fit KNN Regressor ----------
features_knn = ['Open', 'High']
X_train_knn = train_knn[features_knn]
y_train_knn = train_knn['Close']
X_test_knn = test_knn[features_knn]
y_test_knn = test_knn['Close']

knn_model = KNeighborsRegressor(n_neighbors=5, weights='uniform')
knn_model.fit(X_train_knn, y_train_knn)
knn_pred = knn_model.predict(X_test_knn)

# ---------- Report Testing Set MSE ----------
knn_mse = mean_squared_error(y_test_knn, knn_pred)
print(f"8. KNN Regression — Testing Set MSE: {knn_mse:.4f}")

# ========================================================================
# 9. SHARPE RATIO — tsla_2025.xlsx (Risk-Adjusted Returns)
# ========================================================================

# ---------- Load Data ----------
df_sr = pd.read_excel('tsla_2025.xlsx')

# ---------- Remove Observations Where Close Is Missing ----------
df_sr = df_sr.dropna(subset=['Close'])

# ---------- Sort by Date Chronologically Ascending ----------
df_sr['Date'] = pd.to_datetime(df_sr['Date'])
df_sr = df_sr.sort_values('Date').reset_index(drop=True)

# ---------- Calculate Daily Returns (Percentage Change) ----------
df_sr['daily_return'] = df_sr['Close'].pct_change()

# ---------- Remove First Observation With Undefined Return ----------
df_sr = df_sr.dropna(subset=['daily_return']).reset_index(drop=True)

# ---------- Compute Mean and Std of Daily Returns ----------
mean_return = df_sr['daily_return'].mean()
std_return = df_sr['daily_return'].std()

# ---------- Calculate Annualized Sharpe Ratio ----------
sharpe_ratio = (mean_return / std_return) * np.sqrt(252)
print(f"9. Sharpe Ratio (Annualized): {sharpe_ratio:.4f}")

# ========================================================================
# 10. OPTIMAL BUY-HOLD PERIOD — tsla_2025.xlsx (Rolling Returns)
# ========================================================================

# ---------- Load Data ----------
df_bh = pd.read_excel('tsla_2025.xlsx')

# ---------- Sort by Date and Extract Close ----------
df_bh['Date'] = pd.to_datetime(df_bh['Date'])
df_bh = df_bh.sort_values('Date').reset_index(drop=True)

# ---------- Calculate Rolling Returns ----------
df_bh['roll_30'] = df_bh['Close'].pct_change(periods=30)
df_bh['roll_90'] = df_bh['Close'].pct_change(periods=90)
df_bh['roll_180'] = df_bh['Close'].pct_change(periods=180)

# ---------- Compute Mean Return for Each Timeframe ----------
mean_30 = df_bh['roll_30'].dropna().mean()
mean_90 = df_bh['roll_90'].dropna().mean()
mean_180 = df_bh['roll_180'].dropna().mean()

# ---------- Identify Timeframe With Maximum Mean Return ----------
timeframes = {30: mean_30, 90: mean_90, 180: mean_180}
optimal_period = max(timeframes, key=timeframes.get)
print(f"10. Optimal Buy-Hold Period: {optimal_period} days (30d mean={mean_30:.4f}, 90d mean={mean_90:.4f}, 180d mean={mean_180:.4f})")

# ---------- Open-Ended Answer ----------
# The optimal holding period reflects that Tesla's stock historically rewards longer-term investors more than short-term traders, as the compounding effect of sustained growth trends outweighs short-term volatility.

# ========================================================================
# VISUALIZATIONS
# ========================================================================

# ---------- Plot 1: Random Forest Scatter — Actual vs Predicted Close ----------
fig1, ax1 = plt.subplots(figsize=(8, 6))
ax1.scatter(y_test_rf, rf_pred, alpha=0.5, edgecolors='k', linewidths=0.5, s=30)
min_val = min(y_test_rf.min(), rf_pred.min())
max_val = max(y_test_rf.max(), rf_pred.max())
ax1.plot([min_val, max_val], [min_val, max_val], 'r--', linewidth=1.5, label='Perfect Prediction')
ax1.set_xlabel('Actual Close')
ax1.set_ylabel('Predicted Close')
ax1.set_title('Random Forest: Actual vs Predicted Close')
ax1.legend()
plt.tight_layout()
fig1.savefig('rf_scatter.png', dpi=150)
plt.close(fig1)
print("   Plot saved: rf_scatter.png")

# ---------- Plot 2: Gradient Boosting Line — Actual vs Predicted Adj Close ----------
fig2, ax2 = plt.subplots(figsize=(10, 6))
date_index_gb = test_gb['Date']
ax2.plot(date_index_gb, y_test_gb.values, label='Actual Adj Close', linewidth=1.2)
ax2.plot(date_index_gb, gb_pred, label='Predicted Adj Close', linewidth=1.2, linestyle='--')
ax2.set_xlabel('Date')
ax2.set_ylabel('Adj Close')
ax2.set_title('Gradient Boosting: Actual vs Predicted Adj Close')
ax2.legend()
plt.tight_layout()
fig2.savefig('gb_lineplot.png', dpi=150)
plt.close(fig2)
print("   Plot saved: gb_lineplot.png")

# ---------- Plot 3: DBSCAN Box Plot — Standardized Volume by Cluster ----------
fig3, ax3 = plt.subplots(figsize=(8, 6))
df_db_no_noise = df_db[df_db['cluster'] != -1]
clusters_unique = sorted(df_db_no_noise['cluster'].unique())
box_data = [df_db_no_noise[df_db_no_noise['cluster'] == c]['std_volume'].values for c in clusters_unique]
ax3.boxplot(box_data, tick_labels=[str(c) for c in clusters_unique])
ax3.set_xlabel('Cluster Label')
ax3.set_ylabel('Standardized Volume')
ax3.set_title('DBSCAN: Standardized Volume by Cluster')
plt.tight_layout()
fig3.savefig('dbscan_boxplot.png', dpi=150)
plt.close(fig3)
print("   Plot saved: dbscan_boxplot.png")

# ---------- Plot 4: Exponential Smoothing Line — Historical + Forecast ----------
fig4, ax4 = plt.subplots(figsize=(10, 6))
time_idx_hist = range(len(df_hw))
time_idx_fcst = range(len(df_hw), len(df_hw) + 7)
ax4.plot(time_idx_hist, df_hw['Close/Last'].values, label='Historical Close/Last', linewidth=1.0)
ax4.plot(time_idx_fcst, hw_forecast.values, label='7-Step Forecast', linewidth=2.0,
         linestyle='--', color='red', marker='o')
ax4.set_xlabel('Time Index')
ax4.set_ylabel('Close/Last')
ax4.set_title('Holt-Winters: Historical Data and 7-Step Forecast')
ax4.legend()
plt.tight_layout()
fig4.savefig('exp_smoothing_lineplot.png', dpi=150)
plt.close(fig4)
print("   Plot saved: exp_smoothing_lineplot.png")

# ---------- Plot 5: Daily Returns Histogram ----------
fig5, ax5 = plt.subplots(figsize=(8, 6))
ax5.hist(df_sr['daily_return'] * 100, bins=80, edgecolor='black', alpha=0.75)
ax5.set_xlabel('Daily Return (%)')
ax5.set_ylabel('Frequency')
ax5.set_title('Distribution of Daily Returns (tsla_2025.xlsx)')
ax5.axvline(x=mean_return * 100, color='red', linestyle='--', label=f'Mean: {mean_return*100:.3f}%')
ax5.legend()
plt.tight_layout()
fig5.savefig('daily_returns_histogram.png', dpi=150)
plt.close(fig5)
print("   Plot saved: daily_returns_histogram.png")

print("\n=== Analysis Complete ===")
