# ==========================================
# Equity Performance Quantitative Analysis
# ==========================================
# Requirements: pandas, numpy, matplotlib, scipy, scikit-learn, statsmodels
# Input files: Uber.csv, Zoom.csv, Tesla.csv (in same directory)
# Output files: multi_stock_price_chart.png, rolling_volatility_chart.png,
#               volume_return_scatter.png
# ==========================================

import pandas as pd
import numpy as np
import matplotlib.pyplot as plt
from scipy import stats
from sklearn.linear_model import QuantileRegressor
from statsmodels.tsa.arima.model import ARIMA
import warnings
warnings.filterwarnings('ignore')

# ---------- Load CSVs Robustly ----------
print("Loading datasets...")
uber = pd.read_csv('Uber.csv')
zoom = pd.read_csv('Zoom.csv')
tesla = pd.read_csv('Tesla.csv')

# Convert Date to datetime
uber['Date'] = pd.to_datetime(uber['Date'])
zoom['Date'] = pd.to_datetime(zoom['Date'])
tesla['Date'] = pd.to_datetime(tesla['Date'])

# ---------- Clean and Validate Data ----------
print("Cleaning and validating data...")
required_cols = ['Date', 'Open', 'High', 'Low', 'Close', 'Adj Close', 'Volume']

# Remove rows with missing or non-numeric values
for df in [uber, zoom, tesla]:
    for col in ['Open', 'High', 'Low', 'Close', 'Adj Close', 'Volume']:
        df[col] = pd.to_numeric(df[col], errors='coerce')

uber = uber.dropna(subset=required_cols)
zoom = zoom.dropna(subset=required_cols)
tesla = tesla.dropna(subset=required_cols)

# ---------- Merge on Common Dates ----------
print("Merging datasets on common dates...")
# Find common dates across all three datasets
common_dates = set(uber['Date']) & set(zoom['Date']) & set(tesla['Date'])
common_dates = sorted(list(common_dates))

# Filter each dataset to common dates and sort
uber_merged = uber[uber['Date'].isin(common_dates)].sort_values('Date').reset_index(drop=True)
zoom_merged = zoom[zoom['Date'].isin(common_dates)].sort_values('Date').reset_index(drop=True)
tesla_merged = tesla[tesla['Date'].isin(common_dates)].sort_values('Date').reset_index(drop=True)

print(f"Common trading days: {len(common_dates)}")

# ---------- Calculate Daily Returns ----------
print("Calculating daily log returns...")

def calculate_log_returns(df, stock_name):
    """Calculate log returns and remove zero-volume dates"""
    df = df.copy()
    # Calculate log returns
    df['Return'] = np.log(df['Adj Close']) - np.log(df['Adj Close'].shift(1))
    # Remove first row (NaN return)
    df = df.iloc[1:].reset_index(drop=True)
    # Remove dates where Volume = 0
    df = df[df['Volume'] > 0].reset_index(drop=True)
    return df

uber_returns = calculate_log_returns(uber_merged, 'Uber')
zoom_returns = calculate_log_returns(zoom_merged, 'Zoom')
tesla_returns = calculate_log_returns(tesla_merged, 'Tesla')

print(f"Uber valid trading days: {len(uber_returns)}")
print(f"Zoom valid trading days: {len(zoom_returns)}")
print(f"Tesla valid trading days: {len(tesla_returns)}")

# ---------- ARIMA(1,1,1) Analysis for Tesla ----------
print("\n" + "="*60)
print("ARIMA(1,1,1) ANALYSIS FOR TESLA")
print("="*60)

# Use Tesla Adj Close from merged dataset (before removing zero volume)
tesla_adj_close = tesla_merged['Adj Close'].values

# Fit ARIMA(1,1,1) model
print("Fitting ARIMA(1,1,1) model...")
arima_model = ARIMA(tesla_adj_close, order=(1, 1, 1))
arima_fitted = arima_model.fit()

# Generate 30-day ahead forecast
forecast_30day = arima_fitted.forecast(steps=30)
mean_forecast_30day = np.mean(forecast_30day)

print(f"30-day ahead forecast mean: {mean_forecast_30day:.2f}")

# ---------- Rolling One-Step-Ahead Forecast (Last 30 Days) ----------
print("\nEvaluating rolling one-step-ahead forecast over last 30 days...")

n_total = len(tesla_adj_close)
n_eval = 30  # Last 30 days for evaluation

predictions = []
actuals = []

for i in range(n_total - n_eval, n_total):
    # Use data up to day i for training
    train_data = tesla_adj_close[:i]
    # Fit ARIMA model
    model = ARIMA(train_data, order=(1, 1, 1))
    fitted = model.fit()
    # One-step-ahead forecast
    pred = fitted.forecast(steps=1)[0]
    predictions.append(pred)
    actuals.append(tesla_adj_close[i])

predictions = np.array(predictions)
actuals = np.array(actuals)

# Calculate RMSE and MAE
rmse = np.sqrt(np.mean((predictions - actuals) ** 2))
mae = np.mean(np.abs(predictions - actuals))

print(f"RMSE (rolling one-step-ahead): {rmse:.4f}")
print(f"MAE (rolling one-step-ahead): {mae:.4f}")

# ---------- Rolling Volatility Analysis ----------
print("\n" + "="*60)
print("ROLLING VOLATILITY ANALYSIS")
print("="*60)

window = 30

# Calculate 30-day rolling standard deviation of returns
uber_rolling_vol = uber_returns['Return'].rolling(window=window).std()
zoom_rolling_vol = zoom_returns['Return'].rolling(window=window).std()
tesla_rolling_vol = tesla_returns['Return'].rolling(window=window).std()

# Time-average of each rolling volatility series (excluding NaN)
uber_avg_vol = uber_rolling_vol.mean()
zoom_avg_vol = zoom_rolling_vol.mean()
tesla_avg_vol = tesla_rolling_vol.mean()

# Average of the three time-averages
overall_volatility = (uber_avg_vol + zoom_avg_vol + tesla_avg_vol) / 3

print(f"Uber average rolling volatility: {uber_avg_vol:.4f}")
print(f"Zoom average rolling volatility: {zoom_avg_vol:.4f}")
print(f"Tesla average rolling volatility: {tesla_avg_vol:.4f}")
print(f"Overall volatility level: {overall_volatility:.4f}")

# ---------- Quantile Regression (90th Percentile) ----------
print("\n" + "="*60)
print("QUANTILE REGRESSION AT 90TH PERCENTILE")
print("="*60)

# Pool all returns and volumes
pooled_returns = pd.concat([
    uber_returns[['Return', 'Volume']].assign(Stock='Uber'),
    zoom_returns[['Return', 'Volume']].assign(Stock='Zoom'),
    tesla_returns[['Return', 'Volume']].assign(Stock='Tesla')
], ignore_index=True)

# Calculate log(Volume)
pooled_returns['Log_Volume'] = np.log(pooled_returns['Volume'])

# Remove any infinite or NaN values
pooled_returns = pooled_returns.replace([np.inf, -np.inf], np.nan).dropna()

# Fit quantile regression at 90th percentile
X = pooled_returns[['Log_Volume']].values
y = pooled_returns['Return'].values

qr_model = QuantileRegressor(quantile=0.90, alpha=0, solver='highs')
qr_model.fit(X, y)

log_volume_coef = qr_model.coef_[0]

print(f"Coefficient of log(Volume) at 90th quantile: {log_volume_coef:.4f}")

# ---------- Individual Stock Quantile Regression for Sensitivity Analysis ----------
print("\nCalculating individual stock sensitivities at 90th percentile...")

stock_sensitivities = {}

for stock_name, df in [('Uber', uber_returns), ('Zoom', zoom_returns), ('Tesla', tesla_returns)]:
    df_temp = df.copy()
    df_temp['Log_Volume'] = np.log(df_temp['Volume'])
    df_temp = df_temp.replace([np.inf, -np.inf], np.nan).dropna()

    X_stock = df_temp[['Log_Volume']].values
    y_stock = df_temp['Return'].values

    qr_stock = QuantileRegressor(quantile=0.90, alpha=0, solver='highs')
    qr_stock.fit(X_stock, y_stock)

    stock_sensitivities[stock_name] = abs(qr_stock.coef_[0])
    print(f"  {stock_name}: {qr_stock.coef_[0]:.4f} (absolute: {abs(qr_stock.coef_[0]):.4f})")

strongest_sensitivity_stock = max(stock_sensitivities, key=stock_sensitivities.get)

# ---------- Two-Sample T-Test (Uber vs Zoom) ----------
print("\n" + "="*60)
print("TWO-SAMPLE T-TEST: UBER vs ZOOM")
print("="*60)

t_statistic, p_value = stats.ttest_ind(uber_returns['Return'], zoom_returns['Return'])

print(f"T-statistic: {t_statistic:.3f}")
print(f"P-value: {p_value:.6f}")

# ---------- Spearman Rank Correlation ----------
print("\n" + "="*60)
print("SPEARMAN RANK CORRELATION (Volume vs Absolute Return)")
print("="*60)

spearman_correlations = []

for stock_name, df in [('Uber', uber_returns), ('Zoom', zoom_returns), ('Tesla', tesla_returns)]:
    abs_return = np.abs(df['Return'])
    correlation, _ = stats.spearmanr(df['Volume'], abs_return)
    spearman_correlations.append(correlation)
    print(f"{stock_name}: {correlation:.4f}")

avg_spearman = np.mean(spearman_correlations)
print(f"Average Spearman correlation: {avg_spearman:.4f}")

# ---------- Tail Spread Analysis ----------
print("\n" + "="*60)
print("TAIL SPREAD ANALYSIS (90th - 10th Percentile)")
print("="*60)

tail_spreads = []

for stock_name, df in [('Uber', uber_returns), ('Zoom', zoom_returns), ('Tesla', tesla_returns)]:
    p90 = np.percentile(df['Return'], 90)
    p10 = np.percentile(df['Return'], 10)
    spread = p90 - p10
    tail_spreads.append(spread)
    print(f"{stock_name}: 90th={p90:.6f}, 10th={p10:.6f}, Spread={spread:.6f}")

avg_tail_spread = np.mean(tail_spreads)
print(f"Average tail spread: {avg_tail_spread:.4f}")

# ---------- Mean Trading Volume ----------
print("\n" + "="*60)
print("MEAN TRADING VOLUME")
print("="*60)

mean_volumes = []

for stock_name, df in [('Uber', uber_returns), ('Zoom', zoom_returns), ('Tesla', tesla_returns)]:
    mean_vol = df['Volume'].mean()
    mean_volumes.append(mean_vol)
    print(f"{stock_name} mean volume: {mean_vol:.2f}")

avg_mean_volume = np.mean(mean_volumes)
print(f"Average of mean volumes: {avg_mean_volume:.2f}")

# ---------- Visualization 1: Multi-Stock Price Line Chart ----------
print("\n" + "="*60)
print("CREATING VISUALIZATIONS")
print("="*60)

print("Creating Multi-Stock Price Line Chart...")
plt.figure(figsize=(14, 7))
plt.plot(uber_merged['Date'], uber_merged['Adj Close'], label='Uber', linewidth=2)
plt.plot(zoom_merged['Date'], zoom_merged['Adj Close'], label='Zoom', linewidth=2)
plt.plot(tesla_merged['Date'], tesla_merged['Adj Close'], label='Tesla', linewidth=2)
plt.xlabel('Date', fontsize=12)
plt.ylabel('Adj Close', fontsize=12)
plt.title('Multi-Stock Price Line Chart', fontsize=14, fontweight='bold')
plt.legend(fontsize=11)
plt.grid(True, alpha=0.3)
plt.tight_layout()
plt.savefig('multi_stock_price_chart.png', dpi=300)
plt.close()

# Calculate min Adj Close across all three stocks
min_adj_close = min(
    uber_merged['Adj Close'].min(),
    zoom_merged['Adj Close'].min(),
    tesla_merged['Adj Close'].min()
)
print(f"Minimum Adj Close displayed: {min_adj_close:.2f}")

# ---------- Visualization 2: Rolling Volatility Trend Chart ----------
print("Creating Rolling Volatility Trend Chart...")
plt.figure(figsize=(14, 7))
# Use tesla_returns dates (aligned with rolling volatility)
plt.plot(tesla_returns['Date'].iloc[window-1:], tesla_rolling_vol.dropna(),
         color='red', linewidth=2)
plt.xlabel('Date', fontsize=12)
plt.ylabel('30-Day Rolling Standard Deviation', fontsize=12)
plt.title('Rolling Volatility Trend Chart', fontsize=14, fontweight='bold')
plt.grid(True, alpha=0.3)
plt.tight_layout()
plt.savefig('rolling_volatility_chart.png', dpi=300)
plt.close()

max_volatility = tesla_rolling_vol.max()
print(f"Maximum volatility value: {max_volatility:.4f}")

# ---------- Visualization 3: Volume-Return Scatter Map ----------
print("Creating Volume-Return Scatter Map...")
zoom_abs_return = np.abs(zoom_returns['Return'])

plt.figure(figsize=(12, 7))
plt.scatter(zoom_returns['Volume'], zoom_abs_return, alpha=0.5, s=30, color='blue')
plt.xlabel('Volume', fontsize=12)
plt.ylabel('Absolute Daily Return', fontsize=12)
plt.title('Volume–Return Scatter Map', fontsize=14, fontweight='bold')
plt.grid(True, alpha=0.3)
plt.tight_layout()
plt.savefig('volume_return_scatter.png', dpi=300)
plt.close()

max_abs_return_zoom = zoom_abs_return.max()
print(f"Highest absolute return value (Zoom): {max_abs_return_zoom:.4f}")

# ---------- Final Summary Output ----------
print("\n" + "="*60)
print("KEY OUTPUTS SUMMARY")
print("="*60)

print("\n--- ARIMA(1,1,1) for Tesla ---")
print(f"Mean of 30-day forecast: {mean_forecast_30day:.2f}")
print(f"RMSE (rolling one-step): {rmse:.4f}")
print(f"MAE (rolling one-step): {mae:.4f}")

print("\n--- Volatility & Regression ---")
print(f"Overall volatility level: {overall_volatility:.4f}")
print(f"Quantile regression coefficient (log Volume, 90th): {log_volume_coef:.4f}")

print("\n--- Statistical Tests ---")
print(f"T-statistic (Uber vs Zoom): {t_statistic:.3f}")

print("\n--- Correlation & Distribution Metrics ---")
print(f"Average Spearman correlation: {avg_spearman:.4f}")
print(f"Average tail spread: {avg_tail_spread:.4f}")

print("\n--- Volume Metrics ---")
print(f"Average mean volume: {avg_mean_volume:.2f}")

print("\n--- Chart Metrics ---")
print(f"Minimum Adj Close (all stocks): {min_adj_close:.2f}")
print(f"Maximum Tesla volatility: {max_volatility:.4f}")
print(f"Highest absolute return (Zoom): {max_abs_return_zoom:.4f}")

print("\n--- Sensitivity Analysis ---")
print(f"Stock with strongest high-quantile return sensitivity to Volume: {strongest_sensitivity_stock}")

print("\n" + "="*60)
print("ANALYSIS COMPLETE")
print("="*60)
print("Charts saved:")
print("  - multi_stock_price_chart.png")
print("  - rolling_volatility_chart.png")
print("  - volume_return_scatter.png")
