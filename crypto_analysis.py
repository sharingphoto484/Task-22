import pandas as pd
import numpy as np
import matplotlib.pyplot as plt
from sklearn.linear_model import LinearRegression, BayesianRidge, Ridge
from sklearn.isotonic import IsotonicRegression
from sklearn.kernel_ridge import KernelRidge
from sklearn.preprocessing import StandardScaler
from sklearn.metrics import mean_squared_error, r2_score
from statsmodels.nonparametric.smoothers_lowess import lowess
import warnings
warnings.filterwarnings('ignore')

# Load datasets
print("Loading cryptocurrency datasets...")
df_historical = pd.read_excel('crypto_historical_365days.xlsx')
df_monthly = pd.read_excel('crypto_monthly_summary.xlsx')
df_yearly = pd.read_excel('crypto_yearly_performance.xlsx')

print(f"Historical data shape: {df_historical.shape}")
print(f"Monthly summary shape: {df_monthly.shape}")
print(f"Yearly performance shape: {df_yearly.shape}")
print("\n" + "="*80)

# =============================================================================
# TASK 1: Bitcoin Price Forecasting with AR(5) Model
# =============================================================================
print("\n1. BITCOIN PRICE FORECASTING - AUTOREGRESSIVE MODEL AR(5)")
print("-" * 80)

# Filter Bitcoin data
bitcoin_df = df_historical[df_historical['coin_id'] == 'bitcoin'].copy()
bitcoin_df = bitcoin_df.sort_values('date').reset_index(drop=True)

# Extract price series
prices = bitcoin_df['price'].values

# Create lag features (lag 1 through lag 5)
for i in range(1, 6):
    bitcoin_df[f'lag_{i}'] = bitcoin_df['price'].shift(i)

# Remove initial 5 observations lacking complete lag history
bitcoin_df = bitcoin_df.dropna(subset=[f'lag_{i}' for i in range(1, 6)])

# Prepare features and target
X = bitcoin_df[[f'lag_{i}' for i in range(1, 6)]].values
y = bitcoin_df['price'].values
dates = bitcoin_df['date'].values

# Split into 80% training and 20% testing (chronologically)
split_idx = int(len(X) * 0.8)
X_train, X_test = X[:split_idx], X[split_idx:]
y_train, y_test = y[:split_idx], y[split_idx:]
dates_test = dates[split_idx:]

# Fit AR(5) model using OLS
ar_model = LinearRegression()
ar_model.fit(X_train, y_train)

# Generate one-step-ahead predictions
y_pred = ar_model.predict(X_test)

# Calculate RMSE
rmse_ar = np.sqrt(mean_squared_error(y_test, y_pred))
print(f"Bitcoin AR(5) RMSE: {rmse_ar:.2f}")

# Generate line plot
plt.figure(figsize=(12, 6))
plt.plot(dates_test, y_test, label='Actual Price', color='blue', linewidth=2)
plt.plot(dates_test, y_pred, label='Predicted Price', color='red', linestyle='--', linewidth=2)
plt.xlabel('Date')
plt.ylabel('Price (USD)')
plt.title('Bitcoin Price: Actual vs AR(5) Predicted')
plt.legend()
plt.xticks(rotation=45)
plt.tight_layout()
plt.savefig('bitcoin_ar5_forecast.png', dpi=300)
print("Saved: bitcoin_ar5_forecast.png")

# =============================================================================
# TASK 2: Ethereum Volatility Analysis with MA(3) Model
# =============================================================================
print("\n2. ETHEREUM VOLATILITY ANALYSIS - MOVING AVERAGE MODEL MA(3)")
print("-" * 80)

# Filter Ethereum data
ethereum_df = df_historical[df_historical['coin_id'] == 'ethereum'].copy()
ethereum_df = ethereum_df.sort_values('date').reset_index(drop=True)

# Extract volatility_7d series, remove missing values
ethereum_df = ethereum_df.dropna(subset=['volatility_7d'])
volatility = ethereum_df['volatility_7d'].values

# Calculate simple moving average (window=3 for MA(3))
ma_window = 3
ma_values = pd.Series(volatility).rolling(window=ma_window, center=False).mean().values

# Calculate residuals (errors)
residuals = volatility - ma_values

# Create MA features using error lag 1 through lag 3
ethereum_df['residual'] = residuals
for i in range(1, 4):
    ethereum_df[f'error_lag_{i}'] = ethereum_df['residual'].shift(i)

# Remove observations lacking complete error history
ethereum_df = ethereum_df.dropna(subset=[f'error_lag_{i}' for i in range(1, 4)])

# Fit MA(3) model: regress current volatility on lagged residuals
X_ma = ethereum_df[[f'error_lag_{i}' for i in range(1, 4)]].values
y_ma = ethereum_df['volatility_7d'].values

ma_model = LinearRegression()
ma_model.fit(X_ma, y_ma)

# Extract coefficients
ma_coefficients = ma_model.coef_

# Find maximum absolute coefficient
max_abs_coef = np.max(np.abs(ma_coefficients))
print(f"Maximum absolute MA(3) coefficient: {max_abs_coef:.4f}")

# Generate autocorrelation plot
from pandas.plotting import autocorrelation_plot
plt.figure(figsize=(10, 6))
volatility_series = pd.Series(ethereum_df['volatility_7d'].values)
acf_values = [volatility_series.autocorr(lag=i) for i in range(1, 11)]
lags = range(1, 11)
plt.bar(lags, acf_values, color='steelblue', alpha=0.7)
plt.axhline(y=0, color='black', linestyle='-', linewidth=0.5)
plt.axhline(y=1.96/np.sqrt(len(volatility_series)), color='red', linestyle='--', linewidth=1, label='95% CI')
plt.axhline(y=-1.96/np.sqrt(len(volatility_series)), color='red', linestyle='--', linewidth=1)
plt.xlabel('Lag Order')
plt.ylabel('Autocorrelation Coefficient')
plt.title('Ethereum Volatility Autocorrelation Function')
plt.legend()
plt.tight_layout()
plt.savefig('ethereum_volatility_acf.png', dpi=300)
print("Saved: ethereum_volatility_acf.png")

# =============================================================================
# TASK 3: Market-Wide Price Trend Smoothing with LOWESS
# =============================================================================
print("\n3. MARKET-WIDE PRICE TREND SMOOTHING - LOWESS")
print("-" * 80)

# Extract month and avg_price
month_sequence = np.arange(1, len(df_monthly) + 1)
avg_prices = df_monthly['avg_price'].values

# Apply LOWESS smoothing with fraction=0.3
lowess_result = lowess(avg_prices, month_sequence, frac=0.3, it=0, delta=0.0, return_sorted=False)

# Calculate mean absolute deviation
mad_lowess = np.mean(np.abs(avg_prices - lowess_result))
print(f"LOWESS Mean Absolute Deviation (MAD): {mad_lowess:.2f}")

# Generate scatter plot with smoothed curve
plt.figure(figsize=(10, 6))
plt.scatter(month_sequence, avg_prices, color='blue', s=100, alpha=0.6, label='Observed')
plt.plot(month_sequence, lowess_result, color='red', linewidth=3, label='LOWESS Smoothed')
plt.xlabel('Month Sequence')
plt.ylabel('Average Price (USD)')
plt.title('Market-Wide Price Trend with LOWESS Smoothing')
plt.legend()
plt.grid(True, alpha=0.3)
plt.tight_layout()
plt.savefig('lowess_price_smoothing.png', dpi=300)
print("Saved: lowess_price_smoothing.png")

# =============================================================================
# TASK 4: Cumulative Return Monotonicity with Isotonic Regression
# =============================================================================
print("\n4. CUMULATIVE RETURN MONOTONICITY - ISOTONIC REGRESSION")
print("-" * 80)

# Extract total_return and start_price
iso_df = df_yearly[['start_price', 'total_return']].dropna()

# Sort by start_price
iso_df = iso_df.sort_values('start_price')
X_iso = iso_df['start_price'].values
y_iso = iso_df['total_return'].values

# Fit isotonic regression (non-decreasing)
iso_model = IsotonicRegression(increasing=True)
y_iso_pred = iso_model.fit_transform(X_iso, y_iso)

# Calculate R-squared
r2_iso = r2_score(y_iso, y_iso_pred)
print(f"Isotonic Regression R-squared: {r2_iso:.4f}")

# Generate step plot
plt.figure(figsize=(10, 6))
plt.scatter(X_iso, y_iso, color='blue', alpha=0.4, s=50, label='Observed')
plt.plot(X_iso, y_iso_pred, color='red', linewidth=2, drawstyle='steps-post', label='Isotonic Fit')
plt.xlabel('Start Price (USD)')
plt.ylabel('Total Return')
plt.title('Isotonic Regression: Monotonic Return vs Start Price')
plt.legend()
plt.grid(True, alpha=0.3)
plt.tight_layout()
plt.savefig('isotonic_return_analysis.png', dpi=300)
print("Saved: isotonic_return_analysis.png")

# =============================================================================
# TASK 5: Volume Prediction with Bayesian Ridge Regression
# =============================================================================
print("\n5. VOLUME PREDICTION - BAYESIAN RIDGE REGRESSION")
print("-" * 80)

# Filter top 10 coins by market_cap_rank
top10_df = df_historical[df_historical['market_cap_rank'] <= 10].copy()

# Extract features and target
volume_df = top10_df[['volume', 'market_cap', 'price', 'volatility_7d']].dropna()
X_volume = volume_df[['market_cap', 'price', 'volatility_7d']].values
y_volume = volume_df['volume'].values

# Standardize predictors
scaler_volume = StandardScaler()
X_volume_scaled = scaler_volume.fit_transform(X_volume)

# Fit Bayesian Ridge Regression
br_model = BayesianRidge(alpha_init=1.0, lambda_init=1.0, compute_score=True)
br_model.fit(X_volume_scaled, y_volume)

# Extract posterior precision parameter alpha
alpha_posterior = br_model.alpha_
print(f"Bayesian Ridge posterior alpha: {alpha_posterior:.2f}")

# Generate posterior distribution plot
plt.figure(figsize=(10, 6))
coef_names = ['Market Cap', 'Price', 'Volatility 7d']
coef_means = br_model.coef_
coef_std = np.sqrt(np.diag(br_model.sigma_))

x_pos = np.arange(len(coef_names))
plt.bar(x_pos, coef_means, yerr=coef_std*1.96, capsize=10, alpha=0.7, color='steelblue')
plt.xlabel('Coefficient Index')
plt.ylabel('Posterior Mean')
plt.title('Bayesian Ridge Regression: Posterior Coefficient Distribution')
plt.xticks(x_pos, coef_names, rotation=45, ha='right')
plt.axhline(y=0, color='black', linestyle='--', linewidth=0.5)
plt.grid(True, alpha=0.3, axis='y')
plt.tight_layout()
plt.savefig('bayesian_ridge_posterior.png', dpi=300)
print("Saved: bayesian_ridge_posterior.png")

# =============================================================================
# TASK 6: Bitcoin Daily Return Non-Linear Modeling with Kernel Ridge
# =============================================================================
print("\n6. BITCOIN DAILY RETURN NON-LINEAR MODELING - KERNEL RIDGE REGRESSION")
print("-" * 80)

# Filter Bitcoin data, exclude missing daily_return
btc_return_df = df_historical[df_historical['coin_id'] == 'bitcoin'].copy()
btc_return_df = btc_return_df.sort_values('date').reset_index(drop=True)

# Create lagged daily return
btc_return_df['daily_return_lag1'] = btc_return_df['daily_return'].shift(1)

# Remove missing values
btc_return_df = btc_return_df[['daily_return', 'daily_return_lag1', 'volatility_7d']].dropna()

# Extract features and target
X_kr = btc_return_df[['daily_return_lag1', 'volatility_7d']].values
y_kr = btc_return_df['daily_return'].values

# Standardize predictors
scaler_kr = StandardScaler()
X_kr_scaled = scaler_kr.fit_transform(X_kr)

# Fit Kernel Ridge Regression with RBF kernel
kr_model = KernelRidge(alpha=1.0, kernel='rbf', gamma=0.1)
kr_model.fit(X_kr_scaled, y_kr)

# Generate predictions
y_kr_pred = kr_model.predict(X_kr_scaled)

# Calculate MSE
mse_kr = mean_squared_error(y_kr, y_kr_pred)
print(f"Kernel Ridge Regression MSE: {mse_kr:.4f}")

# Generate scatter plot
plt.figure(figsize=(10, 6))
plt.scatter(y_kr, y_kr_pred, alpha=0.5, s=30, color='steelblue')
plt.plot([y_kr.min(), y_kr.max()], [y_kr.min(), y_kr.max()],
         'r--', linewidth=2, label='Perfect Prediction')
plt.xlabel('Actual Daily Return')
plt.ylabel('Predicted Daily Return')
plt.title('Bitcoin Daily Return: Kernel Ridge Regression (RBF)')
plt.legend()
plt.grid(True, alpha=0.3)
plt.tight_layout()
plt.savefig('bitcoin_kernel_ridge.png', dpi=300)
print("Saved: bitcoin_kernel_ridge.png")

# =============================================================================
# TASK 7: Market Concentration Analysis - HHI Calculation
# =============================================================================
print("\n7. MARKET CONCENTRATION ANALYSIS - HERFINDAHL-HIRSCHMAN INDEX")
print("-" * 80)

# Extract end_price for all coins
end_prices = df_yearly['end_price'].values

# Calculate market shares
total_end_price = np.sum(end_prices)
market_shares = end_prices / total_end_price

# Calculate HHI (sum of squared market shares)
hhi = np.sum(market_shares ** 2)
print(f"Herfindahl-Hirschman Index (HHI): {hhi:.4f}")

# =============================================================================
# TASK 8: Temporal Volatility Clustering Detection - Autocorrelation
# =============================================================================
print("\n8. TEMPORAL VOLATILITY CLUSTERING - FIRST-ORDER AUTOCORRELATION")
print("-" * 80)

# Filter Ethereum volatility
eth_vol_df = df_historical[df_historical['coin_id'] == 'ethereum'].copy()
eth_vol_df = eth_vol_df.sort_values('date').reset_index(drop=True)
eth_vol_series = eth_vol_df['volatility_7d'].dropna()

# Calculate first-order autocorrelation
first_order_acf = eth_vol_series.autocorr(lag=1)
print(f"Ethereum volatility first-order autocorrelation: {first_order_acf:.4f}")

# =============================================================================
# TASK 9: Portfolio Diversification Metric - Coefficient of Variation
# =============================================================================
print("\n9. PORTFOLIO DIVERSIFICATION - COEFFICIENT OF VARIATION")
print("-" * 80)

# Group by coin_id, calculate std and mean of daily_return
cv_df = df_historical.groupby('coin_id')['daily_return'].agg(['std', 'mean']).reset_index()
cv_df = cv_df.dropna()

# Calculate coefficient of variation
cv_df['cv'] = cv_df['std'] / np.abs(cv_df['mean'])

# Find coin with minimum CV
min_cv_idx = cv_df['cv'].idxmin()
min_cv_coin_id = cv_df.loc[min_cv_idx, 'coin_id']

# Get coin_name for this coin_id
min_cv_coin_name = df_historical[df_historical['coin_id'] == min_cv_coin_id]['coin_name'].iloc[0]
min_cv_value = cv_df.loc[min_cv_idx, 'cv']

print(f"Coin with minimum coefficient of variation: {min_cv_coin_name}")
print(f"Coefficient of variation: {min_cv_value:.4f}")

# =============================================================================
# TASK 10: Comparative Analysis
# =============================================================================
print("\n" + "="*80)
print("10. COMPARATIVE ANALYSIS: AUTOREGRESSIVE vs BAYESIAN APPROACHES")
print("="*80)

print(f"\nAutoregressive AR(5) Model Performance:")
print(f"  - RMSE: {rmse_ar:.2f}")
print(f"  - Captures temporal dependencies through lag structure")
print(f"  - Point estimates without uncertainty quantification")

print(f"\nBayesian Ridge Regression Performance:")
print(f"  - Posterior alpha: {alpha_posterior:.2f}")
print(f"  - Provides probabilistic predictions with uncertainty")
print(f"  - Automatic relevance determination via regularization")

print(f"\n{'CONCLUSION':^80}")
print("-"*80)
print("""
Based on the analysis results:

The AUTOREGRESSIVE approach captures cryptocurrency price dynamics more
effectively for point prediction tasks due to its explicit modeling of
temporal dependencies through lagged price observations. The AR(5) model
achieved RMSE of {:.2f}, directly incorporating price momentum and
mean-reversion patterns inherent in crypto markets.

The BAYESIAN approach provides superior risk assessment capabilities through
uncertainty quantification. With posterior alpha of {:.2f}, it offers
probabilistic forecasts essential for portfolio optimization and risk
management, though it requires additional predictors beyond temporal lags.

RECOMMENDATION: For cryptocurrency price forecasting, use autoregressive
models for point predictions combined with Bayesian methods for uncertainty
quantification in risk-adjusted decision making.
""".format(rmse_ar, alpha_posterior))

print("\n" + "="*80)
print("ANALYSIS COMPLETE - All visualizations saved")
print("="*80)
