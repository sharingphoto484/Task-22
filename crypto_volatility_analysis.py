# ==========================================
# Cryptocurrency Volatility & Return Analysis Script
# Requirements: pandas, numpy, matplotlib, scipy, scikit-learn, openpyxl, statsmodels
# Input files: crypto_historical_365days.xlsx, crypto_yearly_performance.xlsx, crypto_monthly_summary.xlsx
# Output files: bitcoin_ar_forecast.png, ethereum_ma_autocorr.png, market_lowess_smoothing.png,
#               isotonic_return_fit.png, bayesian_posterior_uncertainty.png, bitcoin_krr_scatter.png
# Key Outputs: RMSE, max MA coefficient, MAD, R-squared, Bayesian alpha, MSE, HHI, autocorrelation, min CV coin
# ==========================================

import pandas as pd
import numpy as np
import matplotlib.pyplot as plt
from sklearn.linear_model import LinearRegression, BayesianRidge
from sklearn.isotonic import IsotonicRegression
from sklearn.kernel_ridge import KernelRidge
from sklearn.preprocessing import StandardScaler
from sklearn.metrics import mean_squared_error, r2_score
from scipy.stats import pearsonr
from statsmodels.nonparametric.smoothers_lowess import lowess
import warnings
warnings.filterwarnings('ignore')

# Open-ended question answer:
# Bayesian ridge regression captures cryptocurrency dynamics more effectively by quantifying parameter uncertainty and preventing overfitting through probabilistic regularization, whereas autoregressive models assume fixed coefficients that cannot adapt to regime changes in volatile crypto markets.

# ---------- Load Data Files ----------
print("Loading cryptocurrency datasets...")
df_historical = pd.read_excel('crypto_historical_365days.xlsx')
df_yearly = pd.read_excel('crypto_yearly_performance.xlsx')
df_monthly = pd.read_excel('crypto_monthly_summary.xlsx')

print(f"Historical data shape: {df_historical.shape}")
print(f"Yearly performance shape: {df_yearly.shape}")
print(f"Monthly summary shape: {df_monthly.shape}\n")

# ==========================================
# TASK 1: Bitcoin Price Forecasting - Autoregressive AR(5)
# ==========================================
print("=" * 60)
print("TASK 1: Bitcoin Price Forecasting - Autoregressive AR(5)")
print("=" * 60)

# ---------- Filter Bitcoin Data ----------
bitcoin_data = df_historical[df_historical['coin_id'] == 'bitcoin'].copy()
bitcoin_data = bitcoin_data.sort_values('date').reset_index(drop=True)

# ---------- Create Lag Features ----------
bitcoin_data['lag1'] = bitcoin_data['price'].shift(1)
bitcoin_data['lag2'] = bitcoin_data['price'].shift(2)
bitcoin_data['lag3'] = bitcoin_data['price'].shift(3)
bitcoin_data['lag4'] = bitcoin_data['price'].shift(4)
bitcoin_data['lag5'] = bitcoin_data['price'].shift(5)

# ---------- Remove Initial Observations with Missing Lags ----------
bitcoin_ar = bitcoin_data.dropna(subset=['lag1', 'lag2', 'lag3', 'lag4', 'lag5']).copy()

# ---------- Prepare Features and Target ----------
X_ar = bitcoin_ar[['lag1', 'lag2', 'lag3', 'lag4', 'lag5']].values
y_ar = bitcoin_ar['price'].values

# ---------- Train-Test Split (80-20) Chronologically ----------
split_idx = int(len(X_ar) * 0.8)
X_train_ar = X_ar[:split_idx]
X_test_ar = X_ar[split_idx:]
y_train_ar = y_ar[:split_idx]
y_test_ar = y_ar[split_idx:]
dates_test = bitcoin_ar['date'].iloc[split_idx:].values

# ---------- Fit AR(5) Model ----------
ar_model = LinearRegression()
ar_model.fit(X_train_ar, y_train_ar)

# ---------- Generate Predictions ----------
y_pred_ar = ar_model.predict(X_test_ar)

# ---------- Calculate RMSE ----------
rmse_ar = np.sqrt(mean_squared_error(y_test_ar, y_pred_ar))
print(f"Bitcoin AR(5) RMSE: {rmse_ar:.2f}")

# ---------- Visualization: Actual vs Predicted ----------
plt.figure(figsize=(12, 5))
plt.plot(dates_test, y_test_ar, label='Actual Price', color='blue', linewidth=2)
plt.plot(dates_test, y_pred_ar, label='Predicted Price', color='red', linewidth=2, linestyle='--')
plt.xlabel('Date')
plt.ylabel('Price (USD)')
plt.title('Bitcoin Price Forecasting - AR(5) Model')
plt.legend()
plt.grid(True, alpha=0.3)
plt.xticks(rotation=45)
plt.tight_layout()
plt.savefig('bitcoin_ar_forecast.png', dpi=300)
plt.close()
print("Saved: bitcoin_ar_forecast.png\n")

# ==========================================
# TASK 2: Ethereum Volatility - Moving Average MA(3)
# ==========================================
print("=" * 60)
print("TASK 2: Ethereum Volatility Analysis - Moving Average MA(3)")
print("=" * 60)

# ---------- Filter Ethereum Data ----------
ethereum_data = df_historical[df_historical['coin_id'] == 'ethereum'].copy()
ethereum_data = ethereum_data.sort_values('date').reset_index(drop=True)

# ---------- Extract Volatility Series ----------
volatility_series = ethereum_data['volatility_7d'].dropna().values

# ---------- Calculate Simple Moving Average and Residuals ----------
window_size = 7
sma = pd.Series(volatility_series).rolling(window=window_size, center=False).mean().values
residuals = volatility_series - sma

# ---------- Create MA Features from Error Lags ----------
residuals_clean = residuals[~np.isnan(residuals)]
error_lag1 = np.roll(residuals_clean, 1)
error_lag2 = np.roll(residuals_clean, 2)
error_lag3 = np.roll(residuals_clean, 3)

# ---------- Remove Initial Observations ----------
n_lags = 3
X_ma = np.column_stack([error_lag1[n_lags:], error_lag2[n_lags:], error_lag3[n_lags:]])
y_ma = volatility_series[window_size + n_lags:len(volatility_series) - (len(volatility_series) - len(residuals_clean))]

# Adjust for proper alignment
min_len = min(len(X_ma), len(residuals_clean) - n_lags)
X_ma = X_ma[:min_len]
y_ma = residuals_clean[n_lags:n_lags + min_len]

# ---------- Fit MA(3) Model ----------
ma_model = LinearRegression()
ma_model.fit(X_ma, y_ma)

# ---------- Find Maximum Absolute Coefficient ----------
coefficients = ma_model.coef_
max_abs_coef = np.max(np.abs(coefficients))
print(f"Maximum Absolute MA Coefficient: {max_abs_coef:.4f}")

# ---------- Calculate Autocorrelation for Visualization ----------
volatility_clean = ethereum_data['volatility_7d'].dropna().values
acf_values = []
for lag in range(1, 11):
    if len(volatility_clean) > lag:
        corr, _ = pearsonr(volatility_clean[:-lag], volatility_clean[lag:])
        acf_values.append(corr)
    else:
        acf_values.append(0)

# ---------- Visualization: Autocorrelation Plot ----------
plt.figure(figsize=(10, 5))
lags = np.arange(1, 11)
plt.bar(lags, acf_values, color='steelblue', alpha=0.7)
plt.axhline(y=0, color='black', linestyle='-', linewidth=0.8)
plt.axhline(y=1.96/np.sqrt(len(volatility_clean)), color='red', linestyle='--', linewidth=1, label='95% Confidence')
plt.axhline(y=-1.96/np.sqrt(len(volatility_clean)), color='red', linestyle='--', linewidth=1)
plt.xlabel('Lag Order')
plt.ylabel('Autocorrelation Coefficient')
plt.title('Ethereum Volatility Autocorrelation Function')
plt.legend()
plt.grid(True, alpha=0.3)
plt.tight_layout()
plt.savefig('ethereum_ma_autocorr.png', dpi=300)
plt.close()
print("Saved: ethereum_ma_autocorr.png\n")

# ==========================================
# TASK 3: Market-Wide Price Trend - LOWESS Smoothing
# ==========================================
print("=" * 60)
print("TASK 3: Market-Wide Price Trend - LOWESS Smoothing")
print("=" * 60)

# ---------- Extract Month and Average Price ----------
df_monthly_clean = df_monthly[['month', 'avg_price']].copy()
df_monthly_clean['month_numeric'] = np.arange(1, len(df_monthly_clean) + 1)

# ---------- Apply LOWESS Smoothing ----------
observed_prices = df_monthly_clean['avg_price'].values
month_numeric = df_monthly_clean['month_numeric'].values

smoothed = lowess(observed_prices, month_numeric, frac=0.3, it=0, delta=0.0, return_sorted=False)

# ---------- Calculate Mean Absolute Deviation ----------
mad_lowess = np.mean(np.abs(observed_prices - smoothed))
print(f"LOWESS Mean Absolute Deviation (MAD): {mad_lowess:.2f}")

# ---------- Visualization: Observed vs Smoothed ----------
plt.figure(figsize=(10, 5))
plt.scatter(month_numeric, observed_prices, color='blue', alpha=0.6, s=80, label='Observed Avg Price')
plt.plot(month_numeric, smoothed, color='red', linewidth=2.5, label='LOWESS Smoothed')
plt.xlabel('Month Sequence')
plt.ylabel('Average Price (USD)')
plt.title('Market-Wide Price Trend - LOWESS Smoothing (fraction=0.3)')
plt.legend()
plt.grid(True, alpha=0.3)
plt.tight_layout()
plt.savefig('market_lowess_smoothing.png', dpi=300)
plt.close()
print("Saved: market_lowess_smoothing.png\n")

# ==========================================
# TASK 4: Cumulative Return Monotonicity - Isotonic Regression
# ==========================================
print("=" * 60)
print("TASK 4: Cumulative Return Monotonicity - Isotonic Regression")
print("=" * 60)

# ---------- Extract Features and Remove Missing Values ----------
isotonic_data = df_yearly[['start_price', 'total_return']].dropna()
isotonic_data = isotonic_data.sort_values('start_price').reset_index(drop=True)

X_isotonic = isotonic_data['start_price'].values
y_isotonic = isotonic_data['total_return'].values

# ---------- Fit Isotonic Regression ----------
isotonic_model = IsotonicRegression(increasing=True)
y_isotonic_pred = isotonic_model.fit_transform(X_isotonic, y_isotonic)

# ---------- Calculate R-Squared ----------
r2_isotonic = r2_score(y_isotonic, y_isotonic_pred)
print(f"Isotonic Regression R-Squared: {r2_isotonic:.4f}")

# ---------- Visualization: Step Plot ----------
plt.figure(figsize=(10, 5))
plt.scatter(X_isotonic, y_isotonic, color='lightblue', alpha=0.5, s=50, label='Observed Returns')
plt.step(X_isotonic, y_isotonic_pred, color='darkblue', linewidth=2.5, where='post', label='Isotonic Fit')
plt.xlabel('Start Price (USD)')
plt.ylabel('Total Return')
plt.title('Cumulative Return Monotonicity - Isotonic Regression')
plt.legend()
plt.grid(True, alpha=0.3)
plt.tight_layout()
plt.savefig('isotonic_return_fit.png', dpi=300)
plt.close()
print("Saved: isotonic_return_fit.png\n")

# ==========================================
# TASK 5: Volume Prediction - Bayesian Ridge Regression
# ==========================================
print("=" * 60)
print("TASK 5: Volume Prediction - Bayesian Ridge Regression")
print("=" * 60)

# ---------- Filter Top 10 Coins by Market Cap Rank ----------
top10_data = df_historical[df_historical['market_cap_rank'] <= 10].copy()

# ---------- Extract Predictors and Target ----------
bayesian_data = top10_data[['volume', 'market_cap', 'price', 'volatility_7d']].dropna()
X_bayesian = bayesian_data[['market_cap', 'price', 'volatility_7d']].values
y_bayesian = bayesian_data['volume'].values

# ---------- Standardize Features ----------
scaler_bayesian = StandardScaler()
X_bayesian_scaled = scaler_bayesian.fit_transform(X_bayesian)

# ---------- Fit Bayesian Ridge Regression ----------
bayesian_model = BayesianRidge(alpha_init=1.0, lambda_init=1.0, compute_score=True)
bayesian_model.fit(X_bayesian_scaled, y_bayesian)

# ---------- Extract Alpha Parameter ----------
alpha_bayesian = bayesian_model.alpha_
print(f"Bayesian Ridge Alpha (Regularization Strength): {alpha_bayesian:.2f}")

# ---------- Visualization: Posterior Distribution ----------
coef_means = bayesian_model.coef_
coef_std = np.sqrt(np.diag(bayesian_model.sigma_))

plt.figure(figsize=(10, 5))
x_pos = np.arange(len(coef_means))
plt.bar(x_pos, coef_means, yerr=coef_std * 1.96, capsize=5, alpha=0.7, color='teal')
plt.xlabel('Coefficient Index')
plt.ylabel('Posterior Mean')
plt.title('Bayesian Ridge Regression - Posterior Distribution with 95% Uncertainty')
plt.xticks(x_pos, ['Market Cap', 'Price', 'Volatility 7d'])
plt.grid(True, alpha=0.3, axis='y')
plt.tight_layout()
plt.savefig('bayesian_posterior_uncertainty.png', dpi=300)
plt.close()
print("Saved: bayesian_posterior_uncertainty.png\n")

# ==========================================
# TASK 6: Bitcoin Daily Return - Kernel Ridge Regression
# ==========================================
print("=" * 60)
print("TASK 6: Bitcoin Daily Return - Kernel Ridge Regression (RBF)")
print("=" * 60)

# ---------- Filter Bitcoin with Non-Missing Returns ----------
bitcoin_krr = df_historical[(df_historical['coin_id'] == 'bitcoin') &
                             (df_historical['daily_return'].notna())].copy()
bitcoin_krr = bitcoin_krr.sort_values('date').reset_index(drop=True)

# ---------- Create Lagged Daily Return ----------
bitcoin_krr['daily_return_lag1'] = bitcoin_krr['daily_return'].shift(1)

# ---------- Remove Missing Predictors ----------
krr_data = bitcoin_krr[['daily_return', 'daily_return_lag1', 'volatility_7d']].dropna()
X_krr = krr_data[['daily_return_lag1', 'volatility_7d']].values
y_krr = krr_data['daily_return'].values

# ---------- Standardize Predictors ----------
scaler_krr = StandardScaler()
X_krr_scaled = scaler_krr.fit_transform(X_krr)

# ---------- Fit Kernel Ridge Regression ----------
krr_model = KernelRidge(alpha=1.0, kernel='rbf', gamma=0.1)
krr_model.fit(X_krr_scaled, y_krr)

# ---------- Generate Predictions ----------
y_krr_pred = krr_model.predict(X_krr_scaled)

# ---------- Calculate MSE ----------
mse_krr = mean_squared_error(y_krr, y_krr_pred)
print(f"Kernel Ridge Regression MSE: {mse_krr:.4f}")

# ---------- Visualization: Scatter Plot with Diagonal ----------
plt.figure(figsize=(8, 8))
plt.scatter(y_krr, y_krr_pred, alpha=0.4, s=20, color='purple')
min_val = min(y_krr.min(), y_krr_pred.min())
max_val = max(y_krr.max(), y_krr_pred.max())
plt.plot([min_val, max_val], [min_val, max_val], 'r--', linewidth=2, label='Perfect Prediction')
plt.xlabel('Actual Daily Return')
plt.ylabel('Predicted Daily Return')
plt.title('Bitcoin Daily Return - Kernel Ridge Regression (RBF Kernel)')
plt.legend()
plt.grid(True, alpha=0.3)
plt.tight_layout()
plt.savefig('bitcoin_krr_scatter.png', dpi=300)
plt.close()
print("Saved: bitcoin_krr_scatter.png\n")

# ==========================================
# TASK 7: Market Concentration - Herfindahl-Hirschman Index
# ==========================================
print("=" * 60)
print("TASK 7: Market Concentration Analysis - HHI")
print("=" * 60)

# ---------- Extract End Prices ----------
end_prices = df_yearly['end_price'].dropna().values

# ---------- Calculate Market Shares ----------
total_price = np.sum(end_prices)
market_shares = end_prices / total_price

# ---------- Calculate HHI ----------
hhi = np.sum(market_shares ** 2)
print(f"Herfindahl-Hirschman Index (HHI): {hhi:.4f}\n")

# ==========================================
# TASK 8: Temporal Volatility Clustering - Autocorrelation
# ==========================================
print("=" * 60)
print("TASK 8: Temporal Volatility Clustering Detection")
print("=" * 60)

# ---------- Filter Ethereum Volatility ----------
ethereum_vol = df_historical[df_historical['coin_id'] == 'ethereum'].copy()
ethereum_vol = ethereum_vol.sort_values('date')
vol_series = ethereum_vol['volatility_7d'].dropna().values

# ---------- Calculate First-Order Autocorrelation ----------
if len(vol_series) > 1:
    first_order_autocorr, _ = pearsonr(vol_series[:-1], vol_series[1:])
    print(f"Ethereum Volatility First-Order Autocorrelation: {first_order_autocorr:.4f}\n")
else:
    print("Insufficient data for autocorrelation calculation\n")

# ==========================================
# TASK 9: Portfolio Diversification - Coefficient of Variation
# ==========================================
print("=" * 60)
print("TASK 9: Portfolio Diversification Metric - Coefficient of Variation")
print("=" * 60)

# ---------- Group by Coin and Calculate CV ----------
cv_analysis = df_historical.groupby('coin_id').agg({
    'daily_return': ['std', 'mean'],
    'coin_name': 'first'
}).reset_index()

cv_analysis.columns = ['coin_id', 'std_return', 'mean_return', 'coin_name']

# ---------- Calculate Coefficient of Variation ----------
cv_analysis['cv'] = cv_analysis['std_return'] / np.abs(cv_analysis['mean_return'])

# ---------- Find Minimum CV ----------
cv_analysis_clean = cv_analysis.dropna(subset=['cv'])
cv_analysis_clean = cv_analysis_clean[np.isfinite(cv_analysis_clean['cv'])]

if len(cv_analysis_clean) > 0:
    min_cv_coin = cv_analysis_clean.loc[cv_analysis_clean['cv'].idxmin(), 'coin_name']
    print(f"Coin with Minimum Coefficient of Variation: {min_cv_coin}\n")
else:
    print("Unable to determine minimum CV coin\n")

# ==========================================
# SUMMARY OF KEY OUTPUTS
# ==========================================
print("=" * 60)
print("SUMMARY OF KEY OUTPUTS")
print("=" * 60)
print(f"1. Bitcoin AR(5) RMSE: {rmse_ar:.2f}")
print(f"2. Ethereum MA(3) Max Absolute Coefficient: {max_abs_coef:.4f}")
print(f"3. LOWESS Smoothing MAD: {mad_lowess:.2f}")
print(f"4. Isotonic Regression R-Squared: {r2_isotonic:.4f}")
print(f"5. Bayesian Ridge Alpha: {alpha_bayesian:.2f}")
print(f"6. Kernel Ridge MSE: {mse_krr:.4f}")
print(f"7. Herfindahl-Hirschman Index: {hhi:.4f}")
print(f"8. Ethereum First-Order Autocorrelation: {first_order_autocorr:.4f}")
print(f"9. Min CV Coin: {min_cv_coin}")
print("\n" + "=" * 60)
print("Analysis Complete - All visualizations saved")
print("=" * 60)
