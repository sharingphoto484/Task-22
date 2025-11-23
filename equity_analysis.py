# ==========================================
# Equity Price Behavior Analysis Script
# Requirements: pandas, numpy, matplotlib, statsmodels, scipy
# Input files: IOC.csv, NTPC.csv, HINDALCO.csv (in same directory)
# Output files: Rolling_Correlation_Line_Chart.png
# ==========================================

import pandas as pd
import numpy as np
import matplotlib.pyplot as plt
from scipy import stats
from statsmodels.tsa.arima.model import ARIMA
from statsmodels.tsa.stattools import grangercausalitytests
import warnings
warnings.filterwarnings('ignore')

# ---------- Load CSVs Robustly ----------
print("Loading datasets...")
ioc = pd.read_csv('IOC.csv')
ntpc = pd.read_csv('NTPC.csv')
hindalco = pd.read_csv('HINDALCO.csv')

# ---------- Select Required Columns ----------
# Note: Using 'Close' as Adj Close since files don't have explicit Adj Close column
required_cols = ['Date', 'Open', 'High', 'Low', 'Close', 'Volume']
ioc = ioc[required_cols].copy()
ntpc = ntpc[required_cols].copy()
hindalco = hindalco[required_cols].copy()

# Rename Close to Adj Close for consistency with requirements
ioc.rename(columns={'Close': 'Adj Close'}, inplace=True)
ntpc.rename(columns={'Close': 'Adj Close'}, inplace=True)
hindalco.rename(columns={'Close': 'Adj Close'}, inplace=True)

# ---------- Convert Date to Datetime ----------
ioc['Date'] = pd.to_datetime(ioc['Date'])
ntpc['Date'] = pd.to_datetime(ntpc['Date'])
hindalco['Date'] = pd.to_datetime(hindalco['Date'])

# ---------- Remove Missing and Non-Numeric Values ----------
print("Cleaning data...")
# Convert numeric columns to numeric, coercing errors to NaN
numeric_cols = ['Open', 'High', 'Low', 'Adj Close', 'Volume']
for df in [ioc, ntpc, hindalco]:
    for col in numeric_cols:
        df[col] = pd.to_numeric(df[col], errors='coerce')

# Drop rows with any missing values
ioc = ioc.dropna()
ntpc = ntpc.dropna()
hindalco = hindalco.dropna()

# ---------- Align Datasets by Common Dates ----------
print("Aligning datasets by common dates...")
ioc_dates = set(ioc['Date'])
ntpc_dates = set(ntpc['Date'])
hindalco_dates = set(hindalco['Date'])

common_dates = ioc_dates.intersection(ntpc_dates).intersection(hindalco_dates)
common_dates = sorted(list(common_dates))

ioc = ioc[ioc['Date'].isin(common_dates)].sort_values('Date').reset_index(drop=True)
ntpc = ntpc[ntpc['Date'].isin(common_dates)].sort_values('Date').reset_index(drop=True)
hindalco = hindalco[hindalco['Date'].isin(common_dates)].sort_values('Date').reset_index(drop=True)

print(f"Common dates found: {len(common_dates)}")

# ---------- Calculate Daily Returns (Log Difference of Adj Close) ----------
print("Calculating daily returns...")
ioc['Return'] = np.log(ioc['Adj Close'] / ioc['Adj Close'].shift(1))
ntpc['Return'] = np.log(ntpc['Adj Close'] / ntpc['Adj Close'].shift(1))
hindalco['Return'] = np.log(hindalco['Adj Close'] / hindalco['Adj Close'].shift(1))

# Remove first row (NaN return)
ioc = ioc.iloc[1:].reset_index(drop=True)
ntpc = ntpc.iloc[1:].reset_index(drop=True)
hindalco = hindalco.iloc[1:].reset_index(drop=True)

# ---------- Exclude Days with Volume = 0 from Return Operations ----------
print("Filtering out zero volume days...")
ioc_returns = ioc[ioc['Volume'] > 0].copy()
ntpc_returns = ntpc[ntpc['Volume'] > 0].copy()
hindalco_returns = hindalco[hindalco['Volume'] > 0].copy()

# Re-align after volume filtering
common_dates_returns = set(ioc_returns['Date']).intersection(
    set(ntpc_returns['Date'])).intersection(set(hindalco_returns['Date']))
common_dates_returns = sorted(list(common_dates_returns))

ioc_returns = ioc_returns[ioc_returns['Date'].isin(common_dates_returns)].sort_values('Date').reset_index(drop=True)
ntpc_returns = ntpc_returns[ntpc_returns['Date'].isin(common_dates_returns)].sort_values('Date').reset_index(drop=True)
hindalco_returns = hindalco_returns[hindalco_returns['Date'].isin(common_dates_returns)].sort_values('Date').reset_index(drop=True)

print(f"Trading days with non-zero volume: {len(common_dates_returns)}")

# ---------- ARIMA(1,1,1) Model on HINDALCO Adj Close ----------
print("\nFitting ARIMA(1,1,1) model on HINDALCO...")
# Use full hindalco dataset (before return calculation) for ARIMA
hindalco_full = hindalco.copy()
arima_model = ARIMA(hindalco_full['Adj Close'], order=(1, 1, 1))
arima_fit = arima_model.fit()

# Generate 30-day forecast
forecast = arima_fit.forecast(steps=30)
forecast_mean = forecast.mean()

print(f"ARIMA Forecast Mean (30 days): {forecast_mean:.2f}")

# ---------- Two-Sample T-Test (IOC vs NTPC Returns) ----------
print("\nPerforming two-sample t-test...")
t_stat, p_value = stats.ttest_ind(ioc_returns['Return'], ntpc_returns['Return'])

print(f"T-Test Results:")
print(f"  T-Statistic: {t_stat:.3f}")
print(f"  P-Value: {p_value:.3f}")

# ---------- 30-Day Rolling Volatility (Standard Deviation) ----------
print("\nCalculating 30-day rolling volatility...")
ioc_returns['Rolling_Vol'] = ioc_returns['Return'].rolling(window=30).std()
ntpc_returns['Rolling_Vol'] = ntpc_returns['Return'].rolling(window=30).std()
hindalco_returns['Rolling_Vol'] = hindalco_returns['Return'].rolling(window=30).std()

# Average rolling volatility across full sample (excluding NaN from first 29 rows)
ioc_mean_vol = ioc_returns['Rolling_Vol'].mean()
ntpc_mean_vol = ntpc_returns['Rolling_Vol'].mean()
hindalco_mean_vol = hindalco_returns['Rolling_Vol'].mean()

print(f"Mean 30-Day Rolling Volatility:")
print(f"  IOC: {ioc_mean_vol:.4f}")
print(f"  NTPC: {ntpc_mean_vol:.4f}")
print(f"  HINDALCO: {hindalco_mean_vol:.4f}")

# Identify lowest volatility company
vol_dict = {'IOC': ioc_mean_vol, 'NTPC': ntpc_mean_vol, 'HINDALCO': hindalco_mean_vol}
lowest_vol_company = min(vol_dict, key=vol_dict.get)
print(f"  Lowest Volatility Company: {lowest_vol_company}")

# ---------- Granger Causality Test (IOC → HINDALCO) ----------
print("\nPerforming Granger causality test (IOC → HINDALCO)...")
# Prepare data: combine IOC and HINDALCO returns
granger_data = pd.DataFrame({
    'HINDALCO': hindalco_returns['Return'].values,
    'IOC': ioc_returns['Return'].values
})

# Remove any remaining NaN values
granger_data = granger_data.dropna()

# Perform test with max lag = 1
granger_result = grangercausalitytests(granger_data[['HINDALCO', 'IOC']], maxlag=1, verbose=False)

# Extract F-statistic and p-value for lag 1
f_stat = granger_result[1][0]['ssr_ftest'][0]
granger_p_value = granger_result[1][0]['ssr_ftest'][1]

print(f"Granger Causality Test Results (Lag 1):")
print(f"  F-Statistic: {f_stat:.3f}")
print(f"  P-Value: {granger_p_value:.3f}")

# ---------- 30-Day Rolling Correlation (IOC vs NTPC) ----------
print("\nCalculating 30-day rolling correlation...")
# Merge IOC and NTPC returns by date for correlation calculation
corr_data = pd.DataFrame({
    'Date': ioc_returns['Date'],
    'IOC_Return': ioc_returns['Return'].values,
    'NTPC_Return': ntpc_returns['Return'].values
})

# Calculate 30-day rolling correlation
corr_data['Rolling_Corr'] = corr_data['IOC_Return'].rolling(window=30).corr(corr_data['NTPC_Return'])

# Remove NaN values from rolling window
corr_data_clean = corr_data.dropna()

# Find minimum correlation
min_corr = corr_data_clean['Rolling_Corr'].min()
print(f"  Minimum Rolling Correlation: {min_corr:.3f}")

# ---------- Visualization: Rolling Correlation Line Chart ----------
print("\nCreating Rolling Correlation Line Chart...")
plt.figure(figsize=(12, 6))
plt.plot(corr_data_clean['Date'], corr_data_clean['Rolling_Corr'], linewidth=1, color='blue')
plt.xlabel('Date', fontsize=12)
plt.ylabel('30-Day Rolling Correlation', fontsize=12)
plt.title('Rolling Correlation Between IOC and NTPC Daily Returns', fontsize=14, fontweight='bold')
plt.grid(True, alpha=0.3)
plt.tight_layout()
plt.savefig('Rolling_Correlation_Line_Chart.png', dpi=300)
print("  Chart saved as: Rolling_Correlation_Line_Chart.png")

# ---------- Final Summary Output ----------
print("\n" + "="*60)
print("FINAL RESULTS SUMMARY")
print("="*60)
print(f"1. ARIMA Forecast Mean (30 days): {forecast_mean:.2f}")
print(f"2. T-Test T-Statistic: {t_stat:.3f}")
print(f"3. T-Test P-Value: {p_value:.3f}")
print(f"4. Mean Rolling Volatility - IOC: {ioc_mean_vol:.4f}")
print(f"5. Mean Rolling Volatility - NTPC: {ntpc_mean_vol:.4f}")
print(f"6. Mean Rolling Volatility - HINDALCO: {hindalco_mean_vol:.4f}")
print(f"7. Granger Causality F-Statistic: {f_stat:.3f}")
print(f"8. Granger Causality P-Value: {granger_p_value:.3f}")
print(f"9. Minimum Rolling Correlation: {min_corr:.3f}")
print(f"10. Lowest Volatility Company: {lowest_vol_company}")
print("="*60)
