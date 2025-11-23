# ==========================================
# Integrated Equity Price Behavior Analysis Script
# ==========================================
# Requirements: pandas, numpy, matplotlib, scipy, statsmodels
# Input files: IOC.csv, NTPC.csv, HINDALCO.csv (in same directory)
# Output files: Rolling_Correlation_Line_Chart.png
#
# This script performs:
# - ARIMA(1,1,1) forecasting on HINDALCO
# - Two-sample t-test on IOC vs NTPC returns
# - Rolling volatility calculations
# - Granger causality testing
# - Rolling correlation visualization
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
ioc_df = pd.read_csv('IOC.csv')
ntpc_df = pd.read_csv('NTPC.csv')
hindalco_df = pd.read_csv('HINDALCO.csv')

# ---------- Select Required Columns ----------
# Note: These files use 'Close' instead of 'Adj Close'
required_cols = ['Date', 'Open', 'High', 'Low', 'Close', 'Volume']

ioc_df = ioc_df[required_cols].copy()
ntpc_df = ntpc_df[required_cols].copy()
hindalco_df = hindalco_df[required_cols].copy()

# Rename Close to Adj Close for consistency with requirements
ioc_df.rename(columns={'Close': 'Adj Close'}, inplace=True)
ntpc_df.rename(columns={'Close': 'Adj Close'}, inplace=True)
hindalco_df.rename(columns={'Close': 'Adj Close'}, inplace=True)

# ---------- Convert Date to datetime ----------
ioc_df['Date'] = pd.to_datetime(ioc_df['Date'], errors='coerce')
ntpc_df['Date'] = pd.to_datetime(ntpc_df['Date'], errors='coerce')
hindalco_df['Date'] = pd.to_datetime(hindalco_df['Date'], errors='coerce')

# ---------- Remove Rows with Missing or Non-numeric Values ----------
def clean_dataframe(df):
    # Drop rows with missing dates
    df = df.dropna(subset=['Date'])

    # Convert numeric columns and drop non-numeric or missing
    numeric_cols = ['Open', 'High', 'Low', 'Adj Close', 'Volume']
    for col in numeric_cols:
        df[col] = pd.to_numeric(df[col], errors='coerce')

    # Drop rows with any missing numeric values
    df = df.dropna(subset=numeric_cols)

    return df

ioc_df = clean_dataframe(ioc_df)
ntpc_df = clean_dataframe(ntpc_df)
hindalco_df = clean_dataframe(hindalco_df)

print(f"IOC rows after cleaning: {len(ioc_df)}")
print(f"NTPC rows after cleaning: {len(ntpc_df)}")
print(f"HINDALCO rows after cleaning: {len(hindalco_df)}")

# ---------- Align Datasets by Common Dates ----------
ioc_dates = set(ioc_df['Date'])
ntpc_dates = set(ntpc_df['Date'])
hindalco_dates = set(hindalco_df['Date'])

common_dates = ioc_dates & ntpc_dates & hindalco_dates
common_dates = sorted(list(common_dates))

print(f"\nCommon dates across all three datasets: {len(common_dates)}")

# Filter to common dates
ioc_df = ioc_df[ioc_df['Date'].isin(common_dates)].sort_values('Date').reset_index(drop=True)
ntpc_df = ntpc_df[ntpc_df['Date'].isin(common_dates)].sort_values('Date').reset_index(drop=True)
hindalco_df = hindalco_df[hindalco_df['Date'].isin(common_dates)].sort_values('Date').reset_index(drop=True)

# ---------- Calculate Daily Returns (Log Difference of Adj Close) ----------
def calculate_returns(df):
    df = df.copy()
    df['Returns'] = np.log(df['Adj Close'] / df['Adj Close'].shift(1))
    return df

ioc_df = calculate_returns(ioc_df)
ntpc_df = calculate_returns(ntpc_df)
hindalco_df = calculate_returns(hindalco_df)

# ---------- Exclude Days with Zero Volume ----------
ioc_df = ioc_df[ioc_df['Volume'] != 0].reset_index(drop=True)
ntpc_df = ntpc_df[ntpc_df['Volume'] != 0].reset_index(drop=True)
hindalco_df = hindalco_df[hindalco_df['Volume'] != 0].reset_index(drop=True)

# Re-align after volume filtering
ioc_dates_nz = set(ioc_df['Date'])
ntpc_dates_nz = set(ntpc_df['Date'])
hindalco_dates_nz = set(hindalco_df['Date'])

common_dates_nz = ioc_dates_nz & ntpc_dates_nz & hindalco_dates_nz
common_dates_nz = sorted(list(common_dates_nz))

ioc_df = ioc_df[ioc_df['Date'].isin(common_dates_nz)].sort_values('Date').reset_index(drop=True)
ntpc_df = ntpc_df[ntpc_df['Date'].isin(common_dates_nz)].sort_values('Date').reset_index(drop=True)
hindalco_df = hindalco_df[hindalco_df['Date'].isin(common_dates_nz)].sort_values('Date').reset_index(drop=True)

print(f"\nAfter volume filtering - Common dates: {len(common_dates_nz)}")

# Recalculate returns after filtering
ioc_df = calculate_returns(ioc_df)
ntpc_df = calculate_returns(ntpc_df)
hindalco_df = calculate_returns(hindalco_df)

# Drop first row (NaN return)
ioc_df = ioc_df.dropna(subset=['Returns']).reset_index(drop=True)
ntpc_df = ntpc_df.dropna(subset=['Returns']).reset_index(drop=True)
hindalco_df = hindalco_df.dropna(subset=['Returns']).reset_index(drop=True)

print(f"\nFinal dataset sizes:")
print(f"IOC: {len(ioc_df)}")
print(f"NTPC: {len(ntpc_df)}")
print(f"HINDALCO: {len(hindalco_df)}")

# ==========================================
# ANALYSIS 1: ARIMA(1,1,1) on HINDALCO
# ==========================================
print("\n" + "="*50)
print("ARIMA(1,1,1) Forecasting on HINDALCO")
print("="*50)

# Fit ARIMA(1,1,1) on Adj Close
hindalco_adj_close = hindalco_df['Adj Close'].values

model = ARIMA(hindalco_adj_close, order=(1, 1, 1))
fitted_model = model.fit()

# Generate 30-day forecast
forecast_result = fitted_model.forecast(steps=30)
forecast_mean = np.mean(forecast_result)

print(f"30-day forecast mean: {forecast_mean:.2f}")

# ==========================================
# ANALYSIS 2: Two-Sample T-Test (IOC vs NTPC)
# ==========================================
print("\n" + "="*50)
print("Two-Sample T-Test: IOC vs NTPC Returns")
print("="*50)

ioc_returns = ioc_df['Returns'].values
ntpc_returns = ntpc_df['Returns'].values

t_stat, p_value = stats.ttest_ind(ioc_returns, ntpc_returns)

print(f"t-statistic: {t_stat:.3f}")
print(f"p-value: {p_value:.3f}")

# ==========================================
# ANALYSIS 3: 30-Day Rolling Volatility
# ==========================================
print("\n" + "="*50)
print("30-Day Rolling Volatility")
print("="*50)

window = 30

ioc_rolling_vol = ioc_df['Returns'].rolling(window=window).std()
ntpc_rolling_vol = ntpc_df['Returns'].rolling(window=window).std()
hindalco_rolling_vol = hindalco_df['Returns'].rolling(window=window).std()

# Calculate mean rolling volatility (excluding NaN values)
ioc_mean_vol = ioc_rolling_vol.mean()
ntpc_mean_vol = ntpc_rolling_vol.mean()
hindalco_mean_vol = hindalco_rolling_vol.mean()

print(f"IOC mean rolling volatility: {ioc_mean_vol:.4f}")
print(f"NTPC mean rolling volatility: {ntpc_mean_vol:.4f}")
print(f"HINDALCO mean rolling volatility: {hindalco_mean_vol:.4f}")

# ==========================================
# ANALYSIS 4: Granger Causality Test (IOC → HINDALCO)
# ==========================================
print("\n" + "="*50)
print("Granger Causality Test: IOC → HINDALCO")
print("="*50)

# Prepare data for Granger causality
granger_data = pd.DataFrame({
    'IOC_Returns': ioc_df['Returns'].values,
    'HINDALCO_Returns': hindalco_df['Returns'].values
})

# Granger causality test (maxlag=1)
# Test if IOC returns cause HINDALCO returns
gc_result = grangercausalitytests(granger_data[['HINDALCO_Returns', 'IOC_Returns']], maxlag=1, verbose=False)

# Extract F-statistic and p-value for lag 1
f_stat = gc_result[1][0]['ssr_ftest'][0]
p_val = gc_result[1][0]['ssr_ftest'][1]

print(f"F-statistic: {f_stat:.3f}")
print(f"p-value: {p_val:.3f}")

# ==========================================
# ANALYSIS 5: Rolling Correlation Visualization
# ==========================================
print("\n" + "="*50)
print("30-Day Rolling Correlation: IOC vs NTPC")
print("="*50)

# Calculate 30-day rolling correlation
rolling_corr = ioc_df['Returns'].rolling(window=window).corr(ntpc_df['Returns'])

# Remove NaN values
rolling_corr_clean = rolling_corr.dropna()
dates_for_corr = ioc_df['Date'][rolling_corr.notna()]

# Find minimum correlation
min_corr = rolling_corr_clean.min()
print(f"Minimum rolling correlation: {min_corr:.3f}")

# Create visualization
plt.figure(figsize=(12, 6))
plt.plot(dates_for_corr, rolling_corr_clean, linewidth=1.5, color='steelblue')
plt.xlabel('Date', fontsize=12)
plt.ylabel('30-Day Rolling Correlation', fontsize=12)
plt.title('Rolling Correlation: IOC vs NTPC Daily Returns', fontsize=14, fontweight='bold')
plt.grid(True, alpha=0.3)
plt.tight_layout()
plt.savefig('Rolling_Correlation_Line_Chart.png', dpi=300, bbox_inches='tight')
print("Visualization saved: Rolling_Correlation_Line_Chart.png")

# ==========================================
# ANALYSIS 6: Identify Lowest Volatility Company
# ==========================================
print("\n" + "="*50)
print("Lowest Volatility Company")
print("="*50)

volatilities = {
    'IOC': ioc_mean_vol,
    'NTPC': ntpc_mean_vol,
    'HINDALCO': hindalco_mean_vol
}

lowest_vol_company = min(volatilities, key=volatilities.get)
print(f"Company with lowest average rolling volatility: {lowest_vol_company}")

# ==========================================
# FINAL OUTPUT SUMMARY
# ==========================================
print("\n" + "="*50)
print("KEY OUTPUTS")
print("="*50)
print(f"1. ARIMA Forecast Mean (30-day): {forecast_mean:.2f}")
print(f"2. Two-Sample T-Test t-statistic: {t_stat:.3f}")
print(f"3. Two-Sample T-Test p-value: {p_value:.3f}")
print(f"4. IOC Mean Rolling Volatility: {ioc_mean_vol:.4f}")
print(f"5. NTPC Mean Rolling Volatility: {ntpc_mean_vol:.4f}")
print(f"6. HINDALCO Mean Rolling Volatility: {hindalco_mean_vol:.4f}")
print(f"7. Granger Causality F-statistic: {f_stat:.3f}")
print(f"8. Granger Causality p-value: {p_val:.3f}")
print(f"9. Minimum Rolling Correlation: {min_corr:.3f}")
print(f"10. Lowest Volatility Company: {lowest_vol_company}")
print("="*50)
