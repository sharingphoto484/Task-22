# ==========================================
# Integrated Equity Market Analysis Script
# Requirements: pandas, numpy, matplotlib, scipy, statsmodels
# Input files: IOC.csv, NTPC.csv, HINDALCO.csv (in same directory)
# Output files: return_volatility_surface.png
# ==========================================

import pandas as pd
import numpy as np
import matplotlib.pyplot as plt
from statsmodels.tsa.vector_ar.vecm import coint_johansen
from statsmodels.tsa.api import VAR
from statsmodels.tsa.stattools import grangercausalitytests
import warnings
warnings.filterwarnings('ignore')

# ---------- Load CSVs Robustly ----------
print("=" * 60)
print("EQUITY MARKET DYNAMICS ANALYSIS")
print("=" * 60)

# Load the three equity datasets
ioc_df = pd.read_csv('IOC.csv')
ntpc_df = pd.read_csv('NTPC.csv')
hindalco_df = pd.read_csv('HINDALCO.csv')

print("\n[1] Loading datasets...")
print(f"    IOC: {len(ioc_df)} rows")
print(f"    NTPC: {len(ntpc_df)} rows")
print(f"    HINDALCO: {len(hindalco_df)} rows")

# ---------- Extract Relevant Columns and Clean Data ----------
def extract_and_clean(df, stock_name):
    """Extract Date, Open, High, Low, Close, Volume and clean data"""
    # Use Close as Adj Close if Adj Close doesn't exist
    if 'Adj Close' in df.columns:
        adj_close_col = 'Adj Close'
    else:
        adj_close_col = 'Close'

    # Select relevant columns
    df_clean = df[['Date', 'Open', 'High', 'Low', 'Close', 'Volume']].copy()
    df_clean['Adj Close'] = df[adj_close_col]

    # Convert Date to datetime
    df_clean['Date'] = pd.to_datetime(df_clean['Date'], errors='coerce')

    # Convert numeric columns to numeric, coercing errors to NaN
    numeric_cols = ['Open', 'High', 'Low', 'Close', 'Adj Close', 'Volume']
    for col in numeric_cols:
        df_clean[col] = pd.to_numeric(df_clean[col], errors='coerce')

    # Remove rows with any missing values
    df_clean = df_clean.dropna()

    # Rename columns to include stock name
    df_clean = df_clean.rename(columns={
        'Open': f'{stock_name}_Open',
        'High': f'{stock_name}_High',
        'Low': f'{stock_name}_Low',
        'Close': f'{stock_name}_Close',
        'Adj Close': f'{stock_name}_Adj_Close',
        'Volume': f'{stock_name}_Volume'
    })

    return df_clean

ioc_clean = extract_and_clean(ioc_df, 'IOC')
ntpc_clean = extract_and_clean(ntpc_df, 'NTPC')
hindalco_clean = extract_and_clean(hindalco_df, 'HINDALCO')

print("\n[2] Data cleaned and numeric values validated")

# ---------- Align Dates Across All Three Datasets ----------
print("\n[3] Aligning dates across all datasets...")

# Merge on common dates only
merged = ioc_clean.merge(ntpc_clean, on='Date', how='inner')
merged = merged.merge(hindalco_clean, on='Date', how='inner')

# Sort by date
merged = merged.sort_values('Date').reset_index(drop=True)

print(f"    Aligned dataset: {len(merged)} rows")
print(f"    Date range: {merged['Date'].min()} to {merged['Date'].max()}")

# ---------- Calculate Daily Log Returns ----------
print("\n[4] Calculating daily log returns...")

# Create a copy for returns calculation that excludes zero volume dates
def calculate_returns(df, stock_name):
    """Calculate log returns excluding zero volume dates"""
    # Filter out zero volume dates for this stock
    df_filtered = df[df[f'{stock_name}_Volume'] > 0].copy()

    # Calculate log returns
    df_filtered[f'{stock_name}_Return'] = np.log(df_filtered[f'{stock_name}_Adj_Close']) - \
                                           np.log(df_filtered[f'{stock_name}_Adj_Close'].shift(1))

    return df_filtered

# For returns, we need to handle each stock's zero volume separately
# First, create returns for all without filtering
merged['IOC_Return'] = np.log(merged['IOC_Adj_Close']) - np.log(merged['IOC_Adj_Close'].shift(1))
merged['NTPC_Return'] = np.log(merged['NTPC_Adj_Close']) - np.log(merged['NTPC_Adj_Close'].shift(1))
merged['HINDALCO_Return'] = np.log(merged['HINDALCO_Adj_Close']) - np.log(merged['HINDALCO_Adj_Close'].shift(1))

# For return-based analysis, we'll filter appropriately
# Create a returns dataset that respects zero volume exclusions
returns_df = merged.copy()
returns_df.loc[returns_df['IOC_Volume'] == 0, 'IOC_Return'] = np.nan
returns_df.loc[returns_df['NTPC_Volume'] == 0, 'NTPC_Return'] = np.nan
returns_df.loc[returns_df['HINDALCO_Volume'] == 0, 'HINDALCO_Return'] = np.nan

# Remove first row (NaN due to differencing) and any rows with NaN returns
returns_df = returns_df.dropna(subset=['IOC_Return', 'NTPC_Return', 'HINDALCO_Return'])

print(f"    Returns calculated for {len(returns_df)} trading days")

# ---------- Johansen Cointegration Test ----------
print("\n" + "=" * 60)
print("JOHANSEN COINTEGRATION TEST")
print("=" * 60)

# Prepare Adj Close series for cointegration test (use full aligned data)
adj_close_matrix = merged[['IOC_Adj_Close', 'NTPC_Adj_Close', 'HINDALCO_Adj_Close']].values

# Determine lag order using Schwarz criterion (BIC) over lags 1-5
# Johansen test with different lag orders
print("\n[5] Selecting optimal lag length by Schwarz criterion...")

from statsmodels.tsa.api import VAR as VAR_for_lag

# Fit VAR to determine optimal lag
var_model_lag = VAR_for_lag(adj_close_matrix)
lag_order_results = []
for lag in range(1, 6):
    try:
        result = var_model_lag.fit(lag)
        lag_order_results.append({
            'lag': lag,
            'aic': result.aic,
            'bic': result.bic,
            'hqic': result.hqic
        })
    except:
        pass

lag_df = pd.DataFrame(lag_order_results)
optimal_lag_bic = lag_df.loc[lag_df['bic'].idxmin(), 'lag']
print(f"    Optimal lag by Schwarz (BIC): {int(optimal_lag_bic)}")

# Run Johansen test with optimal lag
# det_order: -1 = no deterministic terms, 0 = constant term, 1 = linear trend
# We want constant term in cointegration relation, so det_order=0
johansen_result = coint_johansen(adj_close_matrix, det_order=0, k_ar_diff=int(optimal_lag_bic))

# Get cointegration rank at 5% significance level
# Trace statistic tests r=0, r<=1, r<=2
trace_stats = johansen_result.lr1
critical_values = johansen_result.cvt[:, 1]  # 5% critical values

coint_rank = 0
for i in range(len(trace_stats)):
    if trace_stats[i] > critical_values[i]:
        coint_rank = i + 1

print(f"\n[6] Cointegration test results:")
print(f"    Selected lag length: {int(optimal_lag_bic)}")
print(f"    Cointegration rank (at 5% level): {coint_rank}")

# Get normalized cointegration vector (first eigenvector)
# The eigenvectors are in johansen_result.evec
# Normalized so that first element is 1
eigenvector = johansen_result.evec[:, 0]
normalized_vector = eigenvector / eigenvector[0]
ioc_coef = normalized_vector[0]

print(f"    First element of normalized cointegration vector (IOC): {ioc_coef:.4f}")

# ---------- VAR Model Estimation on Returns ----------
print("\n" + "=" * 60)
print("VAR MODEL ESTIMATION")
print("=" * 60)

# Prepare returns dataframe (keep column names)
returns_data = returns_df[['IOC_Return', 'NTPC_Return', 'HINDALCO_Return']].copy()

print("\n[7] Selecting optimal VAR lag length by AIC...")

# Fit VAR and select lag by AIC
var_model = VAR(returns_data)
lag_order_results_var = []
for lag in range(1, 6):
    try:
        result = var_model.fit(lag)
        lag_order_results_var.append({
            'lag': lag,
            'aic': result.aic,
            'bic': result.bic,
            'hqic': result.hqic
        })
    except:
        pass

lag_df_var = pd.DataFrame(lag_order_results_var)
optimal_lag_aic = lag_df_var.loc[lag_df_var['aic'].idxmin(), 'lag']
print(f"    Optimal lag by AIC: {int(optimal_lag_aic)}")

# Fit final VAR model
var_fitted = var_model.fit(int(optimal_lag_aic))
print(f"    VAR({int(optimal_lag_aic)}) model estimated")

# ---------- Forecast Error Variance Decomposition ----------
print("\n[8] Computing forecast error variance decomposition...")

# FEVD for HINDALCO returns
# fevd(periods) returns decomposition for periods steps ahead
fevd_result = var_fitted.fevd(10)  # 10 periods ahead

# One-step-ahead decomposition (index 0 is step 1)
# Order is [IOC, NTPC, HINDALCO]
# We want HINDALCO's variance (index 2) decomposed
hindalco_fevd = fevd_result.decomp[0, 2, :]  # step 1, HINDALCO, all shocks

ioc_contribution = hindalco_fevd[0]
ntpc_contribution = hindalco_fevd[1]
hindalco_contribution = hindalco_fevd[2]

print(f"\n    One-step-ahead FEVD for HINDALCO:")
print(f"      IOC contribution: {ioc_contribution:.3f}")
print(f"      NTPC contribution: {ntpc_contribution:.3f}")
print(f"      HINDALCO contribution: {hindalco_contribution:.3f}")

# ---------- Block Exogeneity Test ----------
print("\n[9] Testing block exogeneity (IOC and NTPC → HINDALCO)...")

# Test if IOC and NTPC jointly Granger-cause HINDALCO
# We need to test if lags of IOC and NTPC predict HINDALCO

# Create lagged variables for the test
# We use the VAR model's Granger causality test
test_result = var_fitted.test_causality(caused='HINDALCO_Return',
                                         causing=['IOC_Return', 'NTPC_Return'],
                                         kind='f')

chi2_stat = test_result.test_statistic
p_value = test_result.pvalue

print(f"    Chi-square test statistic: {chi2_stat:.3f}")
print(f"    P-value: {p_value:.4f}")

# ---------- Median Absolute Returns ----------
print("\n[10] Computing median absolute returns...")

ioc_median_abs = returns_df['IOC_Return'].abs().median()
ntpc_median_abs = returns_df['NTPC_Return'].abs().median()
hindalco_median_abs = returns_df['HINDALCO_Return'].abs().median()

avg_median_abs = (ioc_median_abs + ntpc_median_abs + hindalco_median_abs) / 3

print(f"\n     IOC median absolute return: {ioc_median_abs:.6f}")
print(f"     NTPC median absolute return: {ntpc_median_abs:.6f}")
print(f"     HINDALCO median absolute return: {hindalco_median_abs:.6f}")
print(f"     Average of three medians: {avg_median_abs:.4f}")

# ---------- Rolling Volatility Calculation ----------
print("\n[11] Computing 30-day rolling volatilities...")

# Calculate 30-day rolling standard deviation
window = 30
returns_df['IOC_Rolling_Vol'] = returns_df['IOC_Return'].rolling(window=window).std()
returns_df['NTPC_Rolling_Vol'] = returns_df['NTPC_Return'].rolling(window=window).std()

# Remove NaN values from rolling calculation
plot_df = returns_df.dropna(subset=['IOC_Rolling_Vol', 'NTPC_Rolling_Vol'])

max_ntpc_vol = plot_df['NTPC_Rolling_Vol'].max()
print(f"     Maximum NTPC rolling volatility: {max_ntpc_vol:.4f}")

# ---------- Return Volatility Surface Plot ----------
print("\n[12] Creating Return Volatility Surface Plot...")

fig, ax = plt.subplots(figsize=(14, 8))

# Create scatter plot with color intensity representing IOC volatility
scatter = ax.scatter(plot_df['Date'],
                     plot_df['NTPC_Rolling_Vol'],
                     c=plot_df['IOC_Rolling_Vol'],
                     cmap='plasma',
                     alpha=0.6,
                     s=20)

ax.set_xlabel('Date', fontsize=12, fontweight='bold')
ax.set_ylabel('30-Day Rolling Std Dev of NTPC Returns', fontsize=12, fontweight='bold')
ax.set_title('Return Volatility Surface Plot\nNTPC Volatility vs Date (Color: IOC Volatility)',
             fontsize=14, fontweight='bold', pad=20)

# Add colorbar
cbar = plt.colorbar(scatter, ax=ax)
cbar.set_label('30-Day Rolling Std Dev of IOC Returns', fontsize=11, fontweight='bold')

ax.grid(True, alpha=0.3)
plt.xticks(rotation=45)
plt.tight_layout()
plt.savefig('return_volatility_surface.png', dpi=300, bbox_inches='tight')
print("     Plot saved: return_volatility_surface.png")

# ---------- Identify Dominant Contributor to HINDALCO Variance ----------
print("\n[13] Identifying dominant contributor to HINDALCO variance...")

contributions = {
    'IOC': ioc_contribution,
    'NTPC': ntpc_contribution,
    'HINDALCO': hindalco_contribution
}

dominant_stock = max(contributions, key=contributions.get)
print(f"     Dominant contributor: {dominant_stock} ({contributions[dominant_stock]:.3f})")

# ==========================================
# KEY OUTPUTS SUMMARY
# ==========================================
print("\n" + "=" * 60)
print("KEY OUTPUTS")
print("=" * 60)

print(f"\n1. Johansen Cointegration Test:")
print(f"   - Selected cointegration rank: {coint_rank}")
print(f"   - First element of normalized cointegration vector (IOC): {ioc_coef:.4f}")

print(f"\n2. VAR Model & Variance Decomposition:")
print(f"   - Optimal VAR lag (by AIC): {int(optimal_lag_aic)}")
print(f"   - Proportion of HINDALCO variance from IOC innovations: {ioc_contribution:.3f}")

print(f"\n3. Block Exogeneity Test:")
print(f"   - Chi-square test statistic (IOC & NTPC → HINDALCO): {chi2_stat:.3f}")

print(f"\n4. Median Absolute Returns:")
print(f"   - Average of three medians: {avg_median_abs:.4f}")

print(f"\n5. Volatility Analysis:")
print(f"   - Maximum NTPC rolling volatility: {max_ntpc_vol:.4f}")

print(f"\n6. Variance Decomposition Leadership:")
print(f"   - Stock with highest contribution to HINDALCO variance: {dominant_stock}")

print("\n" + "=" * 60)
print("ANALYSIS COMPLETE")
print("=" * 60)
