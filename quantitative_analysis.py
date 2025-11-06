"""
Quantitative Analysis of Tech Stock Returns
Analyzing volatility persistence, directional predictability, and nonlinear co-movement
"""

import pandas as pd
import numpy as np
import matplotlib.pyplot as plt
from scipy import stats
from scipy.stats import norm
import warnings
warnings.filterwarnings('ignore')

# For GARCH modeling
from arch import arch_model

# For ARIMA and Granger causality
from statsmodels.tsa.arima.model import ARIMA
from statsmodels.tsa.stattools import grangercausalitytests

# For copula modeling
from scipy.stats import rankdata

print("=" * 80)
print("TECH STOCK QUANTITATIVE ANALYSIS")
print("=" * 80)

# ================================================================================
# STEP 1: Load Data and Compute Log Returns
# ================================================================================
print("\n[1] Loading datasets and computing log returns...")

def load_and_compute_returns(filepath, company_name):
    """Load CSV and compute daily log returns from Close prices"""
    df = pd.read_csv(filepath)
    df['Date'] = pd.to_datetime(df['Date'])
    df = df.sort_values('Date').reset_index(drop=True)

    # Exclude rows where Close is missing or zero
    df = df[df['Close'].notna() & (df['Close'] > 0)].copy()

    # Compute log returns: ln(P_t / P_{t-1})
    # Ensure both consecutive Close values are valid (already filtered above)
    df['log_return'] = np.log(df['Close'] / df['Close'].shift(1))

    # Drop the first row (NaN from shift)
    df = df.dropna(subset=['log_return']).reset_index(drop=True)

    print(f"  {company_name}: {len(df)} daily returns computed")
    return df

google_df = load_and_compute_returns('Google.csv', 'Google')
microsoft_df = load_and_compute_returns('Microsoft.csv', 'Microsoft')
amazon_df = load_and_compute_returns('Amazon.csv', 'Amazon')

# Extract return series
google_returns = google_df['log_return'].values
microsoft_returns = microsoft_df['log_return'].values
amazon_returns = amazon_df['log_return'].values

# ================================================================================
# STEP 2: GARCH(1,1) Model for Google Returns
# ================================================================================
print("\n[2] Fitting GARCH(1,1) model to Google returns...")

# Scale returns by 100 for numerical stability in GARCH estimation
google_returns_scaled = google_returns * 100

# Fit GARCH(1,1) model
garch_model = arch_model(google_returns_scaled, vol='Garch', p=1, q=1, rescale=False)
garch_fit = garch_model.fit(disp='off')

# Extract alpha (ARCH term) and beta (GARCH term)
alpha = garch_fit.params['alpha[1]']
beta = garch_fit.params['beta[1]']

print(f"  GARCH(1,1) coefficients:")
print(f"    Alpha (ARCH term): {alpha:.4f}")
print(f"    Beta (GARCH term): {beta:.4f}")

# ================================================================================
# STEP 3: Granger Causality Test (Microsoft → Amazon)
# ================================================================================
print("\n[3] Performing Granger causality test (Microsoft vs Amazon, 2 lags)...")

# Align the two series by date
merged = pd.merge(
    microsoft_df[['Date', 'log_return']].rename(columns={'log_return': 'msft'}),
    amazon_df[['Date', 'log_return']].rename(columns={'log_return': 'amzn'}),
    on='Date'
)

# Granger causality test: does Microsoft Granger-cause Amazon?
granger_result = grangercausalitytests(merged[['amzn', 'msft']], maxlag=2, verbose=False)

# Extract F-statistic and p-value for lag 2
granger_lag2 = granger_result[2][0]['ssr_ftest']
f_statistic = granger_lag2[0]
p_value = granger_lag2[1]

print(f"  Granger causality (Microsoft → Amazon, lag=2):")
print(f"    F-statistic: {f_statistic:.4f}")
print(f"    p-value: {p_value:.4f}")

# ================================================================================
# STEP 4: Rolling 90-Day Correlation (Google vs Amazon)
# ================================================================================
print("\n[4] Computing 90-day rolling correlation (Google vs Amazon)...")

# Merge Google and Amazon returns by date
ga_merged = pd.merge(
    google_df[['Date', 'log_return']].rename(columns={'log_return': 'goog'}),
    amazon_df[['Date', 'log_return']].rename(columns={'log_return': 'amzn'}),
    on='Date'
)

# Calculate rolling correlation with 90-day window
ga_merged['rolling_corr'] = ga_merged['goog'].rolling(window=90).corr(ga_merged['amzn'])

# Mean rolling correlation (excluding NaN values from initial window)
mean_rolling_corr = ga_merged['rolling_corr'].dropna().mean()

print(f"  Mean rolling correlation (90-day window): {mean_rolling_corr:.4f}")

# ================================================================================
# STEP 5: ARIMA(1,1,1) Models and One-Step-Ahead Forecast RMSE
# ================================================================================
print("\n[5] Fitting ARIMA(1,1,1) models and calculating forecast RMSE...")

def fit_arima_and_calculate_rmse(returns, company_name):
    """Fit ARIMA(1,1,1) and compute one-step-ahead forecast RMSE"""
    # Fit ARIMA(1,1,1)
    model = ARIMA(returns, order=(1, 1, 1))
    fit = model.fit()

    # One-step-ahead forecasts (in-sample)
    forecasts = fit.fittedvalues

    # For differenced models, fittedvalues are for the differenced series
    # We need to compare with actual differenced values
    # Since order d=1, the first observation is lost in differencing
    actual_diff = np.diff(returns)

    # The fitted values correspond to the differenced series
    # Match the lengths
    if len(forecasts) == len(actual_diff):
        forecast_values = forecasts
        actual_values = actual_diff
    else:
        # Align by taking the appropriate subset
        min_len = min(len(forecasts), len(actual_diff))
        forecast_values = forecasts[-min_len:]
        actual_values = actual_diff[-min_len:]

    # Calculate RMSE
    rmse = np.sqrt(np.mean((actual_values - forecast_values) ** 2))

    print(f"  {company_name} ARIMA(1,1,1) RMSE: {rmse:.3f}")

    # Return fitted model and standardized residuals for copula analysis
    return fit, fit.resid / np.std(fit.resid)

google_arima_fit, google_std_resid = fit_arima_and_calculate_rmse(google_returns, 'Google')
microsoft_arima_fit, microsoft_std_resid = fit_arima_and_calculate_rmse(microsoft_returns, 'Microsoft')
amazon_arima_fit, amazon_std_resid = fit_arima_and_calculate_rmse(amazon_returns, 'Amazon')

# ================================================================================
# STEP 6: Gaussian Copula Model Using Standardized Residuals
# ================================================================================
print("\n[6] Fitting Gaussian copula model to ARIMA standardized residuals...")

# Align the standardized residuals by length (take minimum common length)
min_len = min(len(google_std_resid), len(microsoft_std_resid), len(amazon_std_resid))
google_std_resid = google_std_resid[-min_len:]
microsoft_std_resid = microsoft_std_resid[-min_len:]
amazon_std_resid = amazon_std_resid[-min_len:]

# Remove any NaN or infinite values
valid_idx = np.isfinite(google_std_resid) & np.isfinite(microsoft_std_resid) & np.isfinite(amazon_std_resid)
google_std_resid = google_std_resid[valid_idx]
microsoft_std_resid = microsoft_std_resid[valid_idx]
amazon_std_resid = amazon_std_resid[valid_idx]

# Use rank transformation to convert to uniform margins (more robust)
# This is the empirical CDF approach
n = len(google_std_resid)
u_google = rankdata(google_std_resid) / (n + 1)
u_microsoft = rankdata(microsoft_std_resid) / (n + 1)
u_amazon = rankdata(amazon_std_resid) / (n + 1)

# Clip to avoid extreme quantiles
u_google = np.clip(u_google, 1e-6, 1 - 1e-6)
u_microsoft = np.clip(u_microsoft, 1e-6, 1 - 1e-6)
u_amazon = np.clip(u_amazon, 1e-6, 1 - 1e-6)

# Estimate Gaussian copula correlation matrix
# Convert to normal quantiles and compute correlation
z_google = norm.ppf(u_google)
z_microsoft = norm.ppf(u_microsoft)
z_amazon = norm.ppf(u_amazon)

# Stack into matrix
Z = np.column_stack([z_google, z_microsoft, z_amazon])

# Compute correlation matrix (this is the copula parameter matrix)
copula_corr_matrix = np.corrcoef(Z.T)

# The copula correlation parameter (average of off-diagonal correlations)
off_diagonal = []
for i in range(3):
    for j in range(i+1, 3):
        off_diagonal.append(copula_corr_matrix[i, j])

copula_corr_param = np.mean(off_diagonal)

print(f"  Gaussian copula correlation parameter (mean): {copula_corr_param:.4f}")
print(f"  Full copula correlation matrix:")
print(f"    {copula_corr_matrix}")

# ================================================================================
# STEP 7: Upper-Tail Dependence Coefficient (90th Percentile)
# ================================================================================
print("\n[7] Calculating upper-tail dependence coefficient...")

# Calculate 90th percentiles for each standardized residual series
q90_google = np.percentile(google_std_resid, 90)
q90_microsoft = np.percentile(microsoft_std_resid, 90)
q90_amazon = np.percentile(amazon_std_resid, 90)

# Joint exceedance: all three exceed their 90th percentiles
joint_exceedance = np.sum(
    (google_std_resid > q90_google) &
    (microsoft_std_resid > q90_microsoft) &
    (amazon_std_resid > q90_amazon)
)

# Total observations
n_obs = len(google_std_resid)

# Empirical upper-tail dependence coefficient
# Standard formula: lambda_U = P(all exceed 90th | at least one exceeds 90th)
# Simplified for equal thresholds: P(all > q90) / P(margin > q90)
# where P(margin > q90) = 0.1 by definition
prob_joint = joint_exceedance / n_obs

# Upper-tail dependence coefficient
# lambda_U = lim_{u→1} P(U2>u, U3>u | U1>u)
# Empirical estimate at 90th percentile:
upper_tail_dep = prob_joint / 0.1  # Normalized by marginal probability

print(f"  Joint exceedance count: {joint_exceedance} out of {n_obs}")
print(f"  Empirical joint exceedance probability: {prob_joint:.6f}")
print(f"  Empirical upper-tail dependence coefficient: {upper_tail_dep:.4f}")

# ================================================================================
# STEP 8: Hexbin Plot (Google vs Amazon Returns) and Pearson Correlation
# ================================================================================
print("\n[8] Generating hexbin plot and calculating Pearson correlation...")

# Merge Google and Amazon returns for plotting
plot_df = pd.merge(
    google_df[['Date', 'log_return']].rename(columns={'log_return': 'google'}),
    amazon_df[['Date', 'log_return']].rename(columns={'log_return': 'amazon'}),
    on='Date'
)

# Calculate Pearson correlation
pearson_corr = plot_df['google'].corr(plot_df['amazon'])

print(f"  Pearson correlation (Google vs Amazon): {pearson_corr:.4f}")

# Create hexbin plot
plt.figure(figsize=(10, 8))
plt.hexbin(plot_df['google'], plot_df['amazon'], gridsize=30, cmap='Blues', mincnt=1)
plt.colorbar(label='Count')
plt.xlabel('Google Daily Returns', fontsize=12)
plt.ylabel('Amazon Daily Returns', fontsize=12)
plt.title(f'Hexbin Plot: Google vs Amazon Daily Returns\nPearson Correlation: {pearson_corr:.4f}',
          fontsize=14, fontweight='bold')
plt.grid(True, alpha=0.3)
plt.tight_layout()
plt.savefig('hexbin_google_amazon.png', dpi=300, bbox_inches='tight')
print(f"  Hexbin plot saved as 'hexbin_google_amazon.png'")

# ================================================================================
# STEP 9: Compare Copula Dependency vs Pearson Correlation
# ================================================================================
print("\n[9] Comparing copula dependency vs Pearson correlation...")

abs_copula = abs(copula_corr_param)
abs_pearson = abs(pearson_corr)

print(f"  |Copula correlation parameter|: {abs_copula:.4f}")
print(f"  |Pearson correlation|: {abs_pearson:.4f}")

if abs_copula > abs_pearson:
    comparison_result = "YES"
    print(f"  Result: YES - |Copula| > |Pearson|")
else:
    comparison_result = "NO"
    print(f"  Result: NO - |Copula| ≤ |Pearson|")

# ================================================================================
# SUMMARY REPORT
# ================================================================================
print("\n" + "=" * 80)
print("SUMMARY OF RESULTS")
print("=" * 80)

print(f"\n1. GARCH(1,1) Coefficients for Google:")
print(f"   Alpha (ARCH term): {alpha:.4f}")
print(f"   Beta (GARCH term): {beta:.4f}")

print(f"\n2. Granger Causality Test (Microsoft → Amazon, lag=2):")
print(f"   F-statistic: {f_statistic:.4f}")
print(f"   p-value: {p_value:.4f}")

print(f"\n3. Mean Rolling Correlation (90-day, Google vs Amazon):")
print(f"   Mean: {mean_rolling_corr:.4f}")

print(f"\n4. ARIMA(1,1,1) One-Step-Ahead Forecast RMSE:")
print(f"   Google: {np.sqrt(np.mean((np.diff(google_returns) - google_arima_fit.fittedvalues[-len(np.diff(google_returns)):]) ** 2)):.3f}")
print(f"   Microsoft: {np.sqrt(np.mean((np.diff(microsoft_returns) - microsoft_arima_fit.fittedvalues[-len(np.diff(microsoft_returns)):]) ** 2)):.3f}")
print(f"   Amazon: {np.sqrt(np.mean((np.diff(amazon_returns) - amazon_arima_fit.fittedvalues[-len(np.diff(amazon_returns)):]) ** 2)):.3f}")

print(f"\n5. Gaussian Copula Model:")
print(f"   Copula correlation parameter: {copula_corr_param:.4f}")
print(f"   Upper-tail dependence coefficient: {upper_tail_dep:.4f}")

print(f"\n6. Pearson Correlation (Google vs Amazon):")
print(f"   Correlation: {pearson_corr:.4f}")

print(f"\n7. Comparison:")
print(f"   Does |Copula| > |Pearson|? {comparison_result}")

print("\n" + "=" * 80)
print("ANALYSIS COMPLETE")
print("=" * 80)
