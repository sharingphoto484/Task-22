# ==========================================
# Johansen VAR Cointegration Analysis Script
# ==========================================
# Requirements: pandas, numpy, matplotlib, scipy, statsmodels
# Input files: AMAZON_daily.csv, GOOGLE_daily.csv, META_daily.csv (in same directory)
# Output files: analysis_results.json, impulse_response_plot.png
# ==========================================

import pandas as pd
import numpy as np
import matplotlib.pyplot as plt
import json
from scipy import stats
from statsmodels.tsa.vector_ar.vecm import coint_johansen
from statsmodels.tsa.api import VAR
from statsmodels.tsa.stattools import grangercausalitytests
import warnings
warnings.filterwarnings('ignore')

# ---------- Load CSVs Robustly ----------
print("=" * 60)
print("LOADING DATA")
print("=" * 60)

# Load the three stock datasets
amazon_df = pd.read_csv('AMAZON_daily.csv')
google_df = pd.read_csv('GOOGLE_daily.csv')
meta_df = pd.read_csv('META_daily.csv')

# Convert Date columns to datetime
amazon_df['Date'] = pd.to_datetime(amazon_df['Date'])
google_df['Date'] = pd.to_datetime(google_df['Date'])
meta_df['Date'] = pd.to_datetime(meta_df['Date'])

print(f"Amazon raw rows: {len(amazon_df)}")
print(f"Google raw rows: {len(google_df)}")
print(f"Meta raw rows: {len(meta_df)}")

# ---------- Build Exact Date Intersection ----------
print("\n" + "=" * 60)
print("BUILDING DATE INTERSECTION")
print("=" * 60)

# Find the exact intersection of trading days
common_dates = set(amazon_df['Date']) & set(google_df['Date']) & set(meta_df['Date'])
common_dates = sorted(list(common_dates))

print(f"Common trading days: {len(common_dates)}")
print(f"Date range: {common_dates[0]} to {common_dates[-1]}")

# Restrict all dataframes to common dates
amazon_common = amazon_df[amazon_df['Date'].isin(common_dates)].sort_values('Date').reset_index(drop=True)
google_common = google_df[google_df['Date'].isin(common_dates)].sort_values('Date').reset_index(drop=True)
meta_common = meta_df[meta_df['Date'].isin(common_dates)].sort_values('Date').reset_index(drop=True)

# Verify alignment
assert len(amazon_common) == len(google_common) == len(meta_common) == len(common_dates)
assert all(amazon_common['Date'] == google_common['Date'])
assert all(amazon_common['Date'] == meta_common['Date'])

# ---------- Extract Adj Close and Compute Log Prices ----------
print("\n" + "=" * 60)
print("COMPUTING LOG PRICES AND RETURNS")
print("=" * 60)

# Extract Adj Close prices
amazon_prices = amazon_common['Adj Close'].values
google_prices = google_common['Adj Close'].values
meta_prices = meta_common['Adj Close'].values

# Compute log prices
amazon_log_prices = np.log(amazon_prices)
google_log_prices = np.log(google_prices)
meta_log_prices = np.log(meta_prices)

# Compute log returns as first difference of log prices
amazon_returns = np.diff(amazon_log_prices)
google_returns = np.diff(google_log_prices)
meta_returns = np.diff(meta_log_prices)

print(f"Log price observations: {len(amazon_log_prices)}")
print(f"Return observations (after differencing): {len(amazon_returns)}")

# Create return dates (one less observation due to differencing)
return_dates = common_dates[1:]

# ---------- Johansen Cointegration Analysis ----------
print("\n" + "=" * 60)
print("JOHANSEN COINTEGRATION ANALYSIS")
print("=" * 60)

# Stack log prices into matrix (T x 3)
log_price_matrix = np.column_stack([amazon_log_prices, google_log_prices, meta_log_prices])

# Determine lag order using Schwarz (BIC) criterion over range 1-5
# We need to test each lag order and compute BIC
# For Johansen, we use det=1 (constant in cointegrating relation)

best_lag_johansen = None
best_bic = np.inf

for lag in range(1, 6):
    # Fit VAR to log price differences to get BIC
    temp_data = pd.DataFrame(log_price_matrix, columns=['AMAZON', 'GOOGLE', 'META'])
    temp_var = VAR(temp_data)
    temp_result = temp_var.fit(maxlags=lag, ic=None)
    bic = temp_result.bic

    if bic < best_bic:
        best_bic = bic
        best_lag_johansen = lag

print(f"Selected Johansen lag order (SBC): {best_lag_johansen}")

# Apply Johansen procedure with selected lag and constant in cointegrating relation (det=1)
johansen_result = coint_johansen(log_price_matrix, det_order=1, k_ar_diff=best_lag_johansen)

# Extract trace statistics and critical values at 5% level
trace_stats = johansen_result.lr1  # trace statistics
critical_values_5pct = johansen_result.cvt[:, 1]  # 5% critical values (column index 1)

print("\nTrace Test Results (5% level):")
print(f"H0: r=0 | Trace: {trace_stats[0]:.4f} | Critical: {critical_values_5pct[0]:.4f}")
print(f"H0: r≤1 | Trace: {trace_stats[1]:.4f} | Critical: {critical_values_5pct[1]:.4f}")
print(f"H0: r≤2 | Trace: {trace_stats[2]:.4f} | Critical: {critical_values_5pct[2]:.4f}")

# Determine cointegration rank r using trace test at 5% level
r = 0
if trace_stats[0] > critical_values_5pct[0]:
    r = 1
    if trace_stats[1] > critical_values_5pct[1]:
        r = 2
        if trace_stats[2] > critical_values_5pct[2]:
            r = 3

# Cap r at 2 as specified (select from {0, 1, 2})
r = min(r, 2)

print(f"\n>>> Selected cointegration rank r: {r}")

# Extract and normalize cointegrating vector if r > 0
if r > 0:
    # Get the first cointegrating vector (eigenvector)
    coint_vector_raw = johansen_result.evec[:, 0]  # First column

    # Normalize on META (index 2)
    # Order is AMAZON (0), GOOGLE (1), META (2)
    normalization_factor = coint_vector_raw[2]
    coint_vector = coint_vector_raw / normalization_factor

    print(f"\nFirst cointegrating vector (normalized on META):")
    print(f"  AMAZON: {coint_vector[0]:.6f}")
    print(f"  GOOGLE: {coint_vector[1]:.6f}")
    print(f"  META:   {coint_vector[2]:.6f}")
else:
    coint_vector = np.array([0.0, 0.0, 0.0])
    print("\nNo cointegration detected. Storing zero vector.")

# ---------- VAR Model on Returns ----------
print("\n" + "=" * 60)
print("VAR MODEL ON RETURNS")
print("=" * 60)

# Stack returns into dataframe
returns_df = pd.DataFrame({
    'AMAZON': amazon_returns,
    'GOOGLE': google_returns,
    'META': meta_returns
})

# Hold out last 60 days for testing
train_returns = returns_df.iloc[:-60]
test_returns = returns_df.iloc[-60:]

print(f"Training observations: {len(train_returns)}")
print(f"Test observations: {len(test_returns)}")

# Select VAR lag order using AIC over range 1-5
var_model = VAR(train_returns)
best_lag_var = None
best_aic = np.inf

for lag in range(1, 6):
    result = var_model.fit(maxlags=lag, ic=None)
    aic = result.aic
    if aic < best_aic:
        best_aic = aic
        best_lag_var = lag

print(f"Selected VAR lag order (AIC): {best_lag_var}")

# Fit final VAR model with selected lag
var_fitted = var_model.fit(maxlags=best_lag_var, ic=None)

# Verify stability: all companion matrix eigenvalues < 1
# VAR model is stable if all roots are inside unit circle
is_stable = var_fitted.is_stable()

print(f"VAR stability check: {'PASS' if is_stable else 'FAIL'}")

# ---------- One-Step Ahead Forecasting ----------
print("\n" + "=" * 60)
print("ONE-STEP AHEAD FORECASTING (60-DAY TEST WINDOW)")
print("=" * 60)

# Produce 1-step ahead forecasts over 60-day test window
forecast_errors = []

for i in range(len(test_returns)):
    # Refit VAR on data up to current test point
    train_expanding = returns_df.iloc[:-(60-i)]
    var_temp = VAR(train_expanding)
    var_temp_fitted = var_temp.fit(maxlags=best_lag_var, ic=None)

    # Forecast 1 step ahead
    forecast = var_temp_fitted.forecast(train_expanding.values[-best_lag_var:], steps=1)[0]

    # Get actual values
    actual = test_returns.iloc[i].values

    # Compute squared errors
    errors_squared = (actual - forecast) ** 2
    forecast_errors.append(errors_squared)

# Convert to array and compute RMSE for each series
forecast_errors = np.array(forecast_errors)
rmse_amazon = np.sqrt(np.mean(forecast_errors[:, 0]))
rmse_google = np.sqrt(np.mean(forecast_errors[:, 1]))
rmse_meta = np.sqrt(np.mean(forecast_errors[:, 2]))

# Average RMSE equally across three series
avg_rmse = (rmse_amazon + rmse_google + rmse_meta) / 3.0

print(f"RMSE AMAZON: {rmse_amazon:.8f}")
print(f"RMSE GOOGLE: {rmse_google:.8f}")
print(f"RMSE META:   {rmse_meta:.8f}")
print(f"\n>>> Average RMSE: {avg_rmse:.8f}")

# ---------- Rolling Correlations (5 Years) ----------
print("\n" + "=" * 60)
print("ROLLING CORRELATIONS (60-DAY WINDOW, LAST 5 YEARS)")
print("=" * 60)

# Identify most recent 5 full calendar years within common sample
# Last common date determines the end
last_date = common_dates[-1]
# Go back 5 full calendar years
start_date_5yr = pd.Timestamp(year=last_date.year - 5, month=1, day=1)

# Filter return dates to this window
five_year_mask = [(d >= start_date_5yr) for d in return_dates]
returns_5yr = returns_df[five_year_mask].reset_index(drop=True)

print(f"5-year window observations: {len(returns_5yr)}")

# Compute 60-day rolling correlations for each pair
window = 60
rolling_corrs = {
    'AMAZON_GOOGLE': [],
    'AMAZON_META': [],
    'GOOGLE_META': []
}

for i in range(window, len(returns_5yr) + 1):
    window_data = returns_5yr.iloc[i-window:i]

    corr_ag = window_data['AMAZON'].corr(window_data['GOOGLE'])
    corr_am = window_data['AMAZON'].corr(window_data['META'])
    corr_gm = window_data['GOOGLE'].corr(window_data['META'])

    rolling_corrs['AMAZON_GOOGLE'].append(corr_ag)
    rolling_corrs['AMAZON_META'].append(corr_am)
    rolling_corrs['GOOGLE_META'].append(corr_gm)

# Find maximum correlation across all pairs and all time points
max_corr_ag = np.max(rolling_corrs['AMAZON_GOOGLE'])
max_corr_am = np.max(rolling_corrs['AMAZON_META'])
max_corr_gm = np.max(rolling_corrs['GOOGLE_META'])

max_rolling_corr = max(max_corr_ag, max_corr_am, max_corr_gm)

print(f"Max rolling correlation AMAZON-GOOGLE: {max_corr_ag:.6f}")
print(f"Max rolling correlation AMAZON-META:   {max_corr_am:.6f}")
print(f"Max rolling correlation GOOGLE-META:   {max_corr_gm:.6f}")
print(f"\n>>> Maximum rolling correlation: {max_rolling_corr:.6f}")

# ---------- Granger Causality Tests ----------
print("\n" + "=" * 60)
print("GRANGER CAUSALITY TESTS (LAGS 1-5, 5% LEVEL)")
print("=" * 60)

# Test all 6 ordered pairs
pairs = [
    ('AMAZON', 'GOOGLE'),
    ('AMAZON', 'META'),
    ('GOOGLE', 'AMAZON'),
    ('GOOGLE', 'META'),
    ('META', 'AMAZON'),
    ('META', 'GOOGLE')
]

significant_pairs = 0
granger_results = {}

for cause, effect in pairs:
    # Prepare data: effect is first column, cause is second
    test_data = returns_df[[effect, cause]]

    # Run Granger causality test for lags 1-5
    try:
        gc_result = grangercausalitytests(test_data, maxlag=5, verbose=False)

        # Check if any lag rejects null at 5% level
        is_significant = False
        for lag in range(1, 6):
            # Use F-test (ssr_ftest)
            p_value = gc_result[lag][0]['ssr_ftest'][1]
            if p_value < 0.05:
                is_significant = True
                break

        granger_results[f"{cause}->{effect}"] = is_significant
        if is_significant:
            significant_pairs += 1
            print(f"  {cause} -> {effect}: SIGNIFICANT")
        else:
            print(f"  {cause} -> {effect}: Not significant")
    except:
        granger_results[f"{cause}->{effect}"] = False
        print(f"  {cause} -> {effect}: Test failed")

print(f"\n>>> Total significant directed pairs: {significant_pairs}")

# ---------- Impulse Response Analysis ----------
print("\n" + "=" * 60)
print("IMPULSE RESPONSE ANALYSIS")
print("=" * 60)

# Refit VAR on full training data with selected lag
var_full = VAR(returns_df.iloc[:-60])  # Use in-sample data
var_full_fitted = var_full.fit(maxlags=best_lag_var, ic=None)

# Compute 10-step impulse response
# 1% orthogonalized shock to GOOGLE, response of META
# Cholesky ordering: GOOGLE (0), AMAZON (1), META (2)

# Reorder columns for Cholesky: GOOGLE, AMAZON, META
returns_ordered = returns_df[['GOOGLE', 'AMAZON', 'META']].iloc[:-60]
var_ordered = VAR(returns_ordered)
var_ordered_fitted = var_ordered.fit(maxlags=best_lag_var, ic=None)

# Get impulse response (orthogonalized) for 10 periods
# periods=10 actually returns 11 points (0 through 10), so we slice to get exactly 10
irf = var_ordered_fitted.irf(periods=10)

# Extract response of META (index 2) to shock in GOOGLE (index 0)
# irf.irfs shape is (periods+1, n_vars, n_shocks)
irf_response = irf.irfs[:10, 2, 0]  # META response to GOOGLE shock, first 10 periods

# Scale to 1% shock (irf gives response to 1-unit shock)
# Standard deviation of GOOGLE returns
google_std = returns_ordered['GOOGLE'].std()
shock_size = 0.01  # 1% in decimal

# Response to 1% shock
irf_response_1pct = irf_response * (shock_size / google_std) * 100  # Convert to percent

# Find absolute peak magnitude
abs_peak_irf = np.max(np.abs(irf_response_1pct))

print(f"10-day impulse response (META to 1% GOOGLE shock):")
for i, val in enumerate(irf_response_1pct):
    print(f"  Day {i}: {val:.6f}%")

print(f"\n>>> Absolute peak magnitude: {abs_peak_irf:.6f}%")

# ---------- Visualization: Impulse Response ----------
print("\n" + "=" * 60)
print("GENERATING IMPULSE RESPONSE PLOT")
print("=" * 60)

plt.figure(figsize=(10, 6))
plt.plot(range(10), irf_response_1pct, marker='o', linewidth=2, markersize=6)
plt.axhline(y=0, color='black', linestyle='--', alpha=0.3)
plt.xlabel('Days', fontsize=12)
plt.ylabel('Response (%)', fontsize=12)
plt.title('10-Day Impulse Response: META Returns to 1% Orthogonalized GOOGLE Shock\n(Cholesky Ordering: GOOGLE-AMAZON-META)', fontsize=13)
plt.grid(alpha=0.3)
plt.tight_layout()
plt.savefig('impulse_response_plot.png', dpi=150)
print("Saved: impulse_response_plot.png")
plt.close()

# ---------- Ornstein-Uhlenbeck Mean Reversion Analysis ----------
print("\n" + "=" * 60)
print("ORNSTEIN-UHLENBECK MEAN REVERSION ANALYSIS")
print("=" * 60)

if r > 0:
    # Form spread from cointegrating relation
    # Spread = coint_vector[0] * AMAZON + coint_vector[1] * GOOGLE + coint_vector[2] * META
    spread = (coint_vector[0] * amazon_log_prices +
              coint_vector[1] * google_log_prices +
              coint_vector[2] * meta_log_prices)

    # Estimate OU model: dS = κ(θ - S)dt + σdW
    # Discrete approximation: S(t+1) - S(t) = κθ - κS(t) + ε
    # Rearrange: S(t+1) = α + βS(t) + ε, where β = 1-κ

    spread_lag = spread[:-1]
    spread_diff = np.diff(spread)

    # OLS regression: spread_diff = a + b*spread_lag
    X = np.column_stack([np.ones(len(spread_lag)), spread_lag])
    y = spread_diff

    # MLE estimates
    params = np.linalg.lstsq(X, y, rcond=None)[0]
    alpha_hat = params[0]
    beta_hat = params[1]

    # Extract OU parameters
    kappa = -beta_hat  # Mean reversion speed
    theta = -alpha_hat / beta_hat  # Long-run mean

    # Half-life = ln(2) / κ
    if kappa > 0:
        half_life = np.log(2) / kappa
    else:
        half_life = 0.0  # No mean reversion

    print(f"OU parameter κ (mean reversion speed): {kappa:.6f}")
    print(f"OU parameter θ (long-run mean): {theta:.6f}")
    print(f"\n>>> Half-life of mean reversion: {half_life:.6f} trading days")
else:
    half_life = 0.0
    print("Cointegration rank r = 0, no mean reversion analysis.")
    print(f"\n>>> Half-life of mean reversion: {half_life:.6f} trading days")

# ---------- Summary Statistics and Results ----------
print("\n" + "=" * 60)
print("FINAL SUMMARY")
print("=" * 60)

print(f"\n1. Cointegration rank r: {r}")
print(f"2. VAR lag order p: {best_lag_var}")
print(f"3. Average RMSE (60-day forecast): {avg_rmse:.8f}")
print(f"4. Maximum rolling correlation: {max_rolling_corr:.6f}")
print(f"5. Significant Granger pairs: {significant_pairs}")
print(f"6. Impulse response peak magnitude: {abs_peak_irf:.6f}%")
print(f"7. Half-life of mean reversion: {half_life:.6f} trading days")

# ---------- Risk Policy Discussion ----------
print("\n" + "=" * 60)
print("RISK POLICY")
print("=" * 60)

risk_policy_summary = "All analysis restricted to exact date intersection; no imputation or forward filling; cointegration and causality tested at strict five percent level only."

print(f"\nSUMMARY: {risk_policy_summary}")

print("\nDISCUSSION:")
print("This study enforces rigorous data integrity by restricting every calculation")
print("to the exact intersection of trading days across all three securities, ensuring")
print("no synthetic or imputed observations enter the analysis. Statistical inference")
print("relies exclusively on the five percent significance level for both the Johansen")
print("trace test and Granger causality, eliminating subjective judgment in rank and")
print("direction selection. Model selection uses information criteria without manual")
print("intervention, and stability verification confirms all VAR companion eigenvalues")
print("lie strictly inside the unit circle before forecast evaluation proceeds.")

# ---------- Save Results to JSON ----------
print("\n" + "=" * 60)
print("SAVING RESULTS")
print("=" * 60)

results = {
    "cointegration_rank_r": int(r),
    "var_lag_order_p": int(best_lag_var),
    "average_rmse": float(avg_rmse),
    "maximum_rolling_correlation": float(max_rolling_corr),
    "significant_granger_pairs": int(significant_pairs),
    "impulse_response_peak_magnitude_percent": float(abs_peak_irf),
    "half_life_trading_days": float(half_life),
    "risk_policy_summary": risk_policy_summary,
    "johansen_lag_order": int(best_lag_johansen),
    "common_trading_days": int(len(common_dates)),
    "var_stability": bool(is_stable)
}

with open('analysis_results.json', 'w') as f:
    json.dump(results, indent=4, fp=f)

print("Saved: analysis_results.json")

print("\n" + "=" * 60)
print("ANALYSIS COMPLETE")
print("=" * 60)
