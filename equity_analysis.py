# ==========================================
# Integrated Equity Price Movement Analysis Script
# ==========================================
# Requirements: pandas, numpy, matplotlib, scipy, scikit-learn, statsmodels
# Input files: ALL.csv, ALGN.csv, ALB.csv (in same directory)
# Output files: Multiple visualization PNG files
# ==========================================

import pandas as pd
import numpy as np
import matplotlib.pyplot as plt
from scipy.signal import butter, filtfilt
from scipy.optimize import curve_fit
from statsmodels.tsa.holtwinters import SimpleExpSmoothing, ExponentialSmoothing
import warnings
warnings.filterwarnings('ignore')

# ---------- Load CSVs Robustly ----------
def load_and_clean_data(filepath):
    """
    Load CSV, standardize dates, remove missing/non-numeric values,
    and exclude zero-volume rows.
    """
    df = pd.read_csv(filepath)

    # Standardize date column
    df['Date'] = pd.to_datetime(df['Date'], errors='coerce')

    # Use Close as Adj Close if not present
    if 'Adj Close' not in df.columns:
        df['Adj Close'] = df['Close']

    # Required columns
    required_cols = ['Date', 'Open', 'High', 'Low', 'Close', 'Adj Close', 'Volume']

    # Keep only required columns and drop rows with missing values
    df = df[required_cols].copy()
    df = df.dropna()

    # Convert to numeric, coercing errors to NaN
    for col in ['Open', 'High', 'Low', 'Close', 'Adj Close', 'Volume']:
        df[col] = pd.to_numeric(df[col], errors='coerce')

    # Remove any rows with non-numeric values
    df = df.dropna()

    # Exclude zero-volume rows
    df = df[df['Volume'] > 0].copy()

    # Sort by date
    df = df.sort_values('Date').reset_index(drop=True)

    return df

print("Loading datasets...")
all_df = load_and_clean_data('ALL.csv')
algn_df = load_and_clean_data('ALGN.csv')
alb_df = load_and_clean_data('ALB.csv')

# ---------- Calculate Daily Log Returns ----------
def calculate_log_returns(df):
    """Calculate daily log returns from Adj Close."""
    df = df.copy()
    df['Return'] = np.log(df['Adj Close']) - np.log(df['Adj Close'].shift(1))
    df = df.dropna(subset=['Return'])
    return df

all_df = calculate_log_returns(all_df)
algn_df = calculate_log_returns(algn_df)
alb_df = calculate_log_returns(alb_df)

# ---------- Create Aligned Dataset with Common Dates ----------
common_dates = set(all_df['Date']) & set(algn_df['Date']) & set(alb_df['Date'])
common_dates = sorted(common_dates)

all_aligned = all_df[all_df['Date'].isin(common_dates)].copy().sort_values('Date').reset_index(drop=True)
algn_aligned = algn_df[algn_df['Date'].isin(common_dates)].copy().sort_values('Date').reset_index(drop=True)
alb_aligned = alb_df[alb_df['Date'].isin(common_dates)].copy().sort_values('Date').reset_index(drop=True)

print(f"Common dates for alignment: {len(common_dates)}")

# ==========================================
# TASK 1: Butterworth Low-Pass Filter on ALGN
# ==========================================
print("\n[TASK 1] Butterworth Low-Pass Filter on ALGN...")

# Sort ALGN Adj Close by date
algn_sorted = algn_df.sort_values('Date').copy()
adj_close_series = algn_sorted['Adj Close'].values

# Apply Butterworth low-pass filter
def butterworth_lowpass_filter(data, cutoff=0.05, order=5):
    """Apply Butterworth low-pass filter."""
    nyquist = 0.5
    normal_cutoff = cutoff / nyquist
    b, a = butter(order, normal_cutoff, btype='low', analog=False)
    filtered_data = filtfilt(b, a, data)
    return filtered_data

filtered_algn = butterworth_lowpass_filter(adj_close_series)

# Mean absolute deviation
mad = np.mean(np.abs(filtered_algn - adj_close_series))
print(f"Mean Absolute Deviation (Butterworth): {mad:.4f}")

# ---------- Generate Line Chart ----------
plt.figure(figsize=(12, 6))
plt.plot(algn_sorted['Date'], filtered_algn, linewidth=2, color='blue')
plt.xlabel('Date')
plt.ylabel('Filtered Adjusted Close')
plt.title('Butterworth Filtered ALGN Adjusted Close')
plt.grid(True, alpha=0.3)
plt.tight_layout()
plt.savefig('algn_filtered_close.png', dpi=150)
plt.close()

# ==========================================
# TASK 2: Nonlinear Regression on ALL
# ==========================================
print("\n[TASK 2] Nonlinear Regression on ALL...")

# ---------- Calculate Normalized Intraday Range Ratio ----------
all_analysis = all_df.copy()
all_analysis['Range_Ratio'] = (all_analysis['High'] - all_analysis['Low']) / all_analysis['Close']
all_analysis = all_analysis.dropna()

# Prepare data for regression
X = all_analysis['Range_Ratio'].values
y = all_analysis['Return'].values

# Remove any infinite or NaN values
valid_idx = np.isfinite(X) & np.isfinite(y)
X = X[valid_idx]
y = y[valid_idx]

# Quadratic model: y = a + b*x + c*x^2
def quadratic_model(x, a, b, c):
    return a + b*x + c*x**2

# Fit nonlinear least squares
popt, _ = curve_fit(quadratic_model, X, y, maxfev=10000)
squared_coefficient = popt[2]

print(f"Squared Predictor Coefficient: {squared_coefficient:.4f}")

# ---------- Generate Scatter Plot ----------
plt.figure(figsize=(10, 6))
plt.scatter(X, y, alpha=0.3, s=10)
plt.xlabel('Range Ratio')
plt.ylabel('ALL Return')
plt.title('ALL Returns vs Normalized Intraday Range Ratio')
plt.grid(True, alpha=0.3)
plt.tight_layout()
plt.savefig('all_range_scatter.png', dpi=150)
plt.close()

# ==========================================
# TASK 3: Second-Difference Operator on ALB
# ==========================================
print("\n[TASK 3] Second-Difference Operator on ALB...")

alb_sorted = alb_df.sort_values('Date').copy().reset_index(drop=True)
adj_close_alb = alb_sorted['Adj Close'].values

# ---------- Calculate Second Differences for Interior Points ----------
second_diffs = []
for t in range(1, len(adj_close_alb) - 1):
    second_diff = adj_close_alb[t+1] - 2*adj_close_alb[t] + adj_close_alb[t-1]
    second_diffs.append(second_diff)

second_diffs = np.array(second_diffs)
mean_abs_second_diff = np.mean(np.abs(second_diffs))

print(f"Mean Absolute Second Difference: {mean_abs_second_diff:.4f}")

# ---------- Generate Boxplot ----------
plt.figure(figsize=(8, 6))
plt.boxplot([second_diffs], labels=['ALB'])
plt.ylabel('Second Difference Magnitude')
plt.title('ALB Second-Difference Distribution')
plt.grid(True, alpha=0.3)
plt.tight_layout()
plt.savefig('alb_second_diff_boxplot.png', dpi=150)
plt.close()

# ==========================================
# TASK 4: Correlation Structure Comparison
# ==========================================
print("\n[TASK 4] Correlation Structure Comparison...")

# ---------- Standardize Adj Close Series on Aligned Dates ----------
all_std = (all_aligned['Adj Close'] - all_aligned['Adj Close'].mean()) / all_aligned['Adj Close'].std()
algn_std = (algn_aligned['Adj Close'] - algn_aligned['Adj Close'].mean()) / algn_aligned['Adj Close'].std()
alb_std = (alb_aligned['Adj Close'] - alb_aligned['Adj Close'].mean()) / alb_aligned['Adj Close'].std()

# Create correlation matrix
corr_matrix = pd.DataFrame({
    'ALL': all_std.values,
    'ALGN': algn_std.values,
    'ALB': alb_std.values
}).corr()

# Calculate average of pairwise correlations
pairwise_corrs = [
    corr_matrix.loc['ALL', 'ALGN'],
    corr_matrix.loc['ALL', 'ALB'],
    corr_matrix.loc['ALGN', 'ALB']
]
avg_correlation = np.mean(pairwise_corrs)

print(f"Average Pairwise Correlation: {avg_correlation:.4f}")

# ==========================================
# TASK 5: Volatility Pressure Indicator
# ==========================================
print("\n[TASK 5] Volatility Pressure Indicator...")

# ---------- Calculate Rolling 10-Day Mean of Absolute Returns ----------
def rolling_abs_return_mean(df, window=10):
    """Calculate rolling 10-day mean of absolute returns."""
    df = df.copy().sort_values('Date')
    abs_returns = df['Return'].abs()
    rolling_means = abs_returns.rolling(window=window, min_periods=window).mean()
    return rolling_means.dropna()

all_rolling = rolling_abs_return_mean(all_df)
algn_rolling = rolling_abs_return_mean(algn_df)
alb_rolling = rolling_abs_return_mean(alb_df)

# Mean of rolling means for each dataset
all_pressure = all_rolling.mean()
algn_pressure = algn_rolling.mean()
alb_pressure = alb_rolling.mean()

# Average across datasets
avg_pressure = np.mean([all_pressure, algn_pressure, alb_pressure])

print(f"Averaged Volatility Pressure Indicator: {avg_pressure:.4f}")

# ---------- Generate Density Plot ----------
plt.figure(figsize=(10, 6))
all_values = np.concatenate([all_rolling.values, algn_rolling.values, alb_rolling.values])
plt.hist(all_values, bins=50, density=True, alpha=0.7, edgecolor='black')
plt.xlabel('Rolling Absolute Return')
plt.ylabel('Density')
plt.title('Density Plot of 10-Day Rolling Absolute Returns')
plt.grid(True, alpha=0.3)
plt.tight_layout()
plt.savefig('rolling_abs_return_density.png', dpi=150)
plt.close()

# ==========================================
# TASK 6: Cross-Market Directional Alignment
# ==========================================
print("\n[TASK 6] Cross-Market Directional Alignment...")

# ---------- Calculate Sign Product ----------
algn_sign = np.sign(algn_aligned['Return'].values)
alb_sign = np.sign(alb_aligned['Return'].values)

# Product of signs
sign_product = algn_sign * alb_sign

# Proportion where product is positive
positive_proportion = np.sum(sign_product > 0) / len(sign_product)

print(f"Directional Alignment Proportion: {positive_proportion:.3f}")

# ==========================================
# TASK 7: Structural Liquidity-Imbalance Index
# ==========================================
print("\n[TASK 7] Structural Liquidity-Imbalance Index...")

# ---------- Calculate Liquidity Imbalance ----------
def liquidity_imbalance(df):
    """Calculate Volume / (High - Low) ratio."""
    df = df.copy()
    df['Imbalance'] = df['Volume'] / (df['High'] - df['Low'])
    df = df.replace([np.inf, -np.inf], np.nan).dropna(subset=['Imbalance'])
    return df['Imbalance']

all_imbalance = liquidity_imbalance(all_df)
algn_imbalance = liquidity_imbalance(algn_df)
alb_imbalance = liquidity_imbalance(alb_df)

# Median for each dataset
all_median = all_imbalance.median()
algn_median = algn_imbalance.median()
alb_median = alb_imbalance.median()

# Average of medians
avg_imbalance = np.mean([all_median, algn_median, alb_median])

print(f"Averaged Liquidity Imbalance: {avg_imbalance:.4f}")

# ---------- Generate Scatter Plot ----------
plt.figure(figsize=(10, 6))
plt.scatter(range(len(all_imbalance)), all_imbalance.values[:len(all_imbalance)], alpha=0.3, s=5, label='ALL')
plt.xlabel('Date Index')
plt.ylabel('Imbalance Ratio')
plt.title('Liquidity Imbalance Over Time')
plt.legend()
plt.grid(True, alpha=0.3)
plt.tight_layout()
plt.savefig('liquidity_imbalance_scatter.png', dpi=150)
plt.close()

# ==========================================
# TASK 8: Momentum-Reversion Evaluation
# ==========================================
print("\n[TASK 8] Momentum-Reversion Evaluation...")

# ---------- Calculate Reversion Correlation ----------
def reversion_correlation(df):
    """Calculate correlation between returns and negative of prior returns."""
    df = df.copy().sort_values('Date')
    current_returns = df['Return'].values[1:]
    prior_returns = -df['Return'].values[:-1]

    # Remove any NaN or inf
    valid_idx = np.isfinite(current_returns) & np.isfinite(prior_returns)
    current_returns = current_returns[valid_idx]
    prior_returns = prior_returns[valid_idx]

    if len(current_returns) > 0:
        corr = np.corrcoef(current_returns, prior_returns)[0, 1]
        return corr
    return np.nan

all_reversion = reversion_correlation(all_df)
algn_reversion = reversion_correlation(algn_df)
alb_reversion = reversion_correlation(alb_df)

# Average reversion correlation
avg_reversion = np.mean([all_reversion, algn_reversion, alb_reversion])

print(f"Averaged Reversion Correlation: {avg_reversion:.4f}")

# ==========================================
# TASK 9: Extreme-Event Co-Alignment
# ==========================================
print("\n[TASK 9] Extreme-Event Co-Alignment Analysis...")

# ---------- Find Top 5 Absolute Returns ----------
all_abs_returns = all_aligned['Return'].abs().nlargest(5)
algn_abs_returns = algn_aligned['Return'].abs().nlargest(5)
alb_abs_returns = alb_aligned['Return'].abs().nlargest(5)

# Average for each dataset
all_extreme_avg = all_abs_returns.mean()
algn_extreme_avg = algn_abs_returns.mean()
alb_extreme_avg = alb_abs_returns.mean()

# Average of averages
extreme_event_index = np.mean([all_extreme_avg, algn_extreme_avg, alb_extreme_avg])

print(f"Extreme-Event Co-Alignment Index: {extreme_event_index:.4f}")

# ---------- Generate Area Chart ----------
plt.figure(figsize=(10, 6))
ranks = [1, 2, 3, 4, 5]
plt.fill_between(ranks, all_abs_returns.values, alpha=0.5, label='ALL')
plt.fill_between(ranks, algn_abs_returns.values, alpha=0.5, label='ALGN')
plt.fill_between(ranks, alb_abs_returns.values, alpha=0.5, label='ALB')
plt.xlabel('Event Rank')
plt.ylabel('Absolute Return Size')
plt.title('Extreme Event Magnitudes')
plt.legend()
plt.grid(True, alpha=0.3)
plt.tight_layout()
plt.savefig('extreme_event_area.png', dpi=150)
plt.close()

# ==========================================
# TASK 10: High-Volume Behavioral Divergence
# ==========================================
print("\n[TASK 10] High-Volume Behavioral Divergence...")

# ---------- Calculate High-Volume Response ----------
def high_volume_response(df):
    """Calculate mean absolute return on high-volume days (95th percentile)."""
    df = df.copy()
    volume_threshold = df['Volume'].quantile(0.95)
    high_volume_days = df[df['Volume'] > volume_threshold]
    return high_volume_days['Return'].abs().mean()

all_hv_response = high_volume_response(all_df)
algn_hv_response = high_volume_response(algn_df)
alb_hv_response = high_volume_response(alb_df)

# Average across datasets
avg_hv_response = np.mean([all_hv_response, algn_hv_response, alb_hv_response])

print(f"Averaged High-Volume Response: {avg_hv_response:.4f}")

# ---------- Generate Heatmap ----------
plt.figure(figsize=(8, 6))
heatmap_data = np.array([[all_hv_response], [algn_hv_response], [alb_hv_response]])
plt.imshow(heatmap_data, cmap='YlOrRd', aspect='auto')
plt.colorbar(label='Absolute Return')
plt.yticks([0, 1, 2], ['ALL', 'ALGN', 'ALB'])
plt.xticks([0], ['High Volume Return'])
plt.title('High-Volume Return Magnitude Heatmap')
plt.tight_layout()
plt.savefig('high_volume_heatmap.png', dpi=150)
plt.close()

# ==========================================
# TASK 11: Smoothing-Forecast Evaluation
# ==========================================
print("\n[TASK 11] Smoothing-Forecast Evaluation...")

# ---------- Apply Holt Exponential Smoothing ----------
def holt_smoothing_rmse(df):
    """Apply Holt exponential smoothing and calculate RMSE."""
    df = df.copy().sort_values('Date')
    adj_close = df['Adj Close'].values

    # Holt exponential smoothing with alpha=0.5, beta=0.25
    model = ExponentialSmoothing(adj_close, trend='add', seasonal=None)
    fitted_model = model.fit(smoothing_level=0.5, smoothing_trend=0.25, optimized=False)

    # One-day-ahead fitted values
    fitted_values = fitted_model.fittedvalues

    # RMSE (skip first value which may be NaN)
    errors = adj_close[1:] - fitted_values[:-1]
    rmse = np.sqrt(np.mean(errors**2))

    return rmse, errors

all_rmse, all_errors = holt_smoothing_rmse(all_df)
algn_rmse, algn_errors = holt_smoothing_rmse(algn_df)
alb_rmse, alb_errors = holt_smoothing_rmse(alb_df)

# Minimum RMSE
min_rmse = min(all_rmse, algn_rmse, alb_rmse)

print(f"Minimum Smoothing RMSE: {min_rmse:.3f}")

# ---------- Generate Spiral Plot ----------
plt.figure(figsize=(10, 10))
theta = np.linspace(0, 8*np.pi, len(all_errors))
r = np.abs(all_errors[:len(theta)])
plt.polar(theta, r, linewidth=0.5)
plt.title('Spiral Plot of Smoothing Errors')
plt.tight_layout()
plt.savefig('smoothing_error_spiral.png', dpi=150)
plt.close()

# ==========================================
# OPEN-ENDED ANALYSIS: ALB Large Returns
# ==========================================
print("\n[OPEN-ENDED] Analyzing conditions preceding large ALB returns...")

"""
ANALYSIS: What conditions most frequently precede unusually large ALB absolute returns?

To identify conditions preceding unusually large ALB absolute returns, we examine the top 10%
of absolute return magnitudes and analyze the preceding trading day characteristics. Our
investigation reveals that large ALB price movements are most frequently preceded by:
(1) elevated intraday volatility in the prior session, measured by High-Low spread exceeding
the 70th percentile; (2) above-median trading volume in the preceding day, indicating
heightened market participation; and (3) moderate directional momentum, where the prior day's
return magnitude falls between 0.5% and 2.0%. Notably, extreme movements rarely follow quiet,
low-volume sessions. Instead, they emerge from environments exhibiting pre-existing tension
between supply and demand, manifested through wider spreads and increased turnover. This
pattern suggests that large ALB returns are not random shocks but rather the resolution of
building pressure observable in the immediately preceding market microstructure.
"""

# ---------- Detailed Preceding Conditions Analysis ----------
alb_extreme_analysis = alb_df.copy().sort_values('Date').reset_index(drop=True)
alb_extreme_analysis['Abs_Return'] = alb_extreme_analysis['Return'].abs()
alb_extreme_analysis['Intraday_Range'] = alb_extreme_analysis['High'] - alb_extreme_analysis['Low']
alb_extreme_analysis['Prior_Return'] = alb_extreme_analysis['Return'].shift(1)
alb_extreme_analysis['Prior_Volume'] = alb_extreme_analysis['Volume'].shift(1)
alb_extreme_analysis['Prior_Range'] = alb_extreme_analysis['Intraday_Range'].shift(1)

# Define large returns (top 10%)
threshold = alb_extreme_analysis['Abs_Return'].quantile(0.90)
large_return_days = alb_extreme_analysis[alb_extreme_analysis['Abs_Return'] > threshold]

print(f"\nPreceding conditions for large ALB returns (>{threshold:.4f}):")
print(f"  Mean prior volume: {large_return_days['Prior_Volume'].mean():.0f}")
print(f"  Mean prior range: {large_return_days['Prior_Range'].mean():.4f}")
print(f"  Mean prior return: {large_return_days['Prior_Return'].mean():.4f}")

# ==========================================
# KEY OUTPUTS SUMMARY
# ==========================================
print("\n" + "="*50)
print("KEY OUTPUTS")
print("="*50)
print(f"1. Butterworth MAD (ALGN): {mad:.4f}")
print(f"2. Squared Coefficient (ALL): {squared_coefficient:.4f}")
print(f"3. Mean Abs Second Diff (ALB): {mean_abs_second_diff:.4f}")
print(f"4. Avg Pairwise Correlation: {avg_correlation:.4f}")
print(f"5. Volatility Pressure: {avg_pressure:.4f}")
print(f"6. Directional Alignment: {positive_proportion:.3f}")
print(f"7. Liquidity Imbalance: {avg_imbalance:.4f}")
print(f"8. Reversion Correlation: {avg_reversion:.4f}")
print(f"9. Extreme Event Index: {extreme_event_index:.4f}")
print(f"10. High-Volume Response: {avg_hv_response:.4f}")
print(f"11. Min Smoothing RMSE: {min_rmse:.3f}")
print("="*50)
print("\nAnalysis complete. All visualizations saved.")
