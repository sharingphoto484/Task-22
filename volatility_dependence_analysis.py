# ==========================================
# GSADF Volatility & Dependence Analysis Script
# ==========================================
# Requirements: pandas, numpy, matplotlib, seaborn, scipy, statsmodels
# Input files: TSLA.csv, TM.csv, VWAGY.csv (in same directory)
# Output files: analysis_results.json, tail_dependence_heatmap.png
# ==========================================

import pandas as pd
import numpy as np
import matplotlib.pyplot as plt
import seaborn as sns
from scipy import stats
from statsmodels.regression.quantile_regression import QuantReg
import json
import warnings
warnings.filterwarnings('ignore')

# ==========================================
# GSADF TEST IMPLEMENTATION
# ==========================================

def adf_test_statistic(y, lag=0):
    """
    Compute ADF test statistic for a series y.
    Optimized version with fixed lag for speed.
    """
    n = len(y)

    if n < 10:
        return -np.inf

    # Create lagged differences
    dy = np.diff(y)
    y_lag = y[:-1]

    # Simple ADF regression without lags for speed
    if lag == 0:
        X = y_lag
        y_dep = dy
    else:
        # Build regression matrix with lags
        X = [y_lag[lag:]]
        for i in range(1, lag + 1):
            if i < len(dy):
                X.append(dy[lag - i:-i] if lag - i > 0 else dy[:len(dy)-i])
        X = np.column_stack(X) if len(X) > 1 else X[0].reshape(-1, 1)
        y_dep = dy[lag:]

    if len(y_dep) < 3:
        return -np.inf

    # OLS regression
    try:
        X_with_const = np.column_stack([np.ones(len(y_dep)), X]) if X.ndim > 1 or lag > 0 else np.column_stack([np.ones(len(y_dep)), X.reshape(-1, 1)])
        beta = np.linalg.lstsq(X_with_const, y_dep, rcond=None)[0]
        residuals = y_dep - X_with_const @ beta
        se = np.sqrt(np.sum(residuals**2) / max(1, len(y_dep) - len(beta)))

        # Standard error of first coefficient (y_lag coefficient)
        XtX_inv = np.linalg.inv(X_with_const.T @ X_with_const + np.eye(X_with_const.shape[1]) * 1e-10)
        se_beta = se * np.sqrt(XtX_inv[1, 1])

        # t-statistic
        t_stat = beta[1] / (se_beta + 1e-10)
        return t_stat
    except:
        return -np.inf

def gsadf_test(series, r0, step=5):
    """
    Generalized Sup ADF test (right-tail).
    Optimized version with windowing step for speed.

    Parameters:
    -----------
    series : array-like
        Time series to test (typically log prices)
    r0 : float
        Minimum window size as fraction of sample
    step : int
        Step size for window iteration (larger = faster but less precise)

    Returns:
    --------
    gsadf_stat : float
        GSADF test statistic
    adf_sequence : array
        Sequence of ADF statistics for datestamping
    """
    T = len(series)
    min_window = int(np.floor(r0 * T))

    adf_stats = []

    # Iterate over windows with step size for efficiency
    for r2 in range(min_window, T + 1, step):
        window_stats = []
        for r1 in range(0, r2 - min_window + 1, step):
            subsample = series[r1:r2]
            if len(subsample) >= min_window:
                stat = adf_test_statistic(subsample, lag=0)
                window_stats.append(stat)

        if window_stats:
            adf_stats.append(max(window_stats))
        else:
            adf_stats.append(-np.inf)

    # Fill in between steps for full sequence
    if step > 1:
        full_stats = []
        for i in range(T - min_window + 1):
            idx = min(i // step, len(adf_stats) - 1)
            full_stats.append(adf_stats[idx])
        adf_stats = full_stats

    gsadf_stat = max(adf_stats) if adf_stats else -np.inf

    return gsadf_stat, np.array(adf_stats)

def gsadf_critical_values(T, r0, num_replications=2000, significance_level=0.05, step=5):
    """
    Obtain critical values via Monte Carlo simulation.

    Parameters:
    -----------
    T : int
        Sample size
    r0 : float
        Minimum window fraction
    num_replications : int
        Number of Monte Carlo replications
    significance_level : float
        Significance level (0.05 for 5%)
    step : int
        Step size for window iteration

    Returns:
    --------
    critical_value : float
        Critical value at specified significance level
    """
    gsadf_stats = []

    print(f"  Running {num_replications} Monte Carlo replications...")
    for i in range(num_replications):
        if (i + 1) % 200 == 0:
            print(f"    Completed {i + 1}/{num_replications} replications...")

        # Generate random walk under null
        innovations = np.random.randn(T)
        random_walk = np.cumsum(innovations)

        stat, _ = gsadf_test(random_walk, r0, step=step)
        gsadf_stats.append(stat)

    critical_value = np.percentile(gsadf_stats, (1 - significance_level) * 100)

    return critical_value

def bsadf_datestamp(series, r0, critical_value, step=1):
    """
    Backward SADF datestamping algorithm.

    Parameters:
    -----------
    series : array-like
        Time series
    r0 : float
        Minimum window fraction
    critical_value : float
        Critical value for comparison
    step : int
        Step size for efficiency

    Returns:
    --------
    explosive_periods : list of tuples
        List of (start_idx, end_idx) for explosive periods
    """
    T = len(series)
    min_window = int(np.floor(r0 * T))

    bsadf_sequence = []

    # Calculate BSADF sequence (finer step for datestamping)
    for r2 in range(min_window, T + 1, step):
        window_stats = []
        for r1 in range(0, r2 - min_window + 1, step):
            subsample = series[r1:r2]
            if len(subsample) >= min_window:
                stat = adf_test_statistic(subsample, lag=0)
                window_stats.append(stat)

        if window_stats:
            bsadf_sequence.append(max(window_stats))
        else:
            bsadf_sequence.append(-np.inf)

    # Identify explosive periods
    explosive_flags = np.array(bsadf_sequence) > critical_value

    # Merge contiguous periods
    episodes = []
    in_episode = False
    start_idx = None

    for i, flag in enumerate(explosive_flags):
        actual_idx = i * step
        if flag and not in_episode:
            start_idx = actual_idx + min_window - 1
            in_episode = True
        elif not flag and in_episode:
            end_idx = actual_idx + min_window - 2
            episodes.append((start_idx, end_idx))
            in_episode = False

    # Handle case where series ends in explosive period
    if in_episode:
        episodes.append((start_idx, len(series) - 1))

    return episodes

# ==========================================
# YANG-ZHANG VOLATILITY
# ==========================================

def yang_zhang_volatility(open_prices, high_prices, low_prices, close_prices, window=30, trading_days=252):
    """
    Calculate Yang-Zhang volatility estimator.

    Parameters:
    -----------
    open_prices, high_prices, low_prices, close_prices : array-like
        OHLC price data
    window : int
        Rolling window size (default 30)
    trading_days : int
        Trading days per year for annualization (default 252)

    Returns:
    --------
    yz_vol : array
        Rolling annualized Yang-Zhang volatility (in decimal form)
    """
    n = len(close_prices)
    yz_vol = np.full(n, np.nan)

    for i in range(window - 1, n):
        start = i - window + 1
        end = i + 1

        o = open_prices[start:end]
        h = high_prices[start:end]
        l = low_prices[start:end]
        c = close_prices[start:end]

        # Overnight volatility
        if start > 0:
            c_prev = np.roll(c, 1)
            c_prev[0] = close_prices[start - 1] if start > 0 else c[0]
            overnight = np.log(o / c_prev)
        else:
            overnight = np.log(o[1:] / c[:-1])

        # Open-to-close volatility
        open_close = np.log(c / o)

        # Rogers-Satchell volatility
        rs = np.log(h / c) * np.log(h / o) + np.log(l / c) * np.log(l / o)

        # Yang-Zhang estimator
        k = 0.34 / (1.34 + (window + 1) / (window - 1))

        sigma_o = np.var(overnight, ddof=1) if len(overnight) > 1 else 0
        sigma_c = np.var(open_close, ddof=1) if len(open_close) > 1 else 0
        sigma_rs = np.mean(rs) if len(rs) > 0 else 0

        # Daily variance
        daily_var = sigma_o + k * sigma_c + (1 - k) * sigma_rs

        # Annualize
        yz_vol[i] = np.sqrt(daily_var * trading_days)

    return yz_vol

# ==========================================
# TAIL DEPENDENCE
# ==========================================

def rolling_tail_dependence(returns1, returns2, window=252, quantile=0.05):
    """
    Calculate rolling lower tail dependence.

    Parameters:
    -----------
    returns1, returns2 : array-like
        Return series for two assets
    window : int
        Rolling window size
    quantile : float
        Quantile threshold (0.05 for 5%)

    Returns:
    --------
    tail_dep : array
        Rolling tail dependence measure
    """
    n = len(returns1)
    tail_dep = np.full(n, np.nan)

    for i in range(window - 1, n):
        start = i - window + 1
        end = i + 1

        r1 = returns1[start:end]
        r2 = returns2[start:end]

        # Calculate thresholds separately for each series
        threshold1 = np.quantile(r1, quantile)
        threshold2 = np.quantile(r2, quantile)

        # Count joint exceedances
        both_below = np.sum((r1 <= threshold1) & (r2 <= threshold2))

        # Tail dependence as fraction
        tail_dep[i] = both_below / window

    return tail_dep

# ==========================================
# AMIHUD ILLIQUIDITY
# ==========================================

def amihud_illiquidity(returns, close_prices, volumes):
    """
    Calculate daily Amihud illiquidity measure.

    Parameters:
    -----------
    returns : array-like
        Daily log returns
    close_prices : array-like
        Daily close prices
    volumes : array-like
        Daily volumes

    Returns:
    --------
    illiq : array
        Daily Amihud illiquidity
    """
    # Amihud = |return| / (Price * Volume)
    illiq = np.abs(returns) / (close_prices * volumes)

    # Handle any infinite or NaN values
    illiq = np.where(np.isfinite(illiq), illiq, np.nan)

    return illiq

# ==========================================
# CoVaR ANALYSIS
# ==========================================

def covar_analysis(returns_source, returns_target, quantile=0.05):
    """
    Estimate CoVaR using quantile regression.

    Parameters:
    -----------
    returns_source : array-like
        Returns of source asset (e.g., TSLA)
    returns_target : array-like
        Returns of target asset (e.g., TM or VWAGY)
    quantile : float
        VaR quantile level

    Returns:
    --------
    delta_covar : float
        Change in CoVaR (VaR at stress - VaR at median)
    p_value : float
        P-value for significance test
    """
    # Remove NaN values
    valid = ~(np.isnan(returns_source) | np.isnan(returns_target))
    rs = returns_source[valid]
    rt = returns_target[valid]

    if len(rs) < 10:
        return np.nan, np.nan

    # Quantile regression: Q_tau(r_target | r_source)
    # Model: r_target = alpha + beta * r_source

    # Fit quantile regression
    X = np.column_stack([np.ones(len(rs)), rs])
    model = QuantReg(rt, X)

    try:
        # Try with Newey-West covariance first
        try:
            result = model.fit(q=quantile, vcov='robust', kernel='epa', bandwidth='bofinger', max_iter=1000)
        except:
            # Fallback to standard fit
            result = model.fit(q=quantile, max_iter=1000)

        alpha = result.params[0]
        beta = result.params[1]

        # Calculate CoVaR at two states
        # State 1: Source at its own 5% quantile
        source_var = np.quantile(rs, quantile)
        covar_stress = alpha + beta * source_var

        # State 2: Source at median
        source_median = np.median(rs)
        covar_median = alpha + beta * source_median

        # Delta CoVaR
        delta_covar = covar_stress - covar_median

        # P-value for beta coefficient
        try:
            p_value = result.pvalues[1]
        except:
            # If p-value calculation fails, use a conservative estimate
            # based on whether the coefficient is large relative to its standard error
            try:
                t_stat = np.abs(beta / (result.bse[1] + 1e-10))
                p_value = 2 * (1 - stats.norm.cdf(t_stat))
            except:
                p_value = 0.5  # Conservative: assume not significant

        return delta_covar, p_value
    except Exception as e:
        print(f"  Error in quantile regression: {str(e)}")
        return np.nan, np.nan

# ---------- Load CSVs Robustly ----------

print("Loading CSV files...")
tsla = pd.read_csv('TSLA.csv')
tm = pd.read_csv('TM.csv')
vwagy = pd.read_csv('VWAGY.csv')

# Verify required columns
required_cols = ['Date', 'Open', 'High', 'Low', 'Close', 'Adj Close', 'Volume']
for df, name in [(tsla, 'TSLA'), (tm, 'TM'), (vwagy, 'VWAGY')]:
    assert all(col in df.columns for col in required_cols), f"{name} missing required columns"

# ---------- Date Intersection ----------

print("Computing exact daily intersection...")
# Convert dates to datetime
tsla['Date'] = pd.to_datetime(tsla['Date'])
tm['Date'] = pd.to_datetime(tm['Date'])
vwagy['Date'] = pd.to_datetime(vwagy['Date'])

# Find common dates
common_dates = set(tsla['Date']) & set(tm['Date']) & set(vwagy['Date'])
common_dates = sorted(common_dates)

print(f"Common dates: {len(common_dates)}")

# Filter to common dates
tsla = tsla[tsla['Date'].isin(common_dates)].sort_values('Date').reset_index(drop=True)
tm = tm[tm['Date'].isin(common_dates)].sort_values('Date').reset_index(drop=True)
vwagy = vwagy[vwagy['Date'].isin(common_dates)].sort_values('Date').reset_index(drop=True)

# Verify alignment
assert len(tsla) == len(tm) == len(vwagy), "Misaligned data after intersection"
assert all(tsla['Date'] == tm['Date']) and all(tm['Date'] == vwagy['Date']), "Date mismatch"

# ---------- Calculate Log Returns ----------

print("Calculating log returns...")
# Log prices
tsla['LogClose'] = np.log(tsla['Close'])
tm['LogClose'] = np.log(tm['Close'])
vwagy['LogClose'] = np.log(vwagy['Close'])

# Log returns (first difference)
tsla['LogReturn'] = tsla['LogClose'].diff()
tm['LogReturn'] = tm['LogClose'].diff()
vwagy['LogReturn'] = vwagy['LogClose'].diff()

# Remove first observation (lost to differencing)
tsla = tsla.iloc[1:].reset_index(drop=True)
tm = tm.iloc[1:].reset_index(drop=True)
vwagy = vwagy.iloc[1:].reset_index(drop=True)

print(f"Sample size after differencing: {len(tsla)}")

# ---------- GSADF Test ----------

print("\n" + "="*50)
print("GSADF Bubble Detection Analysis")
print("="*50)

# Set initial window fraction to 1% of sample
r0 = 0.01
T = len(tsla)
min_window = int(np.floor(r0 * T))

print(f"Sample size: {T}")
print(f"Minimum window (1%): {min_window}")
print(f"Running Monte Carlo simulations for critical values...")

# Get critical value with optimized step size
step_size = 10  # Larger step for faster computation
critical_value = gsadf_critical_values(T, r0, num_replications=2000, significance_level=0.05, step=step_size)
print(f"5% Critical value: {critical_value:.4f}")

# Run GSADF test for each ticker
bubble_episodes = {}

for ticker, df in [('TSLA', tsla), ('TM', tm), ('VWAGY', vwagy)]:
    print(f"\nTesting {ticker}...")

    log_prices = df['LogClose'].values
    gsadf_stat, _ = gsadf_test(log_prices, r0, step=step_size)

    print(f"  GSADF statistic: {gsadf_stat:.4f}")
    print(f"  Reject null: {gsadf_stat > critical_value}")

    # Datestamp explosive periods (use smaller step for more precision)
    episodes = bsadf_datestamp(log_prices, r0, critical_value, step=5)
    bubble_episodes[ticker] = episodes

    print(f"  Number of bubble episodes: {len(episodes)}")

    if episodes:
        for i, (start, end) in enumerate(episodes, 1):
            start_date = df.iloc[start]['Date']
            end_date = df.iloc[end]['Date']
            print(f"    Episode {i}: {start_date.strftime('%Y-%m-%d')} to {end_date.strftime('%Y-%m-%d')}")

# Total distinct bubble episodes
total_episodes = sum(len(episodes) for episodes in bubble_episodes.values())
print(f"\nTotal distinct bubble episodes across all tickers: {total_episodes}")

# ---------- Yang-Zhang Volatility ----------

print("\n" + "="*50)
print("Yang-Zhang Volatility Analysis")
print("="*50)

yz_medians = {}

for ticker, df in [('TSLA', tsla), ('TM', tm), ('VWAGY', vwagy)]:
    print(f"\nCalculating Yang-Zhang volatility for {ticker}...")

    yz_vol = yang_zhang_volatility(
        df['Open'].values,
        df['High'].values,
        df['Low'].values,
        df['Close'].values,
        window=30,
        trading_days=252
    )

    # Calculate median (excluding NaN values)
    median_vol = np.nanmedian(yz_vol)
    yz_medians[ticker] = median_vol

    print(f"  Median annualized Yang-Zhang volatility: {median_vol*100:.4f}%")

# Cross-asset average
cross_asset_avg_yz = np.mean(list(yz_medians.values()))
print(f"\nCross-asset average median Yang-Zhang volatility: {cross_asset_avg_yz*100:.4f}%")

# ---------- Rolling Tail Dependence ----------

print("\n" + "="*50)
print("Rolling Lower Tail Dependence Analysis")
print("="*50)

# Calculate for all three pairs
pairs = [
    ('TSLA', 'TM', tsla['LogReturn'].values, tm['LogReturn'].values),
    ('TSLA', 'VWAGY', tsla['LogReturn'].values, vwagy['LogReturn'].values),
    ('TM', 'VWAGY', tm['LogReturn'].values, vwagy['LogReturn'].values)
]

tail_dep_series = {}
tail_dep_averages = []

for name1, name2, returns1, returns2 in pairs:
    pair_name = f"{name1}-{name2}"
    print(f"\nCalculating tail dependence for {pair_name}...")

    tail_dep = rolling_tail_dependence(returns1, returns2, window=252, quantile=0.05)
    tail_dep_series[pair_name] = tail_dep

    # Average over full sample
    avg_tail_dep = np.nanmean(tail_dep)
    tail_dep_averages.append(avg_tail_dep)

    print(f"  Average lower tail dependence: {avg_tail_dep:.6f}")

# Average across all pairs
avg_lower_tail_dep = np.mean(tail_dep_averages)
print(f"\nAverage lower tail dependence across all pairs: {avg_lower_tail_dep:.6f}")

# ---------- Amihud Illiquidity ----------

print("\n" + "="*50)
print("Amihud Illiquidity Analysis")
print("="*50)

# Calculate daily Amihud for each ticker
amihud_monthly_std = {}

for ticker, df in [('TSLA', tsla), ('TM', tm), ('VWAGY', vwagy)]:
    print(f"\nCalculating Amihud illiquidity for {ticker}...")

    # Daily Amihud
    daily_amihud = amihud_illiquidity(
        df['LogReturn'].values,
        df['Close'].values,
        df['Volume'].values
    )

    # Add to dataframe
    df['Amihud'] = daily_amihud

    # Aggregate to monthly (calendar months)
    df['YearMonth'] = df['Date'].dt.to_period('M')
    monthly_amihud = df.groupby('YearMonth')['Amihud'].mean()

    # Standardize by in-sample mean and std
    mean_amihud = monthly_amihud.mean()
    std_amihud = monthly_amihud.std()

    monthly_amihud_std = (monthly_amihud - mean_amihud) / std_amihud

    # Get last 12 months
    last_12_months = monthly_amihud_std.iloc[-12:]
    avg_last_12 = last_12_months.mean()

    amihud_monthly_std[ticker] = avg_last_12

    print(f"  Mean monthly Amihud: {mean_amihud:.2e}")
    print(f"  Std monthly Amihud: {std_amihud:.2e}")
    print(f"  Last 12-month average (standardized): {avg_last_12:.4f}")

# Average across tickers
twelve_month_avg_std_amihud = np.mean(list(amihud_monthly_std.values()))
print(f"\nTwelve-month average standardized Amihud (cross-asset): {twelve_month_avg_std_amihud:.4f}")

# ---------- Tail Dependence Heatmap ----------

print("\n" + "="*50)
print("Creating Tail Dependence Heatmap")
print("="*50)

# Get last 24 month-ends
last_date = tsla['Date'].iloc[-1]
month_ends = []
current_date = last_date

for i in range(24):
    # Find month end
    month_end_mask = (tsla['Date'].dt.year == current_date.year) & \
                     (tsla['Date'].dt.month == current_date.month)

    if month_end_mask.any():
        month_end_idx = tsla[month_end_mask].index[-1]
        month_ends.append((tsla.iloc[month_end_idx]['Date'], month_end_idx))

    # Move to previous month
    current_date = (current_date - pd.DateOffset(months=1))

month_ends.reverse()
print(f"Found {len(month_ends)} month-end dates")

# Build heatmap data
heatmap_data = []
pair_labels = []

for name1, name2, returns1, returns2 in pairs:
    pair_label = f"{name1}-{name2}"
    pair_labels.append(pair_label)

    row_data = []

    for month_date, month_idx in month_ends:
        # Get tail dependence at this month end
        if month_idx < len(tail_dep_series[pair_label]):
            tail_val = tail_dep_series[pair_label][month_idx]
        else:
            tail_val = np.nan

        row_data.append(tail_val)

    heatmap_data.append(row_data)

# Convert to array
heatmap_array = np.array(heatmap_data)

# Find maximum value in heatmap
max_heatmap_value = np.nanmax(heatmap_array)
print(f"Maximum tail dependence in heatmap: {max_heatmap_value:.6f}")

# Create heatmap visualization
plt.figure(figsize=(16, 6))
sns.heatmap(
    heatmap_array,
    xticklabels=[d.strftime('%Y-%m') for d, _ in month_ends],
    yticklabels=pair_labels,
    cmap='YlOrRd',
    annot=True,
    fmt='.4f',
    cbar_kws={'label': 'Tail Dependence'}
)
plt.xlabel('Month End Date')
plt.ylabel('Stock Pair')
plt.title('Rolling Lower Tail Dependence (252-day window)\nLast 24 Months')
plt.xticks(rotation=45, ha='right')
plt.tight_layout()
plt.savefig('tail_dependence_heatmap.png', dpi=300, bbox_inches='tight')
print("Heatmap saved to tail_dependence_heatmap.png")
plt.close()

# ---------- CoVaR Risk Transmission ----------

print("\n" + "="*50)
print("CoVaR Risk Transmission Analysis")
print("="*50)

print("Assessing risk transmission from TSLA to TM and VWAGY...")

# CoVaR for TSLA -> TM
print("\nTSLA -> TM:")
delta_covar_tm, p_value_tm = covar_analysis(
    tsla['LogReturn'].values,
    tm['LogReturn'].values,
    quantile=0.05
)
print(f"  Delta CoVaR: {delta_covar_tm:.6f}")
print(f"  P-value: {p_value_tm:.4f}")
print(f"  Significant at 5%: {p_value_tm < 0.05}")
print(f"  Positive: {delta_covar_tm > 0}")

# CoVaR for TSLA -> VWAGY
print("\nTSLA -> VWAGY:")
delta_covar_vwagy, p_value_vwagy = covar_analysis(
    tsla['LogReturn'].values,
    vwagy['LogReturn'].values,
    quantile=0.05
)
print(f"  Delta CoVaR: {delta_covar_vwagy:.6f}")
print(f"  P-value: {p_value_vwagy:.4f}")
print(f"  Significant at 5%: {p_value_vwagy < 0.05}")
print(f"  Positive: {delta_covar_vwagy > 0}")

# Determine verdict
avg_delta_covar = (delta_covar_tm + delta_covar_vwagy) / 2
both_significant = (p_value_tm < 0.05) and (p_value_vwagy < 0.05)
avg_positive = avg_delta_covar > 0

verdict = 1 if (avg_positive and both_significant) else 0

print(f"\nAverage Delta CoVaR: {avg_delta_covar:.6f}")
print(f"Both significant at 5%: {both_significant}")
print(f"Average is positive: {avg_positive}")
print(f"Verdict: {verdict}")

# ---------- Final Summary ----------

print("\n" + "="*50)
print("FINAL RESULTS SUMMARY")
print("="*50)

results = {
    "total_bubble_episodes": int(total_episodes),
    "cross_asset_avg_median_yz_volatility_percent": float(cross_asset_avg_yz * 100),
    "average_lower_tail_dependence": float(avg_lower_tail_dep),
    "twelve_month_avg_standardized_amihud": float(twelve_month_avg_std_amihud),
    "max_heatmap_tail_dependence": float(max_heatmap_value),
    "covar_verdict": int(verdict)
}

print(f"\n1. Total distinct bubble episodes: {results['total_bubble_episodes']}")
print(f"2. Cross-asset average median Yang-Zhang volatility: {results['cross_asset_avg_median_yz_volatility_percent']:.4f}%")
print(f"3. Average lower tail dependence: {results['average_lower_tail_dependence']:.6f}")
print(f"4. Twelve-month average standardized Amihud: {results['twelve_month_avg_standardized_amihud']:.4f}")
print(f"5. Maximum heatmap tail dependence: {results['max_heatmap_tail_dependence']:.6f}")
print(f"6. CoVaR risk transmission verdict: {results['covar_verdict']}")

# Save results to JSON
with open('analysis_results.json', 'w') as f:
    json.dump(results, f, indent=2)

print("\nResults saved to analysis_results.json")
print("\nAnalysis complete!")
