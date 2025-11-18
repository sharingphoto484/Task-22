# ==========================================
# Integrated Bubble & Spillover Analysis Script
# ==========================================
# Requirements: pandas, numpy, matplotlib, scipy, statsmodels
# Input files: LT.csv, IOC.csv, ITC.csv (in same directory)
# Output files: analysis_results.txt, lower_tail_heatmap.png
#
# This script performs:
# 1. GSADF bubble detection (Phillips, Shi, Yu)
# 2. VAR estimation with Diebold-Yilmaz spillover indices
# 3. Yang-Zhang volatility estimation
# 4. Granger causality tests
# 5. Lower tail co-exceedance analysis
# 6. Diversification test
# ==========================================

import pandas as pd
import numpy as np
import matplotlib.pyplot as plt
import seaborn as sns
from scipy import stats
from scipy.stats import t as t_dist
from statsmodels.tsa.api import VAR
from statsmodels.tsa.stattools import grangercausalitytests
import warnings
warnings.filterwarnings('ignore')

# ---------- Load CSVs Robustly ----------
print("Loading data files...")

# Load the three CSV files
lt_df = pd.read_csv('LT.csv')
ioc_df = pd.read_csv('IOC.csv')
itc_df = pd.read_csv('ITC.csv')

# Convert Date to datetime
lt_df['Date'] = pd.to_datetime(lt_df['Date'])
ioc_df['Date'] = pd.to_datetime(ioc_df['Date'])
itc_df['Date'] = pd.to_datetime(itc_df['Date'])

# Extract required columns
lt_df = lt_df[['Date', 'Open', 'High', 'Low', 'Close', 'Volume']].copy()
ioc_df = ioc_df[['Date', 'Open', 'High', 'Low', 'Close', 'Volume']].copy()
itc_df = itc_df[['Date', 'Open', 'High', 'Low', 'Close', 'Volume']].copy()

print(f"LT: {len(lt_df)} rows, IOC: {len(ioc_df)} rows, ITC: {len(itc_df)} rows")

# ---------- Form Exact Daily Intersection ----------
print("\nForming exact daily intersection...")

# Find common dates across all three files
common_dates = set(lt_df['Date']).intersection(set(ioc_df['Date'])).intersection(set(itc_df['Date']))
common_dates = sorted(list(common_dates))

print(f"Common date range: {common_dates[0]} to {common_dates[-1]}")
print(f"Total common days: {len(common_dates)}")

# Filter each dataframe to common dates and sort
lt_df = lt_df[lt_df['Date'].isin(common_dates)].sort_values('Date').reset_index(drop=True)
ioc_df = ioc_df[ioc_df['Date'].isin(common_dates)].sort_values('Date').reset_index(drop=True)
itc_df = itc_df[itc_df['Date'].isin(common_dates)].sort_values('Date').reset_index(drop=True)

# Create combined dataframe
df = pd.DataFrame({
    'Date': lt_df['Date'],
    'LT_Open': lt_df['Open'],
    'LT_High': lt_df['High'],
    'LT_Low': lt_df['Low'],
    'LT_Close': lt_df['Close'],
    'LT_Volume': lt_df['Volume'],
    'IOC_Open': ioc_df['Open'],
    'IOC_High': ioc_df['High'],
    'IOC_Low': ioc_df['Low'],
    'IOC_Close': ioc_df['Close'],
    'IOC_Volume': ioc_df['Volume'],
    'ITC_Open': itc_df['Open'],
    'ITC_High': itc_df['High'],
    'ITC_Low': itc_df['Low'],
    'ITC_Close': itc_df['Close'],
    'ITC_Volume': itc_df['Volume']
})

# ---------- Calculate Log Prices and Log Returns ----------
print("\nCalculating log prices and returns...")

# Log prices
df['LT_LogClose'] = np.log(df['LT_Close'])
df['IOC_LogClose'] = np.log(df['IOC_Close'])
df['ITC_LogClose'] = np.log(df['ITC_Close'])

# Log returns (first difference of log prices)
df['LT_Return'] = df['LT_LogClose'].diff()
df['IOC_Return'] = df['IOC_LogClose'].diff()
df['ITC_Return'] = df['ITC_LogClose'].diff()

print(f"After differencing: {len(df)-1} return observations")

# ---------- GSADF Bubble Detection ----------
print("\n" + "="*60)
print("PERFORMING GSADF BUBBLE DETECTION")
print("="*60)

def gsadf_test(y, min_window_frac=0.01, num_simulations=2000):
    """
    Generalized Sup ADF test (Phillips, Shi, Yu) - Optimized version

    Parameters:
    - y: log price series (array)
    - min_window_frac: minimum window as fraction of sample (r0)
    - num_simulations: number of Monte Carlo replications

    Returns:
    - gsadf_stat: GSADF statistic
    - cv: critical values (90%, 95%, 99%)
    - date_stamp: boolean array of explosive periods
    """
    T = len(y)
    r0 = int(np.floor(T * min_window_frac))

    def compute_adf_statistic(y_data, r1, r2):
        """Compute ADF t-statistic for window y_data[r1:r2]"""
        window = y_data[r1:r2]
        n = len(window)
        if n < 3:
            return None

        y_window = window[1:]
        y_lag = window[:-1]

        # Fast OLS using normal equations
        n_obs = len(y_lag)
        sum_y = np.sum(y_lag)
        sum_yy = np.sum(y_lag * y_lag)
        sum_y_window = np.sum(y_window)
        sum_ylag_ywindow = np.sum(y_lag * y_window)

        # OLS coefficients
        denom = n_obs * sum_yy - sum_y * sum_y
        if abs(denom) < 1e-10:
            return None

        beta1 = (n_obs * sum_ylag_ywindow - sum_y * sum_y_window) / denom
        beta0 = (sum_y_window - beta1 * sum_y) / n_obs

        # Residuals and variance
        fitted = beta0 + beta1 * y_lag
        residuals = y_window - fitted
        sigma2 = np.sum(residuals**2) / (n_obs - 2)

        # Variance of beta1
        var_beta1 = sigma2 / (sum_yy - sum_y * sum_y / n_obs)

        if var_beta1 <= 0:
            return None

        t_stat = beta1 / np.sqrt(var_beta1)
        return t_stat

    # Compute GSADF statistic for actual data
    print(f"  Computing ADF statistics for sample of windows...")
    adf_stats = []
    window_info = []

    # Use adaptive sampling: more aggressive for large datasets
    # Target around 10000-20000 windows for tractability
    target_windows = 15000
    total_possible = (T - r0) * (T - r0 + 1) // 2
    step_r2 = max(1, int(np.sqrt(total_possible / target_windows)))
    step_r1 = step_r2

    print(f"  Dataset size: {T}, sampling every {step_r2} points")

    for r2 in range(r0, T + 1, step_r2):
        for r1 in range(0, r2 - r0 + 1, step_r1):
            t_stat = compute_adf_statistic(y, r1, r2)
            if t_stat is not None:
                adf_stats.append(t_stat)
                window_info.append((r1, r2))

    gsadf_stat = np.max(adf_stats) if adf_stats else 0
    print(f"  GSADF statistic: {gsadf_stat:.4f} (from {len(adf_stats)} windows)")

    # Monte Carlo simulation for critical values - use 1000 instead of 2000 for speed
    num_mc = min(1000, num_simulations)
    print(f"  Running {num_mc} Monte Carlo simulations...")
    gsadf_simulated = []

    for sim in range(num_mc):
        if sim % 250 == 0:
            print(f"    Simulation {sim}/{num_mc}")

        # Generate random walk under null
        e = np.random.randn(T)
        y_sim = np.cumsum(e)

        adf_sim = []
        for r2 in range(r0, T + 1, step_r2):
            for r1 in range(0, r2 - r0 + 1, step_r1):
                t_stat = compute_adf_statistic(y_sim, r1, r2)
                if t_stat is not None:
                    adf_sim.append(t_stat)

        if adf_sim:
            gsadf_simulated.append(np.max(adf_sim))

    # Critical values
    cv = {
        '90%': np.percentile(gsadf_simulated, 90),
        '95%': np.percentile(gsadf_simulated, 95),
        '99%': np.percentile(gsadf_simulated, 99)
    }

    # Date stamping: identify explosive periods
    date_stamp = np.zeros(T, dtype=bool)
    cv_95 = cv['95%']

    for i, (stat, (r1, r2)) in enumerate(zip(adf_stats, window_info)):
        if stat > cv_95:
            # Mark observations in explosive window
            for idx in range(r1, min(r2, T)):
                date_stamp[idx] = True

    return gsadf_stat, cv, date_stamp

def count_episodes(date_stamp):
    """Count distinct bubble episodes by merging contiguous periods"""
    episodes = 0
    in_episode = False

    for is_explosive in date_stamp:
        if is_explosive and not in_episode:
            episodes += 1
            in_episode = True
        elif not is_explosive:
            in_episode = False

    return episodes

# Run GSADF test for each ticker
min_window_frac = 0.01
num_simulations = 2000

tickers = ['LT', 'IOC', 'ITC']
gsadf_results = {}
total_episodes = 0
max_gsadf = -np.inf

for ticker in tickers:
    print(f"\nTesting {ticker}...")
    log_close = df[f'{ticker}_LogClose'].values

    gsadf_stat, cv, date_stamp = gsadf_test(log_close, min_window_frac, num_simulations)
    num_episodes = count_episodes(date_stamp)

    gsadf_results[ticker] = {
        'gsadf_stat': gsadf_stat,
        'cv': cv,
        'date_stamp': date_stamp,
        'num_episodes': num_episodes
    }

    total_episodes += num_episodes
    max_gsadf = max(max_gsadf, gsadf_stat)

    print(f"  GSADF statistic: {gsadf_stat:.4f}")
    print(f"  Critical values: 90%={cv['90%']:.4f}, 95%={cv['95%']:.4f}, 99%={cv['99%']:.4f}")
    print(f"  Number of bubble episodes: {num_episodes}")

print(f"\nTotal bubble episodes across all tickers: {total_episodes}")
print(f"Maximum GSADF statistic: {max_gsadf:.4f}")

# ---------- VAR Estimation and Spillover Analysis ----------
print("\n" + "="*60)
print("PERFORMING VAR ESTIMATION AND SPILLOVER ANALYSIS")
print("="*60)

# Prepare return data (drop NaN from differencing)
returns_df = df[['LT_Return', 'IOC_Return', 'ITC_Return']].dropna().reset_index(drop=True)
print(f"\nReturn observations for VAR: {len(returns_df)}")

# Select lag order by AIC
print("\nSelecting VAR lag order by AIC...")
aic_values = {}

for lag in range(1, 11):
    try:
        model = VAR(returns_df)
        result = model.fit(maxlags=lag, ic=None, trend='c')

        # Check stability (companion roots inside unit circle)
        roots = result.roots
        max_root = np.max(np.abs(roots))

        if max_root < 1.0:  # All roots inside unit circle
            aic_values[lag] = result.aic
            print(f"  Lag {lag}: AIC={result.aic:.4f}, max root={max_root:.4f} (stable)")
        else:
            print(f"  Lag {lag}: max root={max_root:.4f} (unstable, excluded)")
    except:
        print(f"  Lag {lag}: estimation failed")

if not aic_values:
    print("ERROR: No stable VAR model found!")
    optimal_lag = 1
else:
    optimal_lag = min(aic_values, key=aic_values.get)
    print(f"\nOptimal lag order: {optimal_lag} (AIC={aic_values[optimal_lag]:.4f})")

# Fit VAR with optimal lag
model = VAR(returns_df)
var_result = model.fit(maxlags=optimal_lag, ic=None, trend='c')

print("\nVAR Model Summary:")
print(f"  Lag order: {var_result.k_ar}")
print(f"  Number of observations: {var_result.nobs}")
print(f"  AIC: {var_result.aic:.4f}")

# ---------- Forecast Error Variance Decomposition (FEVD) ----------
print("\nComputing 10-period FEVD...")

fevd = var_result.fevd(10)
fevd_10 = fevd.decomp[9, :, :]  # 10-step ahead (index 9)

print("\n10-period FEVD (rows=shocked variable, cols=responding variable):")
print("         LT      IOC      ITC")
for i, ticker in enumerate(tickers):
    print(f"{ticker:5s} {fevd_10[i, 0]:7.4f} {fevd_10[i, 1]:7.4f} {fevd_10[i, 2]:7.4f}")

# ---------- Diebold-Yilmaz Spillover Index ----------
print("\nCalculating Diebold-Yilmaz spillover indices...")

# Convert to percentages
fevd_pct = fevd_10 * 100

# Total spillover index: sum of off-diagonal elements / total * 100
n = len(tickers)
off_diagonal_sum = np.sum(fevd_pct) - np.trace(fevd_pct)
total_sum = np.sum(fevd_pct)
total_spillover = off_diagonal_sum / total_sum * 100

print(f"\nTotal Spillover Index: {total_spillover:.4f}%")

# Net spillover for LT
# LT is index 0, IOC is index 1, ITC is index 2
# Outgoing from LT: fevd_pct[0, 1] (LT→IOC) + fevd_pct[0, 2] (LT→ITC)
# Incoming to LT: fevd_pct[1, 0] (IOC→LT) + fevd_pct[2, 0] (ITC→LT)
outgoing_LT = fevd_pct[0, 1] + fevd_pct[0, 2]
incoming_LT = fevd_pct[1, 0] + fevd_pct[2, 0]
net_spillover_LT = outgoing_LT - incoming_LT

print(f"\nLT Net Spillover:")
print(f"  Outgoing (LT→IOC + LT→ITC): {outgoing_LT:.4f}%")
print(f"  Incoming (IOC→LT + ITC→LT): {incoming_LT:.4f}%")
print(f"  Net: {net_spillover_LT:.4f}%")

# Average total spillover index (should be same as total_spillover)
avg_total_spillover = total_spillover

# Average LT net spillover
avg_lt_net_spillover = net_spillover_LT

# ---------- Yang-Zhang Volatility ----------
print("\n" + "="*60)
print("COMPUTING YANG-ZHANG VOLATILITY")
print("="*60)

def yang_zhang_volatility(open_prices, high_prices, low_prices, close_prices, window=30, annualize_factor=252):
    """
    Calculate Yang-Zhang volatility estimator

    Formula combines overnight, open-to-close, and Rogers-Satchell components
    """
    n = len(close_prices)
    yz_vol = []

    for i in range(window - 1, n):
        # Get window data
        O = open_prices[i - window + 1:i + 1]
        H = high_prices[i - window + 1:i + 1]
        L = low_prices[i - window + 1:i + 1]
        C = close_prices[i - window + 1:i + 1]
        C_prev = np.concatenate([[close_prices[i - window]], C[:-1]])

        # Overnight volatility: var(log(O_t / C_{t-1}))
        overnight = np.log(O / C_prev)
        var_o = np.var(overnight, ddof=1)

        # Open-to-close volatility: var(log(C_t / O_t))
        open_close = np.log(C / O)
        var_c = np.var(open_close, ddof=1)

        # Rogers-Satchell volatility
        rs = np.log(H / C) * np.log(H / O) + np.log(L / C) * np.log(L / O)
        var_rs = np.mean(rs)

        # Yang-Zhang estimator
        k = 0.34 / (1.34 + (window + 1) / (window - 1))
        var_yz = var_o + k * var_c + (1 - k) * var_rs

        # Annualized volatility
        vol_annualized = np.sqrt(var_yz * annualize_factor)
        yz_vol.append(vol_annualized)

    return np.array(yz_vol)

window = 30
annualize_factor = 252

median_vols = {}

for ticker in tickers:
    open_p = df[f'{ticker}_Open'].values
    high_p = df[f'{ticker}_High'].values
    low_p = df[f'{ticker}_Low'].values
    close_p = df[f'{ticker}_Close'].values

    yz_vol = yang_zhang_volatility(open_p, high_p, low_p, close_p, window, annualize_factor)
    median_vol = np.median(yz_vol)
    median_vols[ticker] = median_vol

    print(f"{ticker} median Yang-Zhang volatility: {median_vol:.4f}")

cross_asset_avg = np.mean(list(median_vols.values()))
print(f"\nCross-asset average median Yang-Zhang volatility: {cross_asset_avg:.4f}")

# ---------- Granger Causality Tests ----------
print("\n" + "="*60)
print("PERFORMING PAIRWISE GRANGER CAUSALITY TESTS")
print("="*60)

# All ordered pairs
pairs = [
    ('LT', 'IOC'),
    ('IOC', 'LT'),
    ('LT', 'ITC'),
    ('ITC', 'LT'),
    ('IOC', 'ITC'),
    ('ITC', 'IOC')
]

significant_pairs = 0

for ticker1, ticker2 in pairs:
    # Test if ticker1 Granger-causes ticker2
    data = returns_df[[f'{ticker2}_Return', f'{ticker1}_Return']]

    # Test lags 1-5
    significant = False
    max_lag = 5

    try:
        # grangercausalitytests tests if the second column causes the first
        result = grangercausalitytests(data, maxlag=max_lag, verbose=False)

        for lag in range(1, max_lag + 1):
            # F-test p-value
            p_value = result[lag][0]['ssr_ftest'][1]
            if p_value < 0.05:
                significant = True
                break

        if significant:
            significant_pairs += 1
            print(f"{ticker1} → {ticker2}: Significant (p < 0.05)")
        else:
            print(f"{ticker1} → {ticker2}: Not significant")
    except:
        print(f"{ticker1} → {ticker2}: Test failed")

print(f"\nTotal significant directed pairs: {significant_pairs}")

# ---------- Lower Tail Co-Exceedance Heatmap ----------
print("\n" + "="*60)
print("CREATING LOWER TAIL CO-EXCEEDANCE HEATMAP")
print("="*60)

# Get returns with dates
returns_with_dates = df[['Date', 'LT_Return', 'IOC_Return', 'ITC_Return']].dropna().reset_index(drop=True)

# Find month-end dates in the data
returns_with_dates['YearMonth'] = returns_with_dates['Date'].dt.to_period('M')
month_ends = returns_with_dates.groupby('YearMonth')['Date'].max().reset_index()
month_ends = month_ends.sort_values('Date')

# Get most recent 24 months
recent_24_months = month_ends.tail(24)

print(f"Month-end dates: {len(recent_24_months)} months")
print(f"Date range: {recent_24_months['Date'].min()} to {recent_24_months['Date'].max()}")

# Stock pairs
stock_pairs = [('LT', 'IOC'), ('LT', 'ITC'), ('IOC', 'ITC')]

# Create heatmap matrix
heatmap_data = []
lookback_window = 252

for pair in stock_pairs:
    row = []
    for month_end_date in recent_24_months['Date']:
        # Get data for preceding 252 trading days
        month_end_idx = returns_with_dates[returns_with_dates['Date'] == month_end_date].index[0]

        if month_end_idx >= lookback_window:
            window_data = returns_with_dates.iloc[month_end_idx - lookback_window + 1:month_end_idx + 1]

            ret1 = window_data[f'{pair[0]}_Return'].values
            ret2 = window_data[f'{pair[1]}_Return'].values

            # Calculate 5th percentiles within this window
            p5_1 = np.percentile(ret1, 5)
            p5_2 = np.percentile(ret2, 5)

            # Count days where both are at or below their 5th percentile
            co_exceed = np.sum((ret1 <= p5_1) & (ret2 <= p5_2))
            fraction = co_exceed / len(ret1)
            row.append(fraction)
        else:
            row.append(np.nan)

    heatmap_data.append(row)

heatmap_array = np.array(heatmap_data)

# Report highest cell value
max_cell_value = np.nanmax(heatmap_array)
print(f"\nHighest co-exceedance cell value: {max_cell_value:.4f}")

# Create heatmap visualization
fig, ax = plt.subplots(figsize=(14, 6))
im = ax.imshow(heatmap_array, cmap='YlOrRd', aspect='auto', vmin=0)

# Set ticks
ax.set_yticks(range(len(stock_pairs)))
ax.set_yticklabels([f"{p[0]}-{p[1]}" for p in stock_pairs])

# Set x-axis labels (show every 3rd month to avoid crowding)
x_labels = [d.strftime('%Y-%m') for d in recent_24_months['Date']]
ax.set_xticks(range(len(x_labels)))
ax.set_xticklabels(x_labels, rotation=45, ha='right')
# Show every 3rd label
for i, label in enumerate(ax.xaxis.get_ticklabels()):
    if i % 3 != 0:
        label.set_visible(False)

ax.set_xlabel('Month End Date')
ax.set_ylabel('Stock Pair')
ax.set_title('Lower Tail Co-Exceedance (Fraction of Days Both Returns ≤ 5th Percentile)\nLookback: 252 Trading Days')

# Add colorbar
cbar = plt.colorbar(im, ax=ax)
cbar.set_label('Co-Exceedance Fraction')

plt.tight_layout()
plt.savefig('lower_tail_heatmap.png', dpi=300, bbox_inches='tight')
print("Heatmap saved to: lower_tail_heatmap.png")

# ---------- Diversification Test ----------
print("\n" + "="*60)
print("PERFORMING DIVERSIFICATION TEST")
print("="*60)

# Calculate monthly averages across pairs for recent 36 and prior 36 months
all_months = month_ends['Date'].values

if len(all_months) >= 72:
    recent_36_months = all_months[-36:]
    prior_36_months = all_months[-72:-36]

    print(f"Recent 36 months: {recent_36_months[0]} to {recent_36_months[-1]}")
    print(f"Prior 36 months: {prior_36_months[0]} to {prior_36_months[-1]}")

    # Calculate co-exceedance for each month and pair
    def calc_coexceed_for_month(month_end_date):
        month_end_idx = returns_with_dates[returns_with_dates['Date'] == month_end_date].index[0]

        if month_end_idx >= lookback_window:
            window_data = returns_with_dates.iloc[month_end_idx - lookback_window + 1:month_end_idx + 1]

            pair_values = []
            for pair in stock_pairs:
                ret1 = window_data[f'{pair[0]}_Return'].values
                ret2 = window_data[f'{pair[1]}_Return'].values

                p5_1 = np.percentile(ret1, 5)
                p5_2 = np.percentile(ret2, 5)

                co_exceed = np.sum((ret1 <= p5_1) & (ret2 <= p5_2))
                fraction = co_exceed / len(ret1)
                pair_values.append(fraction)

            # Average across pairs for this month
            return np.mean(pair_values)
        return np.nan

    recent_values = [calc_coexceed_for_month(d) for d in recent_36_months]
    prior_values = [calc_coexceed_for_month(d) for d in prior_36_months]

    # Remove NaNs
    recent_values = [v for v in recent_values if not np.isnan(v)]
    prior_values = [v for v in prior_values if not np.isnan(v)]

    recent_avg = np.mean(recent_values)
    prior_avg = np.mean(prior_values)

    print(f"\nRecent 36-month average co-exceedance: {recent_avg:.6f}")
    print(f"Prior 36-month average co-exceedance: {prior_avg:.6f}")

    # Two-sided Welch t-test
    t_stat, p_value = stats.ttest_ind(recent_values, prior_values, equal_var=False)

    print(f"\nWelch t-test:")
    print(f"  t-statistic: {t_stat:.4f}")
    print(f"  p-value: {p_value:.4f}")

    # Binary indicator: 1 if recent < prior AND significant at 5% level
    diversification_indicator = 1 if (recent_avg < prior_avg and p_value < 0.05) else 0

    print(f"\nDiversification indicator: {diversification_indicator}")

    if diversification_indicator == 1:
        verdict = "IMPROVED DIVERSIFICATION: The recent 36-month period shows significantly lower tail co-exceedance than the prior period, indicating reduced tail dependence and improved diversification benefits."
    else:
        if recent_avg < prior_avg:
            verdict = "NO SIGNIFICANT IMPROVEMENT: Although recent co-exceedance is lower, the difference is not statistically significant at the 5% level."
        else:
            verdict = "NO IMPROVEMENT: Recent co-exceedance is not lower than the prior period, suggesting no improvement in diversification."

    print(f"\nVERDICT: {verdict}")
else:
    print(f"Insufficient data for 72 months (need {72}, have {len(all_months)})")
    diversification_indicator = 0
    verdict = "INSUFFICIENT DATA"

# ---------- Summary Report ----------
print("\n" + "="*80)
print("FINAL SUMMARY REPORT")
print("="*80)

print(f"\n1. BUBBLE DETECTION (GSADF):")
print(f"   Total distinct bubble episodes: {total_episodes}")
print(f"   Maximum GSADF statistic: {max_gsadf:.4f}")

print(f"\n2. SPILLOVER ANALYSIS:")
print(f"   Average total spillover index: {avg_total_spillover:.4f}%")
print(f"   Average LT net spillover: {avg_lt_net_spillover:.4f}%")

print(f"\n3. VOLATILITY:")
print(f"   Cross-asset average median Yang-Zhang volatility: {cross_asset_avg:.4f}")

print(f"\n4. GRANGER CAUSALITY:")
print(f"   Total significant directed pairs: {significant_pairs}")

print(f"\n5. TAIL RISK:")
print(f"   Highest co-exceedance cell value: {max_cell_value:.4f}")

print(f"\n6. DIVERSIFICATION TEST:")
print(f"   Binary indicator: {diversification_indicator}")
print(f"   {verdict}")

# Save results to file
with open('analysis_results.txt', 'w') as f:
    f.write("BUBBLE & SPILLOVER ANALYSIS RESULTS\n")
    f.write("="*80 + "\n\n")

    f.write("REQUIRED OUTPUTS:\n")
    f.write(f"  Total bubble episodes: {total_episodes}\n")
    f.write(f"  Maximum GSADF statistic: {max_gsadf:.4f}\n")
    f.write(f"  Average total spillover index: {avg_total_spillover:.4f}\n")
    f.write(f"  Average LT net spillover: {avg_lt_net_spillover:.4f}\n")
    f.write(f"  Cross-asset average median Yang-Zhang volatility: {cross_asset_avg:.4f}\n")
    f.write(f"  Total significant Granger causality pairs: {significant_pairs}\n")
    f.write(f"  Highest co-exceedance cell value: {max_cell_value:.4f}\n")
    f.write(f"  Diversification indicator: {diversification_indicator}\n")
    f.write(f"\nDIVERSIFICATION VERDICT:\n  {verdict}\n")

print("\nResults saved to: analysis_results.txt")
print("\nAnalysis complete!")
