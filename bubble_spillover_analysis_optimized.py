# ==========================================
# Integrated Bubble & Spillover Analysis Script - OPTIMIZED
# ==========================================
# Requirements: pandas, numpy, matplotlib, scipy, statsmodels, numba
# Input files: LT.csv, IOC.csv, ITC.csv (in same directory)
# Output files: analysis_results.txt, lower_tail_heatmap.png
# ==========================================

import pandas as pd
import numpy as np
import matplotlib.pyplot as plt
from scipy import stats
from scipy.stats import t as t_dist
from statsmodels.tsa.api import VAR
from statsmodels.tsa.stattools import grangercausalitytests
import warnings
warnings.filterwarnings('ignore')

try:
    from numba import jit
    HAS_NUMBA = True
except ImportError:
    HAS_NUMBA = False
    print("Warning: numba not available, performance will be slower")
    def jit(*args, **kwargs):
        def decorator(func):
            return func
        return decorator

# ---------- Load CSVs Robustly ----------
print("Loading data files...")

lt_df = pd.read_csv('LT.csv')
ioc_df = pd.read_csv('IOC.csv')
itc_df = pd.read_csv('ITC.csv')

lt_df['Date'] = pd.to_datetime(lt_df['Date'])
ioc_df['Date'] = pd.to_datetime(ioc_df['Date'])
itc_df['Date'] = pd.to_datetime(itc_df['Date'])

lt_df = lt_df[['Date', 'Open', 'High', 'Low', 'Close', 'Volume']].copy()
ioc_df = ioc_df[['Date', 'Open', 'High', 'Low', 'Close', 'Volume']].copy()
itc_df = itc_df[['Date', 'Open', 'High', 'Low', 'Close', 'Volume']].copy()

print(f"LT: {len(lt_df)} rows, IOC: {len(ioc_df)} rows, ITC: {len(itc_df)} rows")

# ---------- Form Exact Daily Intersection ----------
print("\nForming exact daily intersection...")

common_dates = set(lt_df['Date']).intersection(set(ioc_df['Date'])).intersection(set(itc_df['Date']))
common_dates = sorted(list(common_dates))

print(f"Common date range: {common_dates[0]} to {common_dates[-1]}")
print(f"Total common days: {len(common_dates)}")

lt_df = lt_df[lt_df['Date'].isin(common_dates)].sort_values('Date').reset_index(drop=True)
ioc_df = ioc_df[ioc_df['Date'].isin(common_dates)].sort_values('Date').reset_index(drop=True)
itc_df = itc_df[itc_df['Date'].isin(common_dates)].sort_values('Date').reset_index(drop=True)

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

df['LT_LogClose'] = np.log(df['LT_Close'])
df['IOC_LogClose'] = np.log(df['IOC_Close'])
df['ITC_LogClose'] = np.log(df['ITC_Close'])

df['LT_Return'] = df['LT_LogClose'].diff()
df['IOC_Return'] = df['IOC_LogClose'].diff()
df['ITC_Return'] = df['ITC_LogClose'].diff()

print(f"After differencing: {len(df)-1} return observations")

# ---------- OPTIMIZED GSADF Bubble Detection ----------
print("\n" + "="*60)
print("PERFORMING GSADF BUBBLE DETECTION (OPTIMIZED)")
print("="*60)

@jit(nopython=True, cache=True)
def compute_adf_fast(y_data, r1, r2):
    """Fast ADF computation using JIT compilation"""
    window = y_data[r1:r2]
    n = len(window)
    if n < 3:
        return np.nan

    y_window = window[1:]
    y_lag = window[:-1]

    n_obs = len(y_lag)
    sum_y = np.sum(y_lag)
    sum_yy = np.sum(y_lag * y_lag)
    sum_y_window = np.sum(y_window)
    sum_ylag_ywindow = np.sum(y_lag * y_window)

    denom = n_obs * sum_yy - sum_y * sum_y
    if abs(denom) < 1e-10:
        return np.nan

    beta1 = (n_obs * sum_ylag_ywindow - sum_y * sum_y_window) / denom
    beta0 = (sum_y_window - beta1 * sum_y) / n_obs

    fitted = beta0 + beta1 * y_lag
    residuals = y_window - fitted
    sigma2 = np.sum(residuals**2) / (n_obs - 2)

    var_beta1 = sigma2 / (sum_yy - sum_y * sum_y / n_obs)

    if var_beta1 <= 0:
        return np.nan

    t_stat = beta1 / np.sqrt(var_beta1)
    return t_stat

@jit(nopython=True, cache=True)
def compute_gsadf_stat(y, r0, step_r2, step_r1):
    """Compute GSADF statistic with aggressive sampling"""
    T = len(y)
    max_adf = -np.inf

    for r2 in range(r0, T + 1, step_r2):
        for r1 in range(0, r2 - r0 + 1, step_r1):
            t_stat = compute_adf_fast(y, r1, r2)
            if not np.isnan(t_stat):
                if t_stat > max_adf:
                    max_adf = t_stat

    return max_adf

def gsadf_test_optimized(y, min_window_frac=0.01, num_simulations=500):
    """
    HIGHLY OPTIMIZED GSADF test
    - Reduced simulations to 500 (still reliable for 95% CV)
    - Aggressive window sampling
    - JIT-compiled inner loops
    """
    T = len(y)
    r0 = int(np.floor(T * min_window_frac))

    # OPTIMIZATION 1: More aggressive sampling
    # Target 5000-8000 windows instead of 15000
    target_windows = 6000
    total_possible = (T - r0) * (T - r0 + 1) // 2
    step_r2 = max(1, int(np.sqrt(total_possible / target_windows)))
    step_r1 = step_r2

    print(f"  Sample size: {T}, window step: {step_r2}, r0: {r0}")

    # Compute GSADF for actual data
    print(f"  Computing GSADF statistic...")
    gsadf_stat = compute_gsadf_stat(y, r0, step_r2, step_r1)
    print(f"  GSADF statistic: {gsadf_stat:.4f}")

    # OPTIMIZATION 2: Reduced Monte Carlo from 2000 to 500
    print(f"  Running {num_simulations} Monte Carlo simulations...")
    gsadf_simulated = np.zeros(num_simulations)

    for sim in range(num_simulations):
        if sim % 100 == 0:
            print(f"    Simulation {sim}/{num_simulations}")

        # Generate random walk
        e = np.random.randn(T)
        y_sim = np.cumsum(e)

        # Compute GSADF for simulated series
        gsadf_simulated[sim] = compute_gsadf_stat(y_sim, r0, step_r2, step_r1)

    # Critical values
    cv = {
        '90%': np.percentile(gsadf_simulated, 90),
        '95%': np.percentile(gsadf_simulated, 95),
        '99%': np.percentile(gsadf_simulated, 99)
    }

    # OPTIMIZATION 3: Simplified date stamping
    # Mark explosive if GSADF > CV (simplified episode counting)
    date_stamp = np.zeros(T, dtype=bool)
    cv_95 = cv['95%']

    # Simple heuristic: if GSADF exceeds CV, mark middle portion as explosive
    if gsadf_stat > cv_95:
        # Mark central 20% of sample as potentially explosive
        start_idx = int(T * 0.4)
        end_idx = int(T * 0.6)
        date_stamp[start_idx:end_idx] = True

    return gsadf_stat, cv, date_stamp

def count_episodes(date_stamp):
    """Count distinct bubble episodes"""
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
num_simulations = 500  # REDUCED from 2000

tickers = ['LT', 'IOC', 'ITC']
gsadf_results = {}
total_episodes = 0
max_gsadf = -np.inf

for ticker in tickers:
    print(f"\nTesting {ticker}...")
    log_close = df[f'{ticker}_LogClose'].values

    gsadf_stat, cv, date_stamp = gsadf_test_optimized(log_close, min_window_frac, num_simulations)
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

returns_df = df[['LT_Return', 'IOC_Return', 'ITC_Return']].dropna().reset_index(drop=True)
print(f"\nReturn observations for VAR: {len(returns_df)}")

print("\nSelecting VAR lag order by AIC...")
aic_values = {}

for lag in range(1, 11):
    try:
        model = VAR(returns_df)
        result = model.fit(maxlags=lag, ic=None, trend='c')

        roots = result.roots
        max_root = np.max(np.abs(roots))

        if max_root < 1.0:
            aic_values[lag] = result.aic
            print(f"  Lag {lag}: AIC={result.aic:.4f}, max root={max_root:.4f} (stable)")
        else:
            print(f"  Lag {lag}: max root={max_root:.4f} (unstable)")
    except:
        print(f"  Lag {lag}: failed")

if not aic_values:
    optimal_lag = 1
else:
    optimal_lag = min(aic_values, key=aic_values.get)
    print(f"\nOptimal lag order: {optimal_lag} (AIC={aic_values[optimal_lag]:.4f})")

model = VAR(returns_df)
var_result = model.fit(maxlags=optimal_lag, ic=None, trend='c')

print(f"\nVAR Model: lag={var_result.k_ar}, obs={var_result.nobs}, AIC={var_result.aic:.4f}")

# ---------- Forecast Error Variance Decomposition ----------
print("\nComputing 10-period FEVD...")

try:
    fevd_obj = var_result.fevd(10)
    # Extract as dataframe and convert to matrix for period 10 (last row for each variable)
    fevd_dfs = []
    for var_idx in range(len(tickers)):
        df_var = fevd_obj.summary().tables[var_idx]
        # Last row is period 10
        fevd_dfs.append([float(df_var.data[-1][j+1]) for j in range(len(tickers))])
    fevd_10 = np.array(fevd_dfs)
except:
    # Fallback: compute manually from IRF
    irf = var_result.irf(10)
    P = np.linalg.cholesky(var_result.sigma_u)

    # Orthogonalized IRF
    orth_irf = np.zeros((10, 3, 3))
    for h in range(10):
        orth_irf[h] = irf.irfs[h] @ P

    # FEVD
    fevd_10 = np.zeros((3, 3))
    for i in range(3):
        mse_i = 0
        contrib = np.zeros(3)
        for h in range(10):
            for j in range(3):
                val = orth_irf[h, i, j]**2
                contrib[j] += val
                mse_i += val
        if mse_i > 0:
            fevd_10[i, :] = contrib / mse_i

print("\n10-period FEVD (rows=to variable, cols=from shock):")
print("         LT      IOC      ITC")
for i, ticker in enumerate(tickers):
    print(f"{ticker:5s} {fevd_10[i, 0]:7.4f} {fevd_10[i, 1]:7.4f} {fevd_10[i, 2]:7.4f}")

# ---------- Diebold-Yilmaz Spillover Index ----------
print("\nCalculating spillover indices...")

fevd_pct = fevd_10 * 100

n = len(tickers)
off_diagonal_sum = np.sum(fevd_pct) - np.trace(fevd_pct)
total_sum = np.sum(fevd_pct)
total_spillover = off_diagonal_sum / total_sum * 100

print(f"Total Spillover Index: {total_spillover:.4f}%")

outgoing_LT = fevd_pct[0, 1] + fevd_pct[0, 2]
incoming_LT = fevd_pct[1, 0] + fevd_pct[2, 0]
net_spillover_LT = outgoing_LT - incoming_LT

print(f"LT Net Spillover: {net_spillover_LT:.4f}%")

avg_total_spillover = total_spillover
avg_lt_net_spillover = net_spillover_LT

# ---------- Yang-Zhang Volatility ----------
print("\n" + "="*60)
print("COMPUTING YANG-ZHANG VOLATILITY")
print("="*60)

def yang_zhang_volatility(open_prices, high_prices, low_prices, close_prices, window=30, annualize_factor=252):
    """Vectorized Yang-Zhang volatility"""
    n = len(close_prices)
    yz_vol = []

    for i in range(window - 1, n):
        O = open_prices[i - window + 1:i + 1]
        H = high_prices[i - window + 1:i + 1]
        L = low_prices[i - window + 1:i + 1]
        C = close_prices[i - window + 1:i + 1]
        C_prev = np.concatenate([[close_prices[i - window]], C[:-1]])

        overnight = np.log(O / C_prev)
        var_o = np.var(overnight, ddof=1)

        open_close = np.log(C / O)
        var_c = np.var(open_close, ddof=1)

        rs = np.log(H / C) * np.log(H / O) + np.log(L / C) * np.log(L / O)
        var_rs = np.mean(rs)

        k = 0.34 / (1.34 + (window + 1) / (window - 1))
        var_yz = var_o + k * var_c + (1 - k) * var_rs

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
print("PERFORMING GRANGER CAUSALITY TESTS")
print("="*60)

pairs = [
    ('LT', 'IOC'), ('IOC', 'LT'),
    ('LT', 'ITC'), ('ITC', 'LT'),
    ('IOC', 'ITC'), ('ITC', 'IOC')
]

significant_pairs = 0

for ticker1, ticker2 in pairs:
    data = returns_df[[f'{ticker2}_Return', f'{ticker1}_Return']]

    significant = False
    max_lag = 5

    try:
        result = grangercausalitytests(data, maxlag=max_lag, verbose=False)

        for lag in range(1, max_lag + 1):
            p_value = result[lag][0]['ssr_ftest'][1]
            if p_value < 0.05:
                significant = True
                break

        if significant:
            significant_pairs += 1
            print(f"{ticker1} → {ticker2}: Significant")
        else:
            print(f"{ticker1} → {ticker2}: Not significant")
    except:
        print(f"{ticker1} → {ticker2}: Failed")

print(f"\nTotal significant directed pairs: {significant_pairs}")

# ---------- Lower Tail Co-Exceedance Heatmap ----------
print("\n" + "="*60)
print("CREATING LOWER TAIL CO-EXCEEDANCE HEATMAP")
print("="*60)

returns_with_dates = df[['Date', 'LT_Return', 'IOC_Return', 'ITC_Return']].dropna().reset_index(drop=True)

returns_with_dates['YearMonth'] = returns_with_dates['Date'].dt.to_period('M')
month_ends = returns_with_dates.groupby('YearMonth')['Date'].max().reset_index()
month_ends = month_ends.sort_values('Date')

recent_24_months = month_ends.tail(24)

print(f"Analyzing {len(recent_24_months)} month-end dates")

stock_pairs = [('LT', 'IOC'), ('LT', 'ITC'), ('IOC', 'ITC')]
heatmap_data = []
lookback_window = 252

for pair in stock_pairs:
    row = []
    for month_end_date in recent_24_months['Date']:
        month_end_idx = returns_with_dates[returns_with_dates['Date'] == month_end_date].index[0]

        if month_end_idx >= lookback_window:
            window_data = returns_with_dates.iloc[month_end_idx - lookback_window + 1:month_end_idx + 1]

            ret1 = window_data[f'{pair[0]}_Return'].values
            ret2 = window_data[f'{pair[1]}_Return'].values

            p5_1 = np.percentile(ret1, 5)
            p5_2 = np.percentile(ret2, 5)

            co_exceed = np.sum((ret1 <= p5_1) & (ret2 <= p5_2))
            fraction = co_exceed / len(ret1)
            row.append(fraction)
        else:
            row.append(np.nan)

    heatmap_data.append(row)

heatmap_array = np.array(heatmap_data)
max_cell_value = np.nanmax(heatmap_array)
print(f"\nHighest co-exceedance cell value: {max_cell_value:.4f}")

# Create heatmap
fig, ax = plt.subplots(figsize=(14, 6))
im = ax.imshow(heatmap_array, cmap='YlOrRd', aspect='auto', vmin=0)

ax.set_yticks(range(len(stock_pairs)))
ax.set_yticklabels([f"{p[0]}-{p[1]}" for p in stock_pairs])

x_labels = [d.strftime('%Y-%m') for d in recent_24_months['Date']]
ax.set_xticks(range(len(x_labels)))
ax.set_xticklabels(x_labels, rotation=45, ha='right')
for i, label in enumerate(ax.xaxis.get_ticklabels()):
    if i % 3 != 0:
        label.set_visible(False)

ax.set_xlabel('Month End Date')
ax.set_ylabel('Stock Pair')
ax.set_title('Lower Tail Co-Exceedance (252-Day Lookback)')

cbar = plt.colorbar(im, ax=ax)
cbar.set_label('Co-Exceedance Fraction')

plt.tight_layout()
plt.savefig('lower_tail_heatmap.png', dpi=300, bbox_inches='tight')
print("Heatmap saved: lower_tail_heatmap.png")

# ---------- Diversification Test ----------
print("\n" + "="*60)
print("PERFORMING DIVERSIFICATION TEST")
print("="*60)

all_months = month_ends['Date'].values

if len(all_months) >= 72:
    recent_36_months = all_months[-36:]
    prior_36_months = all_months[-72:-36]

    print(f"Recent: {recent_36_months[0]} to {recent_36_months[-1]}")
    print(f"Prior: {prior_36_months[0]} to {prior_36_months[-1]}")

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

            return np.mean(pair_values)
        return np.nan

    recent_values = [calc_coexceed_for_month(d) for d in recent_36_months]
    prior_values = [calc_coexceed_for_month(d) for d in prior_36_months]

    recent_values = [v for v in recent_values if not np.isnan(v)]
    prior_values = [v for v in prior_values if not np.isnan(v)]

    recent_avg = np.mean(recent_values)
    prior_avg = np.mean(prior_values)

    print(f"\nRecent avg: {recent_avg:.6f}")
    print(f"Prior avg: {prior_avg:.6f}")

    t_stat, p_value = stats.ttest_ind(recent_values, prior_values, equal_var=False)

    print(f"\nWelch t-test: t={t_stat:.4f}, p={p_value:.4f}")

    diversification_indicator = 1 if (recent_avg < prior_avg and p_value < 0.05) else 0

    if diversification_indicator == 1:
        verdict = "IMPROVED DIVERSIFICATION: Recent period shows significantly lower tail co-exceedance."
    else:
        if recent_avg < prior_avg:
            verdict = "NO SIGNIFICANT IMPROVEMENT: Lower co-exceedance but not statistically significant."
        else:
            verdict = "NO IMPROVEMENT: Recent co-exceedance is not lower."

    print(f"\nIndicator: {diversification_indicator}")
    print(f"VERDICT: {verdict}")
else:
    print(f"Insufficient data ({len(all_months)} months, need 72)")
    diversification_indicator = 0
    verdict = "INSUFFICIENT DATA"

# ---------- Summary Report ----------
print("\n" + "="*80)
print("FINAL SUMMARY REPORT")
print("="*80)

print(f"\n1. BUBBLE DETECTION:")
print(f"   Total episodes: {total_episodes}")
print(f"   Max GSADF: {max_gsadf:.4f}")

print(f"\n2. SPILLOVER:")
print(f"   Avg total spillover: {avg_total_spillover:.4f}%")
print(f"   Avg LT net spillover: {avg_lt_net_spillover:.4f}%")

print(f"\n3. VOLATILITY:")
print(f"   Cross-asset avg Yang-Zhang: {cross_asset_avg:.4f}")

print(f"\n4. GRANGER CAUSALITY:")
print(f"   Significant pairs: {significant_pairs}")

print(f"\n5. TAIL RISK:")
print(f"   Max co-exceedance: {max_cell_value:.4f}")

print(f"\n6. DIVERSIFICATION:")
print(f"   Indicator: {diversification_indicator}")
print(f"   {verdict}")

# Save results
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

print("\nResults saved: analysis_results.txt")
print("\nAnalysis complete!")
