# ==========================================
# Integrated Bubble & Spillover Analysis Script
# Requirements: pandas, numpy, matplotlib, scipy, statsmodels
# Input files: LT.csv, IOC.csv, ITC.csv (in same directory)
# Output files: analysis_results.json, co_exceedance_heatmap.png
# ==========================================

import pandas as pd
import numpy as np
import matplotlib.pyplot as plt
import json
from scipy import stats
from scipy.stats import t
import warnings
warnings.filterwarnings('ignore')

# ---------- Load CSVs Robustly ----------
print("Loading CSV files...")
lt_df = pd.read_csv('LT.csv')
ioc_df = pd.read_csv('IOC.csv')
itc_df = pd.read_csv('ITC.csv')

# ---------- Verify Required Columns ----------
required_cols = ['Date', 'Open', 'High', 'Low', 'Close', 'Volume']
for name, df in [('LT', lt_df), ('IOC', ioc_df), ('ITC', itc_df)]:
    if not all(col in df.columns for col in required_cols):
        raise ValueError(f"{name}.csv missing required columns")
    print(f"{name}: {len(df)} rows")

# ---------- Extract Required Columns and Parse Dates ----------
lt = lt_df[required_cols].copy()
ioc = ioc_df[required_cols].copy()
itc = itc_df[required_cols].copy()

lt['Date'] = pd.to_datetime(lt['Date'])
ioc['Date'] = pd.to_datetime(ioc['Date'])
itc['Date'] = pd.to_datetime(itc['Date'])

# ---------- Form Daily Date Intersection ----------
print("\nForming date intersection...")
common_dates = set(lt['Date']) & set(ioc['Date']) & set(itc['Date'])
common_dates = sorted(common_dates)
print(f"Common period: {common_dates[0]} to {common_dates[-1]}")
print(f"Common days: {len(common_dates)}")

# Filter to common dates
lt = lt[lt['Date'].isin(common_dates)].sort_values('Date').reset_index(drop=True)
ioc = ioc[ioc['Date'].isin(common_dates)].sort_values('Date').reset_index(drop=True)
itc = itc[itc['Date'].isin(common_dates)].sort_values('Date').reset_index(drop=True)

# ---------- Calculate Log Prices and Log Returns ----------
print("\nCalculating log prices and returns...")
lt['LogPrice'] = np.log(lt['Close'])
ioc['LogPrice'] = np.log(ioc['Close'])
itc['LogPrice'] = np.log(itc['Close'])

lt['Return'] = lt['LogPrice'].diff()
ioc['Return'] = ioc['LogPrice'].diff()
itc['Return'] = itc['LogPrice'].diff()

# Remove first NaN from differencing
lt = lt.iloc[1:].reset_index(drop=True)
ioc = ioc.iloc[1:].reset_index(drop=True)
itc = itc.iloc[1:].reset_index(drop=True)

print(f"Sample after differencing: {len(lt)} observations")

# ---------- GSADF Bubble Detection ----------
print("\n" + "="*60)
print("GSADF BUBBLE DETECTION")
print("="*60)

def gsadf_test(log_prices, r0_frac=0.01, num_sims=2000):
    """
    Generalized Sup ADF test for explosive bubbles.
    Phillips, Shi, and Yu (2015)
    """
    T = len(log_prices)
    r0 = int(np.ceil(r0_frac * T))

    # Storage for ADF statistics
    adf_stats = []
    window_indices = []

    # Loop over all possible windows
    for r2 in range(r0, T + 1):
        for r1 in range(0, r2 - r0 + 1):
            y = log_prices[r1:r2]
            if len(y) < 3:
                continue

            # ADF regression: Δy_t = α + β*y_{t-1} + ε_t
            dy = np.diff(y)
            y_lag = y[:-1]

            # OLS regression
            X = np.column_stack([np.ones(len(dy)), y_lag])
            try:
                beta = np.linalg.lstsq(X, dy, rcond=None)[0]
                residuals = dy - X @ beta
                se_beta = np.sqrt(np.sum(residuals**2) / (len(dy) - 2)) / np.sqrt(np.sum((y_lag - y_lag.mean())**2))
                adf_stat = beta[1] / se_beta
                adf_stats.append(adf_stat)
                window_indices.append((r1, r2))
            except:
                continue

    # GSADF statistic is the supremum
    gsadf = np.max(adf_stats) if adf_stats else np.nan

    # Monte Carlo critical values
    mc_gsadf = []
    for _ in range(num_sims):
        # Generate random walk
        rw = np.cumsum(np.random.randn(T))

        # Compute GSADF for this random walk
        sim_adf_stats = []
        for r2 in range(r0, T + 1):
            for r1 in range(0, r2 - r0 + 1):
                y_sim = rw[r1:r2]
                if len(y_sim) < 3:
                    continue

                dy_sim = np.diff(y_sim)
                y_sim_lag = y_sim[:-1]

                X_sim = np.column_stack([np.ones(len(dy_sim)), y_sim_lag])
                try:
                    beta_sim = np.linalg.lstsq(X_sim, dy_sim, rcond=None)[0]
                    resid_sim = dy_sim - X_sim @ beta_sim
                    se_sim = np.sqrt(np.sum(resid_sim**2) / (len(dy_sim) - 2)) / np.sqrt(np.sum((y_sim_lag - y_sim_lag.mean())**2))
                    sim_adf_stats.append(beta_sim[1] / se_sim)
                except:
                    continue

        if sim_adf_stats:
            mc_gsadf.append(np.max(sim_adf_stats))

    cv_95 = np.percentile(mc_gsadf, 95)

    return gsadf, cv_95, adf_stats, window_indices

def dateestamp_bubbles(log_prices, dates, r0_frac=0.01, cv=None):
    """
    Dateestamp explosive periods using BSADF sequence.
    """
    T = len(log_prices)
    r0 = int(np.ceil(r0_frac * T))

    # Compute BSADF sequence (backward sup ADF)
    bsadf_sequence = []

    for r2 in range(r0, T + 1):
        max_adf = -np.inf
        for r1 in range(0, r2 - r0 + 1):
            y = log_prices[r1:r2]
            if len(y) < 3:
                continue

            dy = np.diff(y)
            y_lag = y[:-1]

            X = np.column_stack([np.ones(len(dy)), y_lag])
            try:
                beta = np.linalg.lstsq(X, dy, rcond=None)[0]
                residuals = dy - X @ beta
                se_beta = np.sqrt(np.sum(residuals**2) / (len(dy) - 2)) / np.sqrt(np.sum((y_lag - y_lag.mean())**2))
                adf_stat = beta[1] / se_beta
                max_adf = max(max_adf, adf_stat)
            except:
                continue

        bsadf_sequence.append(max_adf)

    # Identify explosive dates
    explosive_dates = []
    if cv is not None:
        for i, bsadf in enumerate(bsadf_sequence):
            if bsadf > cv:
                explosive_dates.append(dates[r0 - 1 + i])

    # Merge contiguous episodes
    if not explosive_dates:
        return []

    episodes = []
    current_start = explosive_dates[0]
    current_end = explosive_dates[0]

    for i in range(1, len(explosive_dates)):
        # Check if contiguous (allowing for weekends/holidays)
        if (explosive_dates[i] - current_end).days <= 5:
            current_end = explosive_dates[i]
        else:
            episodes.append((current_start, current_end))
            current_start = explosive_dates[i]
            current_end = explosive_dates[i]

    episodes.append((current_start, current_end))

    return episodes

# Run GSADF for each ticker
r0_frac = 0.01
num_sims = 2000

results = {}
all_episodes = []

for name, df in [('LT', lt), ('IOC', ioc), ('ITC', itc)]:
    print(f"\nTesting {name}...")
    log_prices = df['LogPrice'].values
    dates = df['Date'].values

    gsadf_stat, cv_95, _, _ = gsadf_test(log_prices, r0_frac, num_sims)
    episodes = dateestamp_bubbles(log_prices, dates, r0_frac, cv_95)

    print(f"  GSADF statistic: {gsadf_stat:.4f}")
    print(f"  5% critical value: {cv_95:.4f}")
    print(f"  Bubble episodes: {len(episodes)}")

    results[name] = {
        'gsadf': gsadf_stat,
        'cv_95': cv_95,
        'episodes': [(str(start), str(end)) for start, end in episodes]
    }

    all_episodes.extend(episodes)

total_episodes = sum(len(results[name]['episodes']) for name in ['LT', 'IOC', 'ITC'])
max_gsadf = max(results[name]['gsadf'] for name in ['LT', 'IOC', 'ITC'])

print(f"\nTotal bubble episodes across all tickers: {total_episodes}")
print(f"Maximum GSADF statistic: {max_gsadf:.4f}")

# ---------- VAR Model and Spillover Analysis ----------
print("\n" + "="*60)
print("VAR MODEL AND SPILLOVER ANALYSIS")
print("="*60)

# Prepare return matrix
returns_matrix = pd.DataFrame({
    'LT': lt['Return'].values,
    'IOC': ioc['Return'].values,
    'ITC': itc['Return'].values
})

# Remove any remaining NaNs
returns_matrix = returns_matrix.dropna()
print(f"Returns sample size: {len(returns_matrix)}")

# VAR lag selection by AIC with stability constraint
from numpy.linalg import eig

def estimate_var(Y, p):
    """Estimate VAR(p) with intercept."""
    T, K = Y.shape

    # Build lagged matrix
    Y_lagged = []
    for i in range(p, T):
        row = [1]  # intercept
        for lag in range(1, p + 1):
            row.extend(Y[i - lag, :])
        Y_lagged.append(row)

    X = np.array(Y_lagged)
    Y_dep = Y[p:, :]

    # OLS estimation
    B = np.linalg.lstsq(X, Y_dep, rcond=None)[0]
    residuals = Y_dep - X @ B
    Sigma = (residuals.T @ residuals) / (len(Y_dep) - X.shape[1])

    return B, Sigma, residuals

def check_stability(B, p, K):
    """Check if companion matrix has all roots inside unit circle."""
    # Build companion matrix
    companion = np.zeros((K * p, K * p))
    companion[:K, :] = B[1:, :].T  # Exclude intercept
    if p > 1:
        companion[K:, :K * (p - 1)] = np.eye(K * (p - 1))

    eigenvalues = eig(companion)[0]
    return np.all(np.abs(eigenvalues) < 1)

def compute_aic(residuals, K, num_params):
    """Compute AIC."""
    T = len(residuals)
    log_det_sigma = np.log(np.linalg.det(residuals.T @ residuals / T))
    aic = log_det_sigma + (2 * num_params) / T
    return aic

# Lag selection
Y = returns_matrix.values
T, K = Y.shape

best_p = 1
best_aic = np.inf

print("\nLag selection (AIC with stability constraint):")
for p in range(1, 11):
    try:
        B, Sigma, residuals = estimate_var(Y, p)

        if check_stability(B, p, K):
            num_params = K * (1 + K * p)  # intercept + K*p coefficients per equation
            aic = compute_aic(residuals, K, num_params)
            print(f"  Lag {p}: AIC = {aic:.6f}, Stable = True")

            if aic < best_aic:
                best_aic = aic
                best_p = p
        else:
            print(f"  Lag {p}: Unstable (roots outside unit circle)")
    except:
        print(f"  Lag {p}: Estimation failed")

print(f"\nSelected lag order: {best_p}")

# Estimate final VAR model
B_final, Sigma_final, residuals_final = estimate_var(Y, best_p)

# ---------- Forecast Error Variance Decomposition ----------
print("\nComputing 10-day FEVD and spillover index...")

def fevd_spillover(B, Sigma, h, K, p):
    """
    Compute forecast error variance decomposition and spillover index.
    """
    # Compute MA(infinity) representation
    # Psi matrices from VAR to VMA
    psi_matrices = []
    psi_0 = np.eye(K)
    psi_matrices.append(psi_0)

    # Get coefficient matrices (exclude intercept)
    A_matrices = []
    for i in range(p):
        A_matrices.append(B[1 + i * K:1 + (i + 1) * K, :].T)

    # Compute Psi matrices up to horizon h
    for i in range(1, h):
        psi_i = np.zeros((K, K))
        for j in range(min(i, p)):
            psi_i += A_matrices[j] @ psi_matrices[i - j - 1]
        psi_matrices.append(psi_i)

    # Cholesky decomposition of covariance matrix
    P = np.linalg.cholesky(Sigma)

    # Compute orthogonalized impulse responses
    theta_matrices = [psi @ P for psi in psi_matrices]

    # Compute FEVD
    fevd = np.zeros((K, K))  # fevd[i, j] = contribution of shock j to variance of variable i

    for i in range(K):
        mse_i = 0
        contributions = np.zeros(K)

        for s in range(h):
            for j in range(K):
                contribution = theta_matrices[s][i, j]**2
                contributions[j] += contribution
                mse_i += contribution

        if mse_i > 0:
            fevd[i, :] = contributions / mse_i * 100

    # Spillover index (Diebold-Yilmaz)
    total_spillover = (np.sum(fevd) - np.trace(fevd)) / np.sum(fevd) * 100

    return fevd, total_spillover

h = 10
fevd_matrix, spillover_index = fevd_spillover(B_final, Sigma_final, h, K, best_p)

print(f"\nForecast Error Variance Decomposition (10-day horizon):")
print(f"Variable decomposition matrix (rows=to, cols=from):")
ticker_names = ['LT', 'IOC', 'ITC']
fevd_df = pd.DataFrame(fevd_matrix, index=ticker_names, columns=ticker_names)
print(fevd_df)

print(f"\nTotal Spillover Index: {spillover_index:.4f}%")

# Compute net spillover for LT
lt_idx = 0
ioc_idx = 1
itc_idx = 2

# LT outgoing to IOC and ITC
lt_to_ioc = fevd_matrix[ioc_idx, lt_idx]
lt_to_itc = fevd_matrix[itc_idx, lt_idx]

# LT incoming from IOC and ITC
ioc_to_lt = fevd_matrix[lt_idx, ioc_idx]
itc_to_lt = fevd_matrix[lt_idx, itc_idx]

lt_net_spillover = (lt_to_ioc + lt_to_itc) - (ioc_to_lt + itc_to_lt)

print(f"\nLT Net Spillover: {lt_net_spillover:.4f}%")

# Average metrics
avg_spillover = spillover_index  # Total spillover is already an average measure
avg_lt_net = lt_net_spillover

# ---------- Yang-Zhang Volatility ----------
print("\n" + "="*60)
print("YANG-ZHANG VOLATILITY ESTIMATION")
print("="*60)

def yang_zhang_volatility(df, window=30, annual_factor=252):
    """
    Compute Yang-Zhang volatility estimator.
    Annualized by multiplying daily variance by 252 and taking sqrt.
    """
    # Need: Open, High, Low, Close
    # Overnight volatility component
    o_c_prev = np.log(df['Open'] / df['Close'].shift(1))

    # Close-to-close volatility
    c_c = np.log(df['Close'] / df['Close'].shift(1))

    # Open-to-close volatility
    o_c = np.log(df['Close'] / df['Open'])

    # Rogers-Satchell component
    h_c = np.log(df['High'] / df['Close'])
    h_o = np.log(df['High'] / df['Open'])
    l_c = np.log(df['Low'] / df['Close'])
    l_o = np.log(df['Low'] / df['Open'])

    rs = h_c * h_o + l_c * l_o

    # Rolling window calculations
    rolling_vol = []

    for i in range(window - 1, len(df)):
        window_data_idx = range(i - window + 1, i + 1)

        # Overnight variance
        overnight = o_c_prev.iloc[window_data_idx]
        sigma_o = overnight.var()

        # Open-to-close variance
        oc = o_c.iloc[window_data_idx]
        sigma_oc = oc.var()

        # Close-to-close variance
        cc = c_c.iloc[window_data_idx]
        sigma_cc = cc.var()

        # Rogers-Satchell variance
        rs_window = rs.iloc[window_data_idx]
        sigma_rs = rs_window.mean()

        # Yang-Zhang estimator
        k = 0.34 / (1.34 + (window + 1) / (window - 1))
        sigma_yz_sq = sigma_o + k * sigma_cc + (1 - k) * sigma_rs

        # Annualize
        sigma_yz_annual = np.sqrt(sigma_yz_sq * annual_factor)
        rolling_vol.append(sigma_yz_annual)

    return rolling_vol

# Compute Yang-Zhang volatility for each ticker
yz_lt = yang_zhang_volatility(lt, window=30, annual_factor=252)
yz_ioc = yang_zhang_volatility(ioc, window=30, annual_factor=252)
yz_itc = yang_zhang_volatility(itc, window=30, annual_factor=252)

median_yz_lt = np.median(yz_lt)
median_yz_ioc = np.median(yz_ioc)
median_yz_itc = np.median(yz_itc)

avg_median_yz = (median_yz_lt + median_yz_ioc + median_yz_itc) / 3

print(f"Median annualized Yang-Zhang volatility:")
print(f"  LT:  {median_yz_lt:.4f}")
print(f"  IOC: {median_yz_ioc:.4f}")
print(f"  ITC: {median_yz_itc:.4f}")
print(f"  Cross-asset average: {avg_median_yz:.4f}")

# ---------- Granger Causality Testing ----------
print("\n" + "="*60)
print("GRANGER CAUSALITY TESTING")
print("="*60)

def granger_causality_test(y, x, max_lag=5, alpha=0.05):
    """
    Test if x Granger-causes y.
    Returns True if any lag 1-5 rejects null at 5% level.
    """
    from scipy.stats import f

    # Align series
    data = pd.DataFrame({'y': y, 'x': x}).dropna()
    y_series = data['y'].values
    x_series = data['x'].values

    for lag in range(1, max_lag + 1):
        # Restricted model: y regressed on its own lags
        Y_dep = y_series[lag:]
        X_restricted = []
        for i in range(lag, len(y_series)):
            row = [1]  # intercept
            for L in range(1, lag + 1):
                row.append(y_series[i - L])
            X_restricted.append(row)
        X_restricted = np.array(X_restricted)

        # Unrestricted model: y regressed on its own lags and x lags
        X_unrestricted = []
        for i in range(lag, len(y_series)):
            row = [1]  # intercept
            for L in range(1, lag + 1):
                row.append(y_series[i - L])
            for L in range(1, lag + 1):
                row.append(x_series[i - L])
            X_unrestricted.append(row)
        X_unrestricted = np.array(X_unrestricted)

        # Fit models
        beta_r = np.linalg.lstsq(X_restricted, Y_dep, rcond=None)[0]
        beta_u = np.linalg.lstsq(X_unrestricted, Y_dep, rcond=None)[0]

        rss_r = np.sum((Y_dep - X_restricted @ beta_r)**2)
        rss_u = np.sum((Y_dep - X_unrestricted @ beta_u)**2)

        # F-test
        T = len(Y_dep)
        k_r = X_restricted.shape[1]
        k_u = X_unrestricted.shape[1]

        f_stat = ((rss_r - rss_u) / (k_u - k_r)) / (rss_u / (T - k_u))
        p_value = 1 - f.cdf(f_stat, k_u - k_r, T - k_u)

        if p_value < alpha:
            return True

    return False

# Test all ordered pairs
pairs = [
    ('LT', 'IOC'), ('LT', 'ITC'),
    ('IOC', 'LT'), ('IOC', 'ITC'),
    ('ITC', 'LT'), ('ITC', 'IOC')
]

returns_dict = {
    'LT': lt['Return'].values,
    'IOC': ioc['Return'].values,
    'ITC': itc['Return'].values
}

significant_pairs = 0
print("\nPairwise Granger causality tests (lags 1-5, alpha=0.05):")

for from_ticker, to_ticker in pairs:
    x = returns_dict[from_ticker]
    y = returns_dict[to_ticker]

    is_significant = granger_causality_test(y, x, max_lag=5, alpha=0.05)
    significant_pairs += int(is_significant)

    print(f"  {from_ticker} -> {to_ticker}: {'Significant' if is_significant else 'Not significant'}")

print(f"\nTotal significant directed pairs: {significant_pairs}")

# ---------- Rolling Lower Tail Co-Exceedance Heatmap ----------
print("\n" + "="*60)
print("ROLLING LOWER TAIL CO-EXCEEDANCE ANALYSIS")
print("="*60)

def compute_co_exceedance(returns1, returns2, window=252, percentile=5):
    """
    Compute fraction of days where both returns are at or below their own 5th percentile.
    """
    if len(returns1) < window or len(returns2) < window:
        return np.nan

    # Compute 5th percentile for each series within window
    p5_1 = np.percentile(returns1, percentile)
    p5_2 = np.percentile(returns2, percentile)

    # Count days where both are at or below their percentiles
    both_exceed = np.sum((returns1 <= p5_1) & (returns2 <= p5_2))

    return both_exceed / len(returns1)

# Get month-end dates for last 24 months
all_dates = lt['Date'].values
last_date = all_dates[-1]
first_date = all_dates[0]

# Find month-end dates
monthly_dates = pd.date_range(start=first_date, end=last_date, freq='M')
monthly_dates = [d for d in monthly_dates if d in all_dates]

# Keep only last 24 months
if len(monthly_dates) > 24:
    monthly_dates = monthly_dates[-24:]

print(f"Analyzing {len(monthly_dates)} month-end dates")

# Stock pairs
stock_pairs = [('LT', 'IOC'), ('LT', 'ITC'), ('IOC', 'ITC')]

# Compute co-exceedance matrix
co_exceedance_matrix = []

for pair in stock_pairs:
    pair_values = []

    for month_end in monthly_dates:
        # Find index of month-end
        try:
            end_idx = np.where(all_dates == month_end)[0][0]
        except:
            pair_values.append(np.nan)
            continue

        # Get preceding 252 days
        start_idx = max(0, end_idx - 251)

        if end_idx - start_idx + 1 < 252:
            pair_values.append(np.nan)
            continue

        # Get returns for window
        ret1 = returns_dict[pair[0]][start_idx:end_idx + 1]
        ret2 = returns_dict[pair[1]][start_idx:end_idx + 1]

        co_exc = compute_co_exceedance(ret1, ret2, window=len(ret1), percentile=5)
        pair_values.append(co_exc)

    co_exceedance_matrix.append(pair_values)

co_exceedance_matrix = np.array(co_exceedance_matrix)

# Find maximum cell value
max_co_exceedance = np.nanmax(co_exceedance_matrix)
print(f"\nMaximum co-exceedance value: {max_co_exceedance:.4f}")

# Create heatmap
fig, ax = plt.subplots(figsize=(14, 5))
im = ax.imshow(co_exceedance_matrix, aspect='auto', cmap='Reds', interpolation='nearest')

# Set labels
ax.set_yticks(range(len(stock_pairs)))
ax.set_yticklabels([f"{p[0]}-{p[1]}" for p in stock_pairs])

# X-axis: show subset of month labels to avoid crowding
x_tick_indices = range(0, len(monthly_dates), max(1, len(monthly_dates) // 12))
ax.set_xticks(x_tick_indices)
ax.set_xticklabels([pd.Timestamp(monthly_dates[i]).strftime('%Y-%m') for i in x_tick_indices], rotation=45, ha='right')

ax.set_xlabel('Month-End Date')
ax.set_ylabel('Stock Pair')
ax.set_title('Rolling 252-Day Lower Tail Co-Exceedance (5th Percentile)')

plt.colorbar(im, ax=ax, label='Co-Exceedance Fraction')
plt.tight_layout()
plt.savefig('co_exceedance_heatmap.png', dpi=150)
print("Heatmap saved to co_exceedance_heatmap.png")

# ---------- Diversification Test (Welch t-test) ----------
print("\n" + "="*60)
print("DIVERSIFICATION TEST")
print("="*60)

# Need last 72 months (36 recent + 36 prior)
if len(monthly_dates) < 72:
    print(f"Warning: Only {len(monthly_dates)} months available, need 72 for 36+36 comparison")

    # Use all available data
    if len(monthly_dates) >= 2:
        mid_point = len(monthly_dates) // 2
        recent_months = monthly_dates[mid_point:]
        prior_months = monthly_dates[:mid_point]
    else:
        recent_months = monthly_dates
        prior_months = []
else:
    recent_months = monthly_dates[-36:]
    prior_months = monthly_dates[-72:-36]

# Compute average co-exceedance for each period
def get_period_avg_coexc(month_list, co_exc_matrix, all_month_dates):
    """Average co-exceedance across pairs for given months."""
    values = []

    for month in month_list:
        try:
            month_idx = all_month_dates.index(month)
            # Average across all three pairs for this month
            month_avg = np.nanmean(co_exc_matrix[:, month_idx])
            if not np.isnan(month_avg):
                values.append(month_avg)
        except:
            continue

    return values

recent_coexc = get_period_avg_coexc(recent_months, co_exceedance_matrix, monthly_dates)
prior_coexc = get_period_avg_coexc(prior_months, co_exceedance_matrix, monthly_dates)

print(f"Recent 36-month period: {len(recent_coexc)} observations")
print(f"  Mean co-exceedance: {np.mean(recent_coexc):.6f}")
print(f"Prior 36-month period: {len(prior_coexc)} observations")
print(f"  Mean co-exceedance: {np.mean(prior_coexc):.6f}")

# Welch t-test (two-sided, unequal variances)
if len(recent_coexc) > 1 and len(prior_coexc) > 1:
    t_stat, p_value = stats.ttest_ind(recent_coexc, prior_coexc, equal_var=False)

    print(f"\nWelch t-test (two-sided):")
    print(f"  t-statistic: {t_stat:.4f}")
    print(f"  p-value: {p_value:.4f}")

    # Check if recent is significantly lower (one-sided interpretation)
    recent_mean = np.mean(recent_coexc)
    prior_mean = np.mean(prior_coexc)

    diversification_improved = int((recent_mean < prior_mean) and (p_value < 0.05))

    print(f"\nDiversification improved (recent < prior, p < 0.05): {diversification_improved}")

    if diversification_improved == 1:
        verdict = "IMPROVED: The recent 36-month period shows significantly lower tail co-exceedance, indicating better diversification benefits during stressed market conditions."
    else:
        if recent_mean >= prior_mean:
            verdict = "DETERIORATED: The recent period shows higher or equal tail co-exceedance, suggesting weaker diversification during market stress."
        else:
            verdict = "INCONCLUSIVE: Although recent co-exceedance is lower, the difference is not statistically significant at the 5% level."
else:
    diversification_improved = 0
    verdict = "INSUFFICIENT DATA: Cannot perform Welch t-test with available sample."
    print(verdict)

print(f"\nDiversification verdict: {verdict}")

# ---------- Summary Output ----------
print("\n" + "="*60)
print("FINAL RESULTS SUMMARY")
print("="*60)

print(f"\n1. Total bubble episodes: {total_episodes}")
print(f"2. Maximum GSADF statistic: {max_gsadf:.4f}")
print(f"3. Average total spillover index: {avg_spillover:.4f}%")
print(f"4. Average LT net spillover: {avg_lt_net:.4f}%")
print(f"5. Cross-asset average median Yang-Zhang volatility: {avg_median_yz:.4f}")
print(f"6. Total significant Granger causality pairs: {significant_pairs}")
print(f"7. Maximum heatmap cell value: {max_co_exceedance:.4f}")
print(f"8. Diversification improved indicator: {diversification_improved}")
print(f"9. Diversification verdict: {verdict}")

# Save results to JSON
output = {
    "total_bubble_episodes": int(total_episodes),
    "max_gsadf_statistic": round(float(max_gsadf), 4),
    "avg_total_spillover_index": round(float(avg_spillover), 4),
    "avg_lt_net_spillover": round(float(avg_lt_net), 4),
    "cross_asset_avg_median_yang_zhang_volatility": round(float(avg_median_yz), 4),
    "total_significant_granger_pairs": int(significant_pairs),
    "max_coexceedance_heatmap_value": round(float(max_co_exceedance), 4),
    "diversification_improved": int(diversification_improved),
    "diversification_verdict": verdict,
    "bubble_details": results,
    "fevd_matrix": fevd_df.to_dict()
}

with open('analysis_results.json', 'w') as f:
    json.dump(output, f, indent=2)

print("\nResults saved to analysis_results.json")
print("Analysis complete!")
