# ==========================================
# Integrated Stock Cointegration & Volatility Analysis Script
# ==========================================
# Requirements: pandas, numpy, matplotlib, scipy, statsmodels
# Input files: EICHERMOT.csv, HCLTECH.csv, BPCL.csv (in same directory)
# Output files: analysis_output.json (key outputs only)
#
# Analysis includes:
# - Johansen cointegration test with VECM
# - Two-state Markov switching volatility models
# - Rogers-Satchell volatility estimation
# - Seasonality analysis (day-of-week, rolling heatmap)
# - Drift assessment with Newey-West standard errors
# ==========================================

import pandas as pd
import numpy as np
import json
import warnings
from scipy import stats
from statsmodels.tsa.vector_ar.vecm import coint_johansen, VECM
from statsmodels.regression.linear_model import OLS
from statsmodels.tsa.regime_switching.markov_regression import MarkovRegression
import matplotlib
matplotlib.use('Agg')
import matplotlib.pyplot as plt

warnings.filterwarnings('ignore')

# ---------- Load CSVs Robustly ----------
def load_and_standardize(filepath, ticker_name):
    """
    Load CSV and standardize to required columns: Date, Open, High, Low, Close, Volume
    """
    df = pd.read_csv(filepath)

    # Select and rename to standard format
    df_std = pd.DataFrame({
        'Date': pd.to_datetime(df['Date'], errors='coerce'),
        'Open': pd.to_numeric(df['Open'], errors='coerce'),
        'High': pd.to_numeric(df['High'], errors='coerce'),
        'Low': pd.to_numeric(df['Low'], errors='coerce'),
        'Close': pd.to_numeric(df['Close'], errors='coerce'),
        'Volume': pd.to_numeric(df['Volume'], errors='coerce')
    })

    # Remove rows with missing Date
    df_std = df_std.dropna(subset=['Date'])

    # Sort by Date
    df_std = df_std.sort_values('Date')

    # Keep last record when duplicates occur
    df_std = df_std.drop_duplicates(subset=['Date'], keep='last')

    # Set Date as index
    df_std.set_index('Date', inplace=True)

    # Add ticker column
    df_std['Ticker'] = ticker_name

    return df_std

print("Loading data files...")
eicher = load_and_standardize('EICHERMOT.csv', 'EICHERMOT')
hcl = load_and_standardize('HCLTECH.csv', 'HCLTECH')
bpcl = load_and_standardize('BPCL.csv', 'BPCL')

# ---------- Form Exact Daily Intersection ----------
print("Computing exact daily intersection...")
common_dates = eicher.index.intersection(hcl.index).intersection(bpcl.index)
print(f"Common dates: {len(common_dates)}")

eicher = eicher.loc[common_dates]
hcl = hcl.loc[common_dates]
bpcl = bpcl.loc[common_dates]

# ---------- Compute Log Returns ----------
def compute_log_returns(df):
    """Compute first difference of natural log of Close"""
    df['LogClose'] = np.log(df['Close'])
    df['Return'] = df['LogClose'].diff()
    return df

eicher = compute_log_returns(eicher)
hcl = compute_log_returns(hcl)
bpcl = compute_log_returns(bpcl)

# Remove first row (NaN return)
eicher = eicher.iloc[1:]
hcl = hcl.iloc[1:]
bpcl = bpcl.iloc[1:]

# ---------- Johansen Cointegration Test ----------
print("\nPerforming Johansen cointegration test...")

def johansen_cointegration_analysis():
    """
    Estimate Johansen procedure on ln(Close) for three tickers
    - Constant in cointegrating relation
    - Lag selection by Schwarz criterion (BIC) over 1-5
    - 5% trace test for rank selection
    - If rank > 0, fit VECM and extract adjustment speeds
    """
    # Create matrix of log prices
    log_prices = pd.DataFrame({
        'EICHERMOT': eicher['LogClose'],
        'HCLTECH': hcl['LogClose'],
        'BPCL': bpcl['LogClose']
    })

    # Remove any NaN
    log_prices = log_prices.dropna()

    if len(log_prices) < 10:
        return 0, 0

    # Lag selection by BIC over lags 1-5
    best_lag = 1
    best_bic = np.inf

    for lag in range(1, 6):
        if len(log_prices) < lag + 10:
            continue
        try:
            result = coint_johansen(log_prices, det_order=0, k_ar_diff=lag)
            # Compute BIC approximation (using trace statistic and sample size)
            n = len(log_prices)
            bic = -2 * (-result.trace_stat[0]) + np.log(n) * (3 * lag)
            if bic < best_bic:
                best_bic = bic
                best_lag = lag
        except:
            continue

    # Perform Johansen with selected lag
    try:
        result = coint_johansen(log_prices, det_order=0, k_ar_diff=best_lag)

        # 5% critical values for trace test
        trace_stat = result.trace_stat
        trace_crit_5pct = result.trace_stat_crit_vals[:, 1]  # 5% critical values

        # Determine rank r
        r = 0
        for i in range(len(trace_stat)):
            if trace_stat[i] > trace_crit_5pct[i]:
                r = i + 1
            else:
                break

        if r == 0:
            return 0, 0

        # Fit VECM with selected rank and lag
        vecm = VECM(log_prices, k_ar_diff=best_lag, coint_rank=r, deterministic='ci')
        vecm_fit = vecm.fit()

        # Extract adjustment speeds (alpha coefficients)
        # These are the coefficients on the error correction terms in each return equation
        alpha = vecm_fit.alpha  # Shape: (n_equations, r)

        # Count significant negative adjustment speeds
        # We need to test if each alpha coefficient is significantly negative at 5%
        # Standard errors are in vecm_fit.stderr_alpha
        stderr_alpha = vecm_fit.stderr_alpha
        t_stats = alpha / stderr_alpha

        # Critical value for 5% one-sided test (negative direction)
        t_crit = -1.645  # Approximate

        count_sig_neg = 0
        for i in range(3):  # Three equations (one per ticker)
            # Check if any of the r error correction terms has significant negative coefficient
            for j in range(r):
                if t_stats[i, j] < t_crit:
                    count_sig_neg += 1
                    break  # Count ticker only once

        return r, count_sig_neg

    except Exception as e:
        print(f"Johansen error: {e}")
        return 0, 0

coint_rank, count_sig_neg_adj = johansen_cointegration_analysis()

# ---------- Markov Switching Volatility Models ----------
print("\nFitting Markov switching models...")

def fit_markov_switching(returns, ticker_name):
    """
    Fit two-state Gaussian Markov switching model
    - State-specific variances
    - MLE estimation
    - Kim smoother for smoothed probabilities
    """
    returns_clean = returns.dropna()

    if len(returns_clean) < 50:
        return (0.0, 0.0)

    try:
        # Fit 2-state Markov switching model
        mod = MarkovRegression(
            returns_clean,
            k_regimes=2,
            switching_variance=True,
            trend='c'
        )
        res = mod.fit(maxiter=500, disp=False, search_reps=10)

        # Identify high volatility state (larger variance)
        variances = [res.params[f'sigma2[{i}]'] for i in range(2)]
        high_vol_state = np.argmax(variances)

        # Transition matrix
        trans_mat = res.regime_transition

        # Probability of staying in high volatility state
        p_stay = trans_mat[high_vol_state, high_vol_state]

        # Expected duration = 1 / (1 - p_stay)
        if p_stay < 1.0:
            exp_duration = 1.0 / (1.0 - p_stay)
        else:
            exp_duration = 0.0

        # Stationary probability
        # For 2-state model: pi = [p01/(p01+p10), p10/(p01+p10)]
        p01 = trans_mat[0, 1]
        p10 = trans_mat[1, 0]
        if (p01 + p10) > 0:
            stationary_probs = np.array([p01/(p01+p10), p10/(p01+p10)])
            stat_prob_high = stationary_probs[high_vol_state]
        else:
            stat_prob_high = 0.0

        return (float(exp_duration), float(stat_prob_high))

    except Exception as e:
        print(f"Markov switching error for {ticker_name}: {e}")
        return (0.0, 0.0)

# Fit for each ticker
ms_eicher = fit_markov_switching(eicher['Return'], 'EICHERMOT')
ms_hcl = fit_markov_switching(hcl['Return'], 'HCLTECH')
ms_bpcl = fit_markov_switching(bpcl['Return'], 'BPCL')

# Cross-asset averages
avg_exp_duration = np.mean([ms_eicher[0], ms_hcl[0], ms_bpcl[0]])
avg_stat_prob = np.mean([ms_eicher[1], ms_hcl[1], ms_bpcl[1]])

# ---------- Rogers-Satchell Volatility ----------
print("\nComputing Rogers-Satchell volatility...")

def rogers_satchell_volatility(df, window=30):
    """
    Compute Rogers-Satchell daily variance using High, Low, Open, Close
    RS = sqrt[ln(H/C)*ln(H/O) + ln(L/C)*ln(L/O)]
    """
    H = df['High']
    L = df['Low']
    O = df['Open']
    C = df['Close']

    # RS variance (daily)
    rs_var = np.log(H/C) * np.log(H/O) + np.log(L/C) * np.log(L/O)

    # Rolling 30-day average variance
    rolling_var = rs_var.rolling(window=window).mean()

    # Annualize: multiply variance by 252, then take sqrt
    annualized_vol = np.sqrt(rolling_var * 252)

    return annualized_vol

rs_eicher = rogers_satchell_volatility(eicher)
rs_hcl = rogers_satchell_volatility(hcl)
rs_bpcl = rogers_satchell_volatility(bpcl)

# Median for each ticker, then average across tickers
median_rs_eicher = rs_eicher.median()
median_rs_hcl = rs_hcl.median()
median_rs_bpcl = rs_bpcl.median()

avg_median_rs_vol = np.mean([median_rs_eicher, median_rs_hcl, median_rs_bpcl])

# ---------- Day of Week Seasonality Test ----------
print("\nTesting day-of-week seasonality...")

def test_day_of_week_seasonality():
    """
    One-way ANOVA test for day-of-week effects in returns
    """
    # Combine returns with day of week
    eicher_df = eicher[['Return']].copy()
    eicher_df['Date'] = eicher.index

    hcl_df = hcl[['Return']].copy()
    hcl_df['Date'] = hcl.index

    bpcl_df = bpcl[['Return']].copy()
    bpcl_df['Date'] = bpcl.index

    returns_combined = pd.concat([eicher_df, hcl_df, bpcl_df], ignore_index=True)

    returns_combined['DayOfWeek'] = returns_combined['Date'].dt.dayofweek  # 0=Mon, 4=Fri

    # Filter to Monday-Friday only
    returns_combined = returns_combined[returns_combined['DayOfWeek'] <= 4]
    returns_combined = returns_combined.dropna()

    # Group by day of week
    groups = [returns_combined[returns_combined['DayOfWeek'] == i]['Return'].values
              for i in range(5)]

    # Remove empty groups
    groups = [g for g in groups if len(g) > 0]

    if len(groups) < 2:
        return 0

    # One-way ANOVA
    f_stat, p_value = stats.f_oneway(*groups)

    return 1 if p_value < 0.05 else 0

dow_seasonality_indicator = test_day_of_week_seasonality()

# ---------- Rolling Seasonality Heatmap ----------
print("\nComputing rolling seasonality heatmap...")

def rolling_seasonality_heatmap():
    """
    Create heatmap of median returns by month and day-of-week
    - X-axis: Most recent 24 months in common sample
    - Y-axis: Monday-Friday
    - Each cell: Median daily return (%) for that month/weekday on preceding 252 trading days
    """
    # Combine all returns with dates
    all_returns = pd.DataFrame({
        'Date': list(eicher.index) + list(hcl.index) + list(bpcl.index),
        'Return': list(eicher['Return']) + list(hcl['Return']) + list(bpcl['Return'])
    })
    all_returns = all_returns.dropna()
    all_returns['DayOfWeek'] = all_returns['Date'].dt.dayofweek

    # Get last date and find last 24 months
    last_date = all_returns['Date'].max()
    start_date_24m = last_date - pd.DateOffset(months=24)

    recent_data = all_returns[all_returns['Date'] >= start_date_24m].copy()

    if len(recent_data) == 0:
        return 0.0

    # Get unique year-months in the period
    recent_data['YearMonth'] = recent_data['Date'].dt.to_period('M')
    unique_months = sorted(recent_data['YearMonth'].unique())

    heatmap_values = []

    for month in unique_months:
        month_end = month.to_timestamp('M')

        # Get 252 trading days before month end
        data_before_month = all_returns[all_returns['Date'] <= month_end].tail(252)

        if len(data_before_month) < 10:
            continue

        # Compute median return for each day of week (0-4)
        for dow in range(5):
            dow_data = data_before_month[data_before_month['DayOfWeek'] == dow]['Return']
            if len(dow_data) > 0:
                median_return_pct = dow_data.median() * 100  # Convert to percentage
                heatmap_values.append(median_return_pct)

    if len(heatmap_values) == 0:
        return 0.0

    return max(heatmap_values)

max_heatmap_value = rolling_seasonality_heatmap()

# ---------- Drift Assessment with Newey-West ----------
print("\nPerforming drift assessment...")

def drift_assessment_newey_west():
    """
    Test if equal-weight average daily return > 0
    - One-sample t-test with Newey-West standard errors (lag 5)
    """
    # Equal weight average daily return across three tickers
    avg_return = (eicher['Return'] + hcl['Return'] + bpcl['Return']) / 3.0
    avg_return = avg_return.dropna()

    if len(avg_return) < 10:
        return 0

    # Mean return
    mean_return = avg_return.mean()

    if mean_return <= 0:
        return 0

    # Newey-West standard error
    # Use OLS regression with constant only
    y = avg_return.values
    X = np.ones((len(y), 1))

    model = OLS(y, X)
    # Fit with HAC (Newey-West) covariance, lag 5
    results = model.fit(cov_type='HAC', cov_kwds={'maxlags': 5})

    # t-statistic (from HAC-robust results)
    t_stat = results.tvalues[0]

    # One-sided p-value (test if mean > 0)
    p_value = 1 - stats.t.cdf(t_stat, df=len(y)-1)

    return 1 if (mean_return > 0 and p_value < 0.05) else 0

drift_indicator = drift_assessment_newey_west()

# ---------- Output Key Results ----------
print("\n" + "="*60)
print("KEY OUTPUTS")
print("="*60)

output = {
    "cointegration_rank": int(coint_rank),
    "count_significant_negative_adjustment_speed": int(count_sig_neg_adj),
    "avg_expected_duration_high_vol_state_days": round(avg_exp_duration, 4),
    "avg_stationary_prob_high_vol_state": round(avg_stat_prob, 4),
    "avg_median_annualized_rogers_satchell_vol": round(avg_median_rs_vol, 4),
    "day_of_week_seasonality_indicator": int(dow_seasonality_indicator),
    "max_rolling_seasonality_heatmap_value": round(max_heatmap_value, 4),
    "drift_assessment_indicator": int(drift_indicator)
}

for key, value in output.items():
    print(f"{key}: {value}")

# Save to JSON
with open('analysis_output.json', 'w') as f:
    json.dump(output, f, indent=2)

print("\nAnalysis complete. Results saved to analysis_output.json")
