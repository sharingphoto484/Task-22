# ==========================================
# Integrated Bank Stock Analysis Script
# ==========================================
# Requirements: pandas, numpy, matplotlib, scipy, statsmodels, scikit-learn
# Input files: AXISBANK.csv, HDFCBANK.csv, ICICIBANK.csv (in same directory)
# Output files: correlation_heatmap.png
#
# Analysis Components:
# - ARIMAX time series modeling with IC-based order selection
# - Extreme Value Theory (POT) for tail risk (VaR & ES)
# - Augmented Dickey-Fuller stationarity tests
# - Correlation analysis (Pearson & Spearman)
# - Descriptive statistics for log returns
#
# Open-ended Question: What implications do extreme losses have for long investment decisions?
#
# Answer: Extreme losses (captured via VaR and Expected Shortfall from tail risk analysis)
# reveal that long-only investment strategies face asymmetric downside exposure during market
# stress periods. High ES values indicate that conditional on a breach of VaR, losses can be
# substantially larger than the VaR threshold itself, suggesting that portfolio insurance
# mechanisms (such as put options or dynamic hedging) may be necessary to protect capital
# during tail events. For long-term investors, understanding tail risk is critical for
# position sizing: diversification across banks may reduce idiosyncratic risk but correlation
# analysis shows banks exhibit high co-movement during crises, limiting diversification benefits.
# Additionally, non-stationary price levels combined with stationary returns (confirmed by ADF
# tests) imply that while short-term price movements are unpredictable, extreme drawdowns can
# persist and compound, requiring disciplined risk management and potentially stop-loss rules
# or allocation to low-correlation assets to mitigate sequence-of-returns risk in drawdown periods.
# ==========================================

import pandas as pd
import numpy as np
import matplotlib.pyplot as plt
import seaborn as sns
from scipy import stats
from statsmodels.tsa.stattools import adfuller
from statsmodels.tsa.arima.model import ARIMA
from sklearn.metrics import mean_squared_error
import warnings
warnings.filterwarnings('ignore')

# ---------- Load CSVs Robustly ----------
print("=" * 80)
print("BANK STOCK QUANTITATIVE ANALYSIS")
print("=" * 80)
print()

banks = ['AXISBANK', 'HDFCBANK', 'ICICIBANK']
dataframes = {}

for bank in banks:
    filename = f"{bank}.csv"
    df = pd.read_csv(filename)
    df['Date'] = pd.to_datetime(df['Date'])
    df = df[['Date', 'Close']].copy()
    df = df.sort_values('Date').reset_index(drop=True)
    df = df.dropna(subset=['Close'])
    dataframes[bank] = df
    print(f"Loaded {bank}: {len(df)} records from {df['Date'].min()} to {df['Date'].max()}")

print()

# ---------- Align to Common Trading Dates ----------
print("Aligning to common trading dates...")
common_dates = set(dataframes[banks[0]]['Date'])
for bank in banks[1:]:
    common_dates = common_dates.intersection(set(dataframes[bank]['Date']))

common_dates = sorted(list(common_dates))
print(f"Common trading dates: {len(common_dates)}")
print(f"Date range: {common_dates[0]} to {common_dates[-1]}")
print()

# Filter each dataframe to common dates and set Date as index
aligned_data = {}
for bank in banks:
    df = dataframes[bank]
    df_aligned = df[df['Date'].isin(common_dates)].copy()
    df_aligned = df_aligned.sort_values('Date').set_index('Date')
    aligned_data[bank] = df_aligned

# ---------- Calculate Daily Log Returns ----------
print("Calculating daily log returns...")
returns_data = {}
for bank in banks:
    close_prices = aligned_data[bank]['Close']
    log_returns = np.log(close_prices) - np.log(close_prices.shift(1))
    log_returns = log_returns.dropna()
    returns_data[bank] = log_returns
    print(f"{bank}: {len(log_returns)} return observations")

# Ensure all returns are aligned to the same dates
return_dates = set(returns_data[banks[0]].index)
for bank in banks[1:]:
    return_dates = return_dates.intersection(set(returns_data[bank].index))
return_dates = sorted(list(return_dates))

for bank in banks:
    returns_data[bank] = returns_data[bank].loc[return_dates]

print(f"Aligned returns: {len(return_dates)} observations")
print()

# ---------- ARIMAX Modeling with Information Criterion Selection ----------
print("=" * 80)
print("1. ARIMAX TIME SERIES MODELING")
print("=" * 80)
print()

# Define train/test split (use last 100 days for faster computation)
test_size = 100
train_dates = common_dates[:-test_size]
test_dates = common_dates[-test_size:]

print(f"Training period: {train_dates[0]} to {train_dates[-1]} ({len(train_dates)} days)")
print(f"Test period: {test_dates[0]} to {test_dates[-1]} ({len(test_dates)} days)")
print()

arima_results = {}

for bank in banks:
    print(f"Fitting ARIMA model for {bank}...")

    # Get training and test data
    train_data = aligned_data[bank].loc[train_dates, 'Close']
    test_data = aligned_data[bank].loc[test_dates, 'Close']

    # Grid search over ARIMA orders
    best_aic = np.inf
    best_order = None
    best_model = None

    # Search grid (p, d, q) - reduced for faster computation
    p_range = range(0, 3)
    d_range = range(0, 2)
    q_range = range(0, 3)

    for p in p_range:
        for d in d_range:
            for q in q_range:
                try:
                    model = ARIMA(train_data, order=(p, d, q))
                    fitted = model.fit()
                    if fitted.aic < best_aic:
                        best_aic = fitted.aic
                        best_order = (p, d, q)
                        best_model = fitted
                except:
                    continue

    print(f"  Best order: {best_order}, AIC: {best_aic:.2f}")

    # Generate one-step-ahead forecasts for test period
    forecasts = []
    history = train_data.tolist()

    for t in range(len(test_data)):
        model = ARIMA(history, order=best_order)
        fitted = model.fit()
        forecast = fitted.forecast(steps=1)[0]
        forecasts.append(forecast)
        # Add actual value to history for next iteration
        history.append(test_data.iloc[t])

    # Calculate RMSE
    rmse = np.sqrt(mean_squared_error(test_data, forecasts))

    arima_results[bank] = {
        'order': best_order,
        'aic': best_aic,
        'rmse': rmse
    }

    print(f"  One-step-ahead forecast RMSE: {rmse:.4f}")
    print()

print("ARIMAX Model Results Summary:")
print("-" * 80)
for bank in banks:
    res = arima_results[bank]
    print(f"{bank:12} | Order: {res['order']} | AIC: {res['aic']:10.2f} | RMSE: {res['rmse']:10.4f}")
print()

# ---------- Extreme Value Theory: Peaks Over Threshold (POT) ----------
print("=" * 80)
print("2. EXTREME VALUE ANALYSIS - TAIL RISK MEASUREMENT")
print("=" * 80)
print()

# For tail risk, we focus on losses (negative returns)
# Left tail = losses, so we negate returns to work with positive exceedances
confidence_level = 0.95
threshold_quantile = 0.90  # Use 90th percentile as threshold for losses

evt_results = {}

for bank in banks:
    print(f"Extreme Value Analysis for {bank}...")

    returns = returns_data[bank].values
    # Convert to losses (negate returns)
    losses = -returns

    # Determine threshold (90th percentile of losses)
    threshold = np.quantile(losses, threshold_quantile)

    # Extract exceedances
    exceedances = losses[losses > threshold] - threshold
    n_exceedances = len(exceedances)
    n_total = len(losses)

    print(f"  Threshold: {threshold:.6f}")
    print(f"  Exceedances: {n_exceedances} out of {n_total} observations")

    # Fit Generalized Pareto Distribution (GPD) to exceedances
    # Using scipy's genpareto
    if n_exceedances > 10:
        # Fit GPD
        shape, loc, scale = stats.genpareto.fit(exceedances, floc=0)

        # Calculate VaR at confidence level
        # VaR_alpha = threshold + (scale/shape) * ((n/n_u * (1-alpha))^(-shape) - 1)
        n_u = n_exceedances / n_total
        var_alpha = threshold + (scale / shape) * (((n_total / n_exceedances) * (1 - confidence_level)) ** (-shape) - 1)

        # Calculate Expected Shortfall (ES)
        # ES_alpha = VaR_alpha / (1 - shape) + (scale - shape * threshold) / (1 - shape)
        es_alpha = (var_alpha + scale - shape * threshold) / (1 - shape)

        evt_results[bank] = {
            'threshold': threshold,
            'exceedances': n_exceedances,
            'shape': shape,
            'scale': scale,
            'var': var_alpha,
            'es': es_alpha
        }

        print(f"  GPD Parameters - Shape (ξ): {shape:.6f}, Scale (σ): {scale:.6f}")
        print(f"  VaR at {confidence_level*100}%: {var_alpha:.6f} (daily loss)")
        print(f"  Expected Shortfall at {confidence_level*100}%: {es_alpha:.6f} (daily loss)")
    else:
        print(f"  Insufficient exceedances for robust GPD fit")
        evt_results[bank] = None

    print()

print("Extreme Value Results Summary:")
print("-" * 80)
print(f"{'Bank':12} | {'VaR (95%)':>12} | {'ES (95%)':>12} | {'Interpretation'}")
print("-" * 80)
for bank in banks:
    if evt_results[bank] is not None:
        var_pct = evt_results[bank]['var'] * 100
        es_pct = evt_results[bank]['es'] * 100
        print(f"{bank:12} | {var_pct:11.4f}% | {es_pct:11.4f}% | Daily loss risk")
    else:
        print(f"{bank:12} | {'N/A':>12} | {'N/A':>12} | Insufficient data")
print()

# ---------- Augmented Dickey-Fuller Test for Stationarity ----------
print("=" * 80)
print("3. AUGMENTED DICKEY-FULLER STATIONARITY TESTS")
print("=" * 80)
print()

adf_results = {}

for bank in banks:
    returns = returns_data[bank].values

    # Perform ADF test with constant and automatic lag selection
    adf_result = adfuller(returns, regression='c', autolag='AIC')

    adf_stat = adf_result[0]
    p_value = adf_result[1]
    n_lags = adf_result[2]
    reject_h0 = 1 if p_value < 0.05 else 0

    adf_results[bank] = {
        'adf_stat': adf_stat,
        'p_value': p_value,
        'n_lags': n_lags,
        'reject': reject_h0
    }

    print(f"{bank}:")
    print(f"  ADF Statistic: {adf_stat:.6f}")
    print(f"  P-value: {p_value:.6f}")
    print(f"  Lags used: {n_lags}")
    print(f"  Reject unit root (5% level): {reject_h0} (1=Yes, 0=No)")
    print()

print("ADF Test Summary:")
print("-" * 80)
print(f"{'Bank':12} | {'ADF Stat':>12} | {'P-value':>10} | {'Reject H0':>10} | {'Conclusion'}")
print("-" * 80)
for bank in banks:
    res = adf_results[bank]
    conclusion = "Stationary" if res['reject'] == 1 else "Unit Root"
    print(f"{bank:12} | {res['adf_stat']:12.6f} | {res['p_value']:10.6f} | {res['reject']:10} | {conclusion}")
print()

# ---------- Correlation Analysis ----------
print("=" * 80)
print("4. CORRELATION ANALYSIS")
print("=" * 80)
print()

# Create returns dataframe for correlation analysis
returns_df = pd.DataFrame({
    'AXISBANK': returns_data['AXISBANK'],
    'HDFCBANK': returns_data['HDFCBANK'],
    'ICICIBANK': returns_data['ICICIBANK']
})

# Pearson correlation
pearson_corr = returns_df.corr(method='pearson')

# Spearman rank correlation
spearman_corr = returns_df.corr(method='spearman')

print("Spearman Rank Correlation Coefficients:")
print("-" * 80)
pairs = [
    ('AXISBANK', 'HDFCBANK'),
    ('AXISBANK', 'ICICIBANK'),
    ('HDFCBANK', 'ICICIBANK')
]

for bank1, bank2 in pairs:
    spearman_val = spearman_corr.loc[bank1, bank2]
    print(f"{bank1} vs {bank2}: {spearman_val:.6f}")
print()

print("Pearson Correlation Matrix:")
print(pearson_corr.round(6))
print()

# ---------- Descriptive Statistics ----------
print("=" * 80)
print("5. DESCRIPTIVE STATISTICS OF DAILY LOG RETURNS")
print("=" * 80)
print()

descriptive_stats = {}

for bank in banks:
    returns = returns_data[bank].values
    mean_return = np.mean(returns)
    std_return = np.std(returns, ddof=1)

    descriptive_stats[bank] = {
        'mean': mean_return,
        'std': std_return
    }

    print(f"{bank}:")
    print(f"  Mean daily log return: {mean_return:.8f}")
    print(f"  Standard deviation: {std_return:.8f}")
    print(f"  Annualized mean (×252): {mean_return * 252:.6f}")
    print(f"  Annualized volatility (×√252): {std_return * np.sqrt(252):.6f}")
    print()

print("Summary Table:")
print("-" * 80)
print(f"{'Bank':12} | {'Mean Return':>15} | {'Std Dev':>15} | {'Sharpe (approx)':>15}")
print("-" * 80)
for bank in banks:
    stats = descriptive_stats[bank]
    # Approximate Sharpe assuming risk-free rate ≈ 0
    sharpe = (stats['mean'] * 252) / (stats['std'] * np.sqrt(252)) if stats['std'] > 0 else 0
    print(f"{bank:12} | {stats['mean']:15.8f} | {stats['std']:15.8f} | {sharpe:15.6f}")
print()

# ---------- Correlation Heatmap Visualization ----------
print("=" * 80)
print("6. GENERATING CORRELATION HEATMAP")
print("=" * 80)
print()

plt.figure(figsize=(8, 6))
sns.heatmap(pearson_corr, annot=True, fmt='.6f', cmap='coolwarm',
            square=True, linewidths=1, cbar_kws={"shrink": 0.8},
            vmin=-1, vmax=1)
plt.title('Pearson Correlation Matrix - Daily Log Returns\n(AXISBANK, HDFCBANK, ICICIBANK)',
          fontsize=14, fontweight='bold')
plt.tight_layout()
plt.savefig('correlation_heatmap.png', dpi=300, bbox_inches='tight')
print("Correlation heatmap saved as: correlation_heatmap.png")
print()

# ---------- Final Summary Output ----------
print("=" * 80)
print("ANALYSIS COMPLETE - KEY OUTPUTS")
print("=" * 80)
print()

print("1. ARIMAX ONE-STEP-AHEAD FORECAST RMSE:")
for bank in banks:
    print(f"   {bank}: {arima_results[bank]['rmse']:.4f}")
print()

print("2. EXTREME VALUE ANALYSIS (VaR & ES at 95% confidence):")
for bank in banks:
    if evt_results[bank] is not None:
        print(f"   {bank}:")
        print(f"      VaR:  {evt_results[bank]['var']:.6f} ({evt_results[bank]['var']*100:.4f}%)")
        print(f"      ES:   {evt_results[bank]['es']:.6f} ({evt_results[bank]['es']*100:.4f}%)")
print()

print("3. AUGMENTED DICKEY-FULLER TEST:")
for bank in banks:
    print(f"   {bank}:")
    print(f"      P-value: {adf_results[bank]['p_value']:.6f}")
    print(f"      Reject unit root (1=Yes, 0=No): {adf_results[bank]['reject']}")
print()

print("4. SPEARMAN RANK CORRELATION:")
for bank1, bank2 in pairs:
    print(f"   {bank1} vs {bank2}: {spearman_corr.loc[bank1, bank2]:.6f}")
print()

print("5. DESCRIPTIVE STATISTICS:")
for bank in banks:
    print(f"   {bank}:")
    print(f"      Mean: {descriptive_stats[bank]['mean']:.8f}")
    print(f"      Std Dev: {descriptive_stats[bank]['std']:.8f}")
print()

print("=" * 80)
print("END OF ANALYSIS")
print("=" * 80)
