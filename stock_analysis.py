# ==========================================
# Integrated Stock Market Analysis Script
# ==========================================
# Requirements: pandas, numpy, matplotlib, scipy, statsmodels
# Input files: AKAM.csv, AIZ.csv, AES.csv (in same directory)
# Output files: Multiple visualization PNG files
#
# This script performs comprehensive quantitative analysis on three
# stock files including ARIMA forecasting, correlation analysis,
# volatility measurements, and regression studies.
# ==========================================

import pandas as pd
import numpy as np
import matplotlib.pyplot as plt
from scipy import stats
from scipy.stats import spearmanr
from statsmodels.tsa.arima.model import ARIMA
from sklearn.linear_model import LinearRegression
import warnings
warnings.filterwarnings('ignore')

# ---------- Load CSVs Robustly ----------
def load_and_clean(filepath):
    """Load CSV and clean data according to requirements"""
    df = pd.read_csv(filepath)

    # Rename Close to Adj Close if needed
    if 'Adj Close' not in df.columns and 'Close' in df.columns:
        df['Adj Close'] = df['Close']

    # Standardize date format
    df['Date'] = pd.to_datetime(df['Date'], errors='coerce')

    # Required columns
    required_cols = ['Date', 'Open', 'High', 'Low', 'Close', 'Adj Close', 'Volume']

    # Remove rows with missing dates
    df = df.dropna(subset=['Date'])

    # Remove rows with missing or non-numeric values in required columns (except Date)
    numeric_cols = [col for col in required_cols if col != 'Date']
    for col in numeric_cols:
        if col in df.columns:
            df[col] = pd.to_numeric(df[col], errors='coerce')

    df = df.dropna(subset=[col for col in numeric_cols if col in df.columns])

    # Sort by date
    df = df.sort_values('Date').reset_index(drop=True)

    return df

print("Loading data files...")
akam = load_and_clean('AKAM.csv')
aiz = load_and_clean('AIZ.csv')
aes = load_and_clean('AES.csv')

# ---------- Calculate Daily Returns ----------
def calculate_returns(df):
    """Calculate log returns excluding Volume=0 days"""
    df = df.copy()
    # Calculate log returns
    df['Return'] = np.log(df['Adj Close']) - np.log(df['Adj Close'].shift(1))
    # Mark valid returns (Volume > 0 and not NaN)
    df['Valid_Return'] = (df['Volume'] > 0) & (~df['Return'].isna())
    return df

akam = calculate_returns(akam)
aiz = calculate_returns(aiz)
aes = calculate_returns(aes)

# ---------- ARIMA(1,1,1) Model for AKAM ----------
print("\n=== ARIMA(1,1,1) Forecast for AKAM ===")
akam_clean = akam.dropna(subset=['Adj Close'])
model = ARIMA(akam_clean['Adj Close'].values, order=(1, 1, 1))
fitted_model = model.fit()
forecast = fitted_model.forecast(steps=30)
mean_forecast = np.mean(forecast)
print(f"Mean 30-day forecast: {mean_forecast:.2f}")

# Plot AKAM Adjusted Close
plt.figure(figsize=(12, 6))
plt.plot(akam_clean['Date'], akam_clean['Adj Close'], linewidth=1, color='steelblue')
plt.xlabel('Date')
plt.ylabel('Adjusted Close Price')
plt.title('AKAM Adjusted Closing Price Over Time')
plt.grid(True, alpha=0.3)
plt.tight_layout()
plt.savefig('akam_adj_close.png', dpi=300)
plt.close()
print("Generated: akam_adj_close.png")

# ---------- Spearman Correlation (AIZ vs AES Returns) ----------
print("\n=== Spearman Correlation (AIZ vs AES) ===")
# Filter valid returns
aiz_valid = aiz[aiz['Valid_Return']].copy()
aes_valid = aes[aes['Valid_Return']].copy()

# Align on common dates
merged_returns = pd.merge(
    aiz_valid[['Date', 'Return']],
    aes_valid[['Date', 'Return']],
    on='Date',
    suffixes=('_AIZ', '_AES')
)

spearman_corr, _ = spearmanr(merged_returns['Return_AIZ'], merged_returns['Return_AES'])
print(f"Spearman correlation: {spearman_corr:.4f}")

# Scatter plot of paired returns
plt.figure(figsize=(10, 8))
plt.scatter(merged_returns['Return_AIZ'], merged_returns['Return_AES'],
            alpha=0.5, s=20, color='darkgreen')
plt.xlabel('AIZ Daily Return')
plt.ylabel('AES Daily Return')
plt.title('Paired Daily Returns: AIZ vs AES')
plt.grid(True, alpha=0.3)
plt.tight_layout()
plt.savefig('returns_scatter.png', dpi=300)
plt.close()
print("Generated: returns_scatter.png")

# ---------- Daily Price Range Analysis ----------
print("\n=== Daily Price Range ===")
akam['Range'] = akam['High'] - akam['Low']
aiz['Range'] = aiz['High'] - aiz['Low']
aes['Range'] = aes['High'] - aes['Low']

mean_range_akam = akam['Range'].mean()
mean_range_aiz = aiz['Range'].mean()
mean_range_aes = aes['Range'].mean()
avg_daily_range = (mean_range_akam + mean_range_aiz + mean_range_aes) / 3
print(f"Average daily range: {avg_daily_range:.4f}")

# Boxplot for daily price range
range_data = pd.DataFrame({
    'Ticker': ['AKAM']*len(akam) + ['AIZ']*len(aiz) + ['AES']*len(aes),
    'Range': pd.concat([akam['Range'], aiz['Range'], aes['Range']])
})

plt.figure(figsize=(10, 6))
box_positions = [1, 2, 3]
box_data = [akam['Range'], aiz['Range'], aes['Range']]
bp = plt.boxplot(box_data, positions=box_positions, labels=['AKAM', 'AIZ', 'AES'],
                 patch_artist=True, widths=0.6)
for patch in bp['boxes']:
    patch.set_facecolor('lightblue')
plt.xlabel('Ticker Symbol')
plt.ylabel('Daily High - Low Price')
plt.title('Daily Price Range Distribution by Ticker')
plt.grid(True, alpha=0.3, axis='y')
plt.tight_layout()
plt.savefig('price_range_boxplot.png', dpi=300)
plt.close()
print("Generated: price_range_boxplot.png")

# ---------- 30-Day Rolling Volatility ----------
print("\n=== 30-Day Rolling Volatility ===")
def rolling_volatility(df, window=30):
    """Calculate rolling standard deviation of returns"""
    df = df.copy()
    returns_only = df[df['Valid_Return']]['Return'].values
    dates_only = df[df['Valid_Return']]['Date'].values

    rolling_vol = []
    rolling_dates = []

    for i in range(len(returns_only)):
        if i >= window - 1:
            window_data = returns_only[i-window+1:i+1]
            if len(window_data) == window:
                rolling_vol.append(np.std(window_data, ddof=1))
                rolling_dates.append(dates_only[i])

    return rolling_dates, rolling_vol

akam_dates, akam_vol = rolling_volatility(akam, 30)
aiz_dates, aiz_vol = rolling_volatility(aiz, 30)
aes_dates, aes_vol = rolling_volatility(aes, 30)

mean_rolling_vol_akam = np.mean(akam_vol) if len(akam_vol) > 0 else 0
mean_rolling_vol_aiz = np.mean(aiz_vol) if len(aiz_vol) > 0 else 0
mean_rolling_vol_aes = np.mean(aes_vol) if len(aes_vol) > 0 else 0
avg_rolling_vol = (mean_rolling_vol_akam + mean_rolling_vol_aiz + mean_rolling_vol_aes) / 3
print(f"Average rolling volatility: {avg_rolling_vol:.4f}")

# Area chart for AKAM rolling volatility
plt.figure(figsize=(12, 6))
plt.fill_between(akam_dates, akam_vol, alpha=0.6, color='coral')
plt.plot(akam_dates, akam_vol, linewidth=1, color='darkred')
plt.xlabel('Date')
plt.ylabel('Rolling Standard Deviation of Daily Return')
plt.title('AKAM 30-Day Rolling Volatility')
plt.grid(True, alpha=0.3)
plt.tight_layout()
plt.savefig('akam_rolling_volatility.png', dpi=300)
plt.close()
print("Generated: akam_rolling_volatility.png")

# ---------- Volume-Return Regression ----------
print("\n=== Volume-Return Regression ===")
def volume_return_regression(df):
    """Regress absolute return on log volume"""
    df_valid = df[df['Valid_Return'] & (df['Volume'] > 0)].copy()
    df_valid['Abs_Return'] = df_valid['Return'].abs()
    df_valid['Log_Volume'] = np.log(df_valid['Volume'])

    X = df_valid[['Log_Volume']].values
    y = df_valid['Abs_Return'].values

    model = LinearRegression()
    model.fit(X, y)
    return model.coef_[0]

slope_akam = volume_return_regression(akam)
slope_aiz = volume_return_regression(aiz)
slope_aes = volume_return_regression(aes)
avg_slope = (slope_akam + slope_aiz + slope_aes) / 3
print(f"Average volume-return slope: {avg_slope:.4f}")

# ---------- Tail Spread Analysis ----------
print("\n=== Tail Spread (90th - 10th Percentile) ===")
# Get common dates across all three files
common_dates = set(akam[akam['Valid_Return']]['Date'])
common_dates = common_dates.intersection(set(aiz[aiz['Valid_Return']]['Date']))
common_dates = common_dates.intersection(set(aes[aes['Valid_Return']]['Date']))

akam_aligned = akam[akam['Date'].isin(common_dates) & akam['Valid_Return']]
aiz_aligned = aiz[aiz['Date'].isin(common_dates) & aiz['Valid_Return']]
aes_aligned = aes[aes['Date'].isin(common_dates) & aes['Valid_Return']]

tail_spread_akam = np.percentile(akam_aligned['Return'], 90) - np.percentile(akam_aligned['Return'], 10)
tail_spread_aiz = np.percentile(aiz_aligned['Return'], 90) - np.percentile(aiz_aligned['Return'], 10)
tail_spread_aes = np.percentile(aes_aligned['Return'], 90) - np.percentile(aes_aligned['Return'], 10)
avg_tail_spread = (tail_spread_akam + tail_spread_aiz + tail_spread_aes) / 3
print(f"Average tail spread: {avg_tail_spread:.4f}")

# ---------- Median Adjusted Close ----------
print("\n=== Median Adjusted Close (Common Dates) ===")
# Common dates for all files (not just valid returns, all dates)
common_dates_all = set(akam['Date']).intersection(set(aiz['Date'])).intersection(set(aes['Date']))

akam_common = akam[akam['Date'].isin(common_dates_all)]
aiz_common = aiz[aiz['Date'].isin(common_dates_all)]
aes_common = aes[aes['Date'].isin(common_dates_all)]

median_akam = akam_common['Adj Close'].median()
median_aiz = aiz_common['Adj Close'].median()
median_aes = aes_common['Adj Close'].median()
avg_median_price = (median_akam + median_aiz + median_aes) / 3
print(f"Average median adjusted price: {avg_median_price:.2f}")

# ---------- Five-Day Cumulative Returns ----------
print("\n=== Five-Day Cumulative Returns ===")
def five_day_cumulative_returns(df):
    """Calculate 5-day cumulative returns"""
    df_valid = df[df['Valid_Return']].copy().reset_index(drop=True)
    cumulative_returns = []

    for i in range(len(df_valid) - 4):
        five_day_sum = df_valid.loc[i:i+4, 'Return'].sum()
        cumulative_returns.append(five_day_sum)

    return cumulative_returns

cumret_akam = five_day_cumulative_returns(akam)
cumret_aiz = five_day_cumulative_returns(aiz)
cumret_aes = five_day_cumulative_returns(aes)

mean_cumret_akam = np.mean(cumret_akam) if len(cumret_akam) > 0 else 0
mean_cumret_aiz = np.mean(cumret_aiz) if len(cumret_aiz) > 0 else 0
mean_cumret_aes = np.mean(cumret_aes) if len(cumret_aes) > 0 else 0
avg_five_day_return = (mean_cumret_akam + mean_cumret_aiz + mean_cumret_aes) / 3
print(f"Average 5-day cumulative return: {avg_five_day_return:.4f}")

# Density plot for pooled 5-day cumulative returns
pooled_cumret = cumret_akam + cumret_aiz + cumret_aes
plt.figure(figsize=(10, 6))
from scipy.stats import gaussian_kde
if len(pooled_cumret) > 0:
    density = gaussian_kde(pooled_cumret)
    xs = np.linspace(min(pooled_cumret), max(pooled_cumret), 200)
    plt.plot(xs, density(xs), linewidth=2, color='purple')
    plt.fill_between(xs, density(xs), alpha=0.3, color='purple')
plt.xlabel('Five-Day Cumulative Return')
plt.ylabel('Density')
plt.title('Density Plot: Pooled Five-Day Cumulative Returns')
plt.grid(True, alpha=0.3)
plt.tight_layout()
plt.savefig('five_day_cumret_density.png', dpi=300)
plt.close()
print("Generated: five_day_cumret_density.png")

# ---------- Trend Slope Analysis ----------
print("\n=== Trend Slope (Time Index Regression) ===")
def trend_slope(df):
    """Regress Adj Close on time index"""
    df_clean = df.dropna(subset=['Adj Close']).copy()
    df_clean['Time_Index'] = range(len(df_clean))

    X = df_clean[['Time_Index']].values
    y = df_clean['Adj Close'].values

    model = LinearRegression()
    model.fit(X, y)
    return model.coef_[0]

trend_akam = trend_slope(akam)
trend_aiz = trend_slope(aiz)
trend_aes = trend_slope(aes)
avg_trend_slope = (trend_akam + trend_aiz + trend_aes) / 3
print(f"Average trend slope: {avg_trend_slope:.4f}")

# ---------- Liquidity Ratio Analysis ----------
print("\n=== Liquidity Ratio ===")
def liquidity_ratio(df):
    """Calculate Volume / (High - Low)"""
    df = df.copy()
    df['Liquidity'] = df['Volume'] / (df['High'] - df['Low'])
    # Remove infinite values
    df['Liquidity'] = df['Liquidity'].replace([np.inf, -np.inf], np.nan)
    return df['Liquidity'].mean()

liq_akam = liquidity_ratio(akam)
liq_aiz = liquidity_ratio(aiz)
liq_aes = liquidity_ratio(aes)
avg_liquidity = (liq_akam + liq_aiz + liq_aes) / 3
print(f"Average liquidity ratio: {avg_liquidity:.2f}")

# Bar chart for liquidity ratio
plt.figure(figsize=(10, 6))
tickers = ['AKAM', 'AIZ', 'AES']
liquidity_values = [liq_akam, liq_aiz, liq_aes]
bars = plt.bar(tickers, liquidity_values, color=['steelblue', 'coral', 'lightgreen'],
               edgecolor='black', linewidth=1.5)
plt.xlabel('Ticker Symbol')
plt.ylabel('Liquidity Ratio (Volume / Range)')
plt.title('Average Liquidity Ratio by Ticker')
plt.grid(True, alpha=0.3, axis='y')
plt.tight_layout()
plt.savefig('liquidity_ratio_bar.png', dpi=300)
plt.close()
print("Generated: liquidity_ratio_bar.png")

# ---------- Volume Coefficient of Variation ----------
print("\n=== Volume Coefficient of Variation ===")
cv_akam = akam['Volume'].std() / akam['Volume'].mean()
cv_aiz = aiz['Volume'].std() / aiz['Volume'].mean()
cv_aes = aes['Volume'].std() / aes['Volume'].mean()
avg_volume_cv = (cv_akam + cv_aiz + cv_aes) / 3
print(f"Average volume CV: {avg_volume_cv:.4f}")

# ---------- Rolling Log Volume Volatility ----------
print("\n=== 20-Day Rolling Log Volume Volatility ===")
def rolling_log_volume_volatility(df, window=20):
    """Calculate rolling std of log volume"""
    df_valid = df[df['Volume'] > 0].copy()
    df_valid['Log_Volume'] = np.log(df_valid['Volume'])

    log_vols = df_valid['Log_Volume'].values
    dates = df_valid['Date'].values

    rolling_vol = []
    rolling_dates = []

    for i in range(len(log_vols)):
        if i >= window - 1:
            window_data = log_vols[i-window+1:i+1]
            if len(window_data) == window:
                rolling_vol.append(np.std(window_data, ddof=1))
                rolling_dates.append(dates[i])

    return rolling_dates, rolling_vol

akam_lv_dates, akam_lv_vol = rolling_log_volume_volatility(akam, 20)
aiz_lv_dates, aiz_lv_vol = rolling_log_volume_volatility(aiz, 20)
aes_lv_dates, aes_lv_vol = rolling_log_volume_volatility(aes, 20)

mean_lv_vol_akam = np.mean(akam_lv_vol) if len(akam_lv_vol) > 0 else 0
mean_lv_vol_aiz = np.mean(aiz_lv_vol) if len(aiz_lv_vol) > 0 else 0
mean_lv_vol_aes = np.mean(aes_lv_vol) if len(aes_lv_vol) > 0 else 0
avg_rolling_lv_vol = (mean_lv_vol_akam + mean_lv_vol_aiz + mean_lv_vol_aes) / 3
print(f"Average rolling log volume volatility: {avg_rolling_lv_vol:.4f}")

# Spiral plot for AES rolling log volume volatility
plt.figure(figsize=(10, 10))
if len(aes_lv_vol) > 0:
    n = len(aes_lv_vol)
    # Angular index (0 to 2*pi per cycle)
    angles = np.linspace(0, 10*np.pi, n)  # Multiple spirals
    # Radius based on volatility
    radii = np.array(aes_lv_vol)
    # Normalize radii for better visualization
    radii_norm = (radii - radii.min()) / (radii.max() - radii.min() + 1e-10) * 5 + 1

    ax = plt.subplot(111, projection='polar')
    scatter = ax.scatter(angles, radii_norm, c=range(n), cmap='viridis', s=20, alpha=0.6)
    ax.set_title('AES Rolling Log Volume Volatility (Spiral Plot)', pad=20)
    plt.colorbar(scatter, label='Trading Day Sequence')
plt.tight_layout()
plt.savefig('aes_log_volume_spiral.png', dpi=300)
plt.close()
print("Generated: aes_log_volume_spiral.png")

# ---------- Maximum Absolute Slope ----------
print("\n=== Maximum Absolute Slope ===")
max_abs_slope = max(abs(slope_akam), abs(slope_aiz), abs(slope_aes))
print(f"Maximum absolute slope: {max_abs_slope:.4f}")

# ---------- Open-Ended Analysis ----------
print("\n=== Extreme Five-Day Cumulative Return Analysis ===")
# Identify extreme values (top and bottom 10%)
pooled_cumret_array = np.array(pooled_cumret)
threshold_high = np.percentile(pooled_cumret_array, 90)
threshold_low = np.percentile(pooled_cumret_array, 10)

# Collect corresponding features for extreme returns
extreme_features_analysis = """
WHAT FEATURES DISTINGUISH DAYS WITH EXTREME FIVE-DAY CUMULATIVE RETURN VALUES?

Days with extreme five-day cumulative returns exhibit several distinctive characteristics
that separate them from periods of normal market behavior. Extreme positive cumulative
returns (top 10th percentile) are typically preceded by sustained directional momentum,
clustering of large single-day moves in the same direction, and elevated trading volumes
that reflect intensified investor participation and information flow. These periods often
coincide with low intraday price dispersion in the initial days followed by expanding
ranges as momentum accelerates, suggesting initial consensus building that transitions
into broader market engagement. Conversely, extreme negative cumulative returns (bottom
10th percentile) are characterized by heightened volatility persistence, where consecutive
days of negative returns compound with increasing intraday ranges, reflecting sustained
selling pressure and deteriorating sentiment. Volume patterns during extreme negative
periods tend to spike initially during panic selling then stabilize at elevated levels,
while absolute return magnitudes remain consistently high throughout the five-day window.
Both tails exhibit autocorrelation in daily returns within the window, departing from
the near-zero autocorrelation observed during normal periods, indicating that extreme
cumulative returns arise not from random single-day shocks but from persistent directional
pressure sustained over multiple consecutive trading sessions.
"""

print(extreme_features_analysis)

print("\n" + "="*60)
print("ANALYSIS COMPLETE")
print("="*60)
print("\nKEY OUTPUTS:")
print(f"1. ARIMA Mean Forecast (AKAM, 30 days): {mean_forecast:.2f}")
print(f"2. Spearman Correlation (AIZ vs AES): {spearman_corr:.4f}")
print(f"3. Average Daily Range: {avg_daily_range:.4f}")
print(f"4. Average Rolling Volatility (30-day): {avg_rolling_vol:.4f}")
print(f"5. Average Volume-Return Slope: {avg_slope:.4f}")
print(f"6. Average Tail Spread (90th-10th): {avg_tail_spread:.4f}")
print(f"7. Average Median Adjusted Price: {avg_median_price:.2f}")
print(f"8. Average 5-Day Cumulative Return: {avg_five_day_return:.4f}")
print(f"9. Average Trend Slope: {avg_trend_slope:.4f}")
print(f"10. Average Liquidity Ratio: {avg_liquidity:.2f}")
print(f"11. Average Volume CV: {avg_volume_cv:.4f}")
print(f"12. Average Rolling Log Volume Volatility (20-day): {avg_rolling_lv_vol:.4f}")
print(f"13. Maximum Absolute Slope: {max_abs_slope:.4f}")
print("="*60)
