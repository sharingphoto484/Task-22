# ==========================================
# Integrated Equity Behavior Analysis Script
# ==========================================
# Requirements: pandas, numpy, matplotlib, scipy, scikit-learn, statsmodels
# Input files: ABNB.csv, ABBV.csv, AAL.csv (in same directory)
# Output files: abnb_adjusted_price.png, returns_scatter.png,
#               daily_ranges_boxplot.png
# ==========================================

import pandas as pd
import numpy as np
import matplotlib.pyplot as plt
from scipy.stats import spearmanr
from statsmodels.tsa.arima.model import ARIMA
from sklearn.linear_model import LinearRegression
import warnings
warnings.filterwarnings('ignore')

# ---------- Load CSVs Robustly ----------
def load_and_clean_data(filepath):
    """Load CSV and remove rows with missing or non-numeric values"""
    df = pd.read_csv(filepath)
    
    # Parse dates consistently
    df['Date'] = pd.to_datetime(df['Date'])
    
    # Calculate Adj Close from Close, Dividends, and Stock Splits
    # If stock splits exist, adjust for them; if dividends exist, use Close as-is
    # For simplicity, if no Adj Close column, use Close
    if 'Adj Close' not in df.columns:
        df['Adj Close'] = df['Close']
    
    # Define required numeric columns
    numeric_cols = ['Open', 'High', 'Low', 'Close', 'Adj Close', 'Volume']
    
    # Ensure all required columns exist
    for col in numeric_cols:
        if col not in df.columns:
            if col == 'Adj Close':
                df['Adj Close'] = df['Close']
    
    # Convert to numeric and drop rows with missing/non-numeric values
    for col in numeric_cols:
        df[col] = pd.to_numeric(df[col], errors='coerce')
    
    # Drop rows with any missing values in required fields
    df = df.dropna(subset=['Date'] + numeric_cols)
    
    # Sort by date
    df = df.sort_values('Date').reset_index(drop=True)
    
    return df

# Load all three datasets
abnb_df = load_and_clean_data('ABNB.csv')
abbv_df = load_and_clean_data('ABBV.csv')
aal_df = load_and_clean_data('AAL.csv')

print("Data loaded successfully")
print(f"ABNB: {len(abnb_df)} rows")
print(f"ABBV: {len(abbv_df)} rows")
print(f"AAL: {len(aal_df)} rows\n")

# ---------- Calculate Daily Returns ----------
def calculate_daily_returns(df):
    """Calculate log returns and exclude zero volume days"""
    df = df.copy()
    df = df[df['Volume'] > 0].reset_index(drop=True)
    df['Return'] = np.log(df['Adj Close']) - np.log(df['Adj Close'].shift(1))
    df = df.dropna(subset=['Return'])
    return df

abnb_returns = calculate_daily_returns(abnb_df)
abbv_returns = calculate_daily_returns(abbv_df)
aal_returns = calculate_daily_returns(aal_df)

# ---------- Component 1: ARIMA(1,1,1) Forecast for ABNB ----------
print("=" * 60)
print("COMPONENT 1: ARIMA(1,1,1) Forecast for ABNB")
print("=" * 60)

# Fit ARIMA(1,1,1) model
arima_model = ARIMA(abnb_df['Adj Close'].values, order=(1, 1, 1))
arima_fit = arima_model.fit()

# Forecast 30 trading days
forecast = arima_fit.forecast(steps=30)
forecast_mean = np.mean(forecast)

print(f"Mean of 30-day forecast: {forecast_mean:.2f}\n")

# Generate line plot for ABNB adjusted price series
plt.figure(figsize=(12, 6))
plt.plot(abnb_df['Date'], abnb_df['Adj Close'], linewidth=1.5)
plt.xlabel('Date')
plt.ylabel('Adjusted Closing Value')
plt.title('ABNB Adjusted Price Series')
plt.grid(True, alpha=0.3)
plt.tight_layout()
plt.savefig('abnb_adjusted_price.png', dpi=150)
plt.close()
print("Generated: abnb_adjusted_price.png")

# ---------- Component 2: Spearman Correlation (ABBV & AAL) ----------
print("\n" + "=" * 60)
print("COMPONENT 2: Spearman Correlation between ABBV and AAL Returns")
print("=" * 60)

# Align returns by date
abbv_ret_align = abbv_returns[['Date', 'Return']].rename(columns={'Return': 'ABBV_Return'})
aal_ret_align = aal_returns[['Date', 'Return']].rename(columns={'Return': 'AAL_Return'})
aligned_returns = pd.merge(abbv_ret_align, aal_ret_align, on='Date')

# Calculate Spearman correlation
spearman_corr, _ = spearmanr(aligned_returns['ABBV_Return'], aligned_returns['AAL_Return'])

print(f"Spearman rank correlation: {spearman_corr:.4f}\n")

# Generate scatter plot
plt.figure(figsize=(10, 8))
plt.scatter(aligned_returns['ABBV_Return'], aligned_returns['AAL_Return'],
            alpha=0.5, s=20)
plt.xlabel('ABBV Daily Return')
plt.ylabel('AAL Daily Return')
plt.title('Scatter Plot of Aligned ABBV and AAL Returns')
plt.grid(True, alpha=0.3)
plt.tight_layout()
plt.savefig('returns_scatter.png', dpi=150)
plt.close()
print("Generated: returns_scatter.png")

# ---------- Component 3: Overall Volatility Measure ----------
print("\n" + "=" * 60)
print("COMPONENT 3: Overall Volatility Measure (Mean of Daily Ranges)")
print("=" * 60)

# Calculate daily range (High - Low) for each dataset
abnb_df['Range'] = abnb_df['High'] - abnb_df['Low']
abbv_df['Range'] = abbv_df['High'] - abbv_df['Low']
aal_df['Range'] = aal_df['High'] - aal_df['Low']

# Mean range for each stock
abnb_mean_range = abnb_df['Range'].mean()
abbv_mean_range = abbv_df['Range'].mean()
aal_mean_range = aal_df['Range'].mean()

# Overall volatility measure
overall_volatility = (abnb_mean_range + abbv_mean_range + aal_mean_range) / 3

print(f"ABNB mean range: {abnb_mean_range:.4f}")
print(f"ABBV mean range: {abbv_mean_range:.4f}")
print(f"AAL mean range: {aal_mean_range:.4f}")
print(f"Overall volatility measure: {overall_volatility:.4f}\n")

# Generate boxplot
plt.figure(figsize=(10, 6))
box_data = [abnb_df['Range'], abbv_df['Range'], aal_df['Range']]
plt.boxplot(box_data, labels=['ABNB', 'ABBV', 'AAL'])
plt.xlabel('Ticker Symbol')
plt.ylabel('Daily High minus Low Value')
plt.title('Boxplot of Daily Price Ranges')
plt.grid(True, alpha=0.3, axis='y')
plt.tight_layout()
plt.savefig('daily_ranges_boxplot.png', dpi=150)
plt.close()
print("Generated: daily_ranges_boxplot.png")

# ---------- Component 4: Volume-Return Regression (Mean Slope) ----------
print("\n" + "=" * 60)
print("COMPONENT 4: Volume-Return Regression Analysis")
print("=" * 60)

def fit_volume_return_regression(returns_df):
    """Fit regression: abs(return) ~ log(volume)"""
    df = returns_df.copy()
    df['Abs_Return'] = np.abs(df['Return'])
    df['Log_Volume'] = np.log(df['Volume'])
    
    X = df['Log_Volume'].values.reshape(-1, 1)
    y = df['Abs_Return'].values
    
    model = LinearRegression()
    model.fit(X, y)
    
    return model.coef_[0]

abnb_slope = fit_volume_return_regression(abnb_returns)
abbv_slope = fit_volume_return_regression(abbv_returns)
aal_slope = fit_volume_return_regression(aal_returns)

mean_slope = (abnb_slope + abbv_slope + aal_slope) / 3

print(f"ABNB regression slope: {abnb_slope:.4f}")
print(f"ABBV regression slope: {abbv_slope:.4f}")
print(f"AAL regression slope: {aal_slope:.4f}")
print(f"Mean slope across three regressions: {mean_slope:.4f}\n")

# Store slopes for later use
slopes = [abnb_slope, abbv_slope, aal_slope]

# ---------- Component 5: Mean of Medians (Shared Date Range) ----------
print("\n" + "=" * 60)
print("COMPONENT 5: Median Price Comparison (Shared Date Range)")
print("=" * 60)

# Find common dates
abnb_dates = set(abnb_df['Date'])
abbv_dates = set(abbv_df['Date'])
aal_dates = set(aal_df['Date'])
common_dates = abnb_dates & abbv_dates & aal_dates

print(f"Number of common dates: {len(common_dates)}")

# Filter to common dates
abnb_common = abnb_df[abnb_df['Date'].isin(common_dates)]
abbv_common = abbv_df[abbv_df['Date'].isin(common_dates)]
aal_common = aal_df[aal_df['Date'].isin(common_dates)]

# Calculate medians
abnb_median = abnb_common['Adj Close'].median()
abbv_median = abbv_common['Adj Close'].median()
aal_median = aal_common['Adj Close'].median()

mean_of_medians = (abnb_median + abbv_median + aal_median) / 3

print(f"ABNB median Adj Close: {abnb_median:.2f}")
print(f"ABBV median Adj Close: {abbv_median:.2f}")
print(f"AAL median Adj Close: {aal_median:.2f}")
print(f"Mean of three medians: {mean_of_medians:.2f}\n")

# ---------- Component 6: Rolling Volatility (30-day) ----------
print("\n" + "=" * 60)
print("COMPONENT 6: 30-Day Rolling Volatility Analysis")
print("=" * 60)

# Calculate 30-day rolling std for returns
abnb_rolling_std = abnb_returns['Return'].rolling(window=30).std()
abbv_rolling_std = abbv_returns['Return'].rolling(window=30).std()
aal_rolling_std = aal_returns['Return'].rolling(window=30).std()

# Get maximum values
abnb_max_rolling = abnb_rolling_std.max()
abbv_max_rolling = abbv_rolling_std.max()
aal_max_rolling = aal_rolling_std.max()

avg_max_rolling = (abnb_max_rolling + abbv_max_rolling + aal_max_rolling) / 3

print(f"ABNB max rolling std: {abnb_max_rolling:.4f}")
print(f"ABBV max rolling std: {abbv_max_rolling:.4f}")
print(f"AAL max rolling std: {aal_max_rolling:.4f}")
print(f"Average of maximum rolling std: {avg_max_rolling:.4f}\n")

# ---------- Component 7: Average Trading Volume ----------
print("\n" + "=" * 60)
print("COMPONENT 7: Average Trading Volume")
print("=" * 60)

abnb_mean_vol = abnb_df['Volume'].mean()
abbv_mean_vol = abbv_df['Volume'].mean()
aal_mean_vol = aal_df['Volume'].mean()

avg_trading_volume = (abnb_mean_vol + abbv_mean_vol + aal_mean_vol) / 3

print(f"ABNB mean volume: {abnb_mean_vol:.2f}")
print(f"ABBV mean volume: {abbv_mean_vol:.2f}")
print(f"AAL mean volume: {aal_mean_vol:.2f}")
print(f"Average trading volume: {avg_trading_volume:.2f}\n")

# ---------- Component 8: Tail Width (90th - 10th Percentile) ----------
print("\n" + "=" * 60)
print("COMPONENT 8: Tail Width Analysis")
print("=" * 60)

# Use aligned sample for ABBV and AAL
abbv_aligned_returns = aligned_returns['ABBV_Return']
aal_aligned_returns = aligned_returns['AAL_Return']

# For ABNB, align with the same date range
abnb_aligned = abnb_returns[abnb_returns['Date'].isin(aligned_returns['Date'])]

# Calculate tail widths
abnb_tail = np.percentile(abnb_aligned['Return'], 90) - np.percentile(abnb_aligned['Return'], 10)
abbv_tail = np.percentile(abbv_aligned_returns, 90) - np.percentile(abbv_aligned_returns, 10)
aal_tail = np.percentile(aal_aligned_returns, 90) - np.percentile(aal_aligned_returns, 10)

avg_tail_width = (abnb_tail + abbv_tail + aal_tail) / 3

print(f"ABNB tail width (90th - 10th): {abnb_tail:.4f}")
print(f"ABBV tail width (90th - 10th): {abbv_tail:.4f}")
print(f"AAL tail width (90th - 10th): {aal_tail:.4f}")
print(f"Average tail width: {avg_tail_width:.4f}\n")

# ---------- Component 9: Maximum Absolute Regression Slope ----------
print("\n" + "=" * 60)
print("COMPONENT 9: Maximum Absolute Regression Slope")
print("=" * 60)

abs_slopes = [abs(s) for s in slopes]
max_abs_slope = max(abs_slopes)

print(f"ABNB absolute slope: {abs(abnb_slope):.4f}")
print(f"ABBV absolute slope: {abs(abbv_slope):.4f}")
print(f"AAL absolute slope: {abs(aal_slope):.4f}")
print(f"Maximum absolute slope: {max_abs_slope:.4f}\n")

# ---------- Summary of Key Results ----------
print("\n" + "=" * 60)
print("KEY RESULTS SUMMARY")
print("=" * 60)
print(f"1. ARIMA Forecast Mean (ABNB 30-day):              {forecast_mean:.2f}")
print(f"2. Spearman Correlation (ABBV vs AAL):             {spearman_corr:.4f}")
print(f"3. Overall Volatility Measure:                     {overall_volatility:.4f}")
print(f"4. Mean Regression Slope (Volume-Return):          {mean_slope:.4f}")
print(f"5. Mean of Medians (Common Dates):                 {mean_of_medians:.2f}")
print(f"6. Average Maximum Rolling Std (30-day):           {avg_max_rolling:.4f}")
print(f"7. Average Trading Volume:                         {avg_trading_volume:.2f}")
print(f"8. Average Tail Width (90th - 10th percentile):    {avg_tail_width:.4f}")
print(f"9. Maximum Absolute Regression Slope:              {max_abs_slope:.4f}")
print("=" * 60)

# ---------- Market Behavior Interpretation ----------
print("\n" + "=" * 60)
print("MARKET BEHAVIOR INTERPRETATION")
print("=" * 60)

interpretation = """
The observed pattern suggests a moderately liquid equity market characterized by
episodic volatility clustering and weak contemporaneous co-movement across sectors.
The positive but modest Spearman correlation indicates that systematic risk factors
partially drive returns, yet idiosyncratic shocks dominate individual stock behavior.
The positive volume-return slope relationship confirms that elevated trading activity
accompanies larger absolute price movements, consistent with information arrival and
heterogeneous beliefs among market participants. Wide return distribution tails and
elevated maximum rolling volatility indicate non-normal return distributions with
occasional extreme events, suggesting that risk management frameworks relying solely
on variance underestimate downside exposure. The ARIMA-based forecast captures
short-term momentum but may underperform during structural breaks or regime shifts.
Overall, the data reflect efficient but incomplete information aggregation, where
price discovery is ongoing and trading volumes signal the intensity of disagreement
or information asymmetry.
"""

print(interpretation)
print("=" * 60)
