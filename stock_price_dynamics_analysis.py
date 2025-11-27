# ==========================================
# Stock Price Dynamics Analysis Script
# Requirements: pandas, numpy, matplotlib, scipy, scikit-learn, statsmodels
# Input files: EICHERMOT.csv, HEROMOTOCO.csv, MARUTI.csv (in same directory)
# Output files: spiral_chart.png
# ==========================================

import pandas as pd
import numpy as np
import matplotlib.pyplot as plt
from scipy.stats import spearmanr
from sklearn.linear_model import LogisticRegression
import statsmodels.api as sm
from statsmodels.regression.quantile_regression import QuantReg
import warnings
warnings.filterwarnings('ignore')

# ---------- Load CSVs Robustly ----------
print("Loading datasets...")
eicher = pd.read_csv('EICHERMOT.csv')
hero = pd.read_csv('HEROMOTOCO.csv')
maruti = pd.read_csv('MARUTI.csv')

# ---------- Prepare Individual Datasets ----------
print("Preparing individual datasets...")

# Select required columns and rename for consistency
def prepare_stock_data(df, stock_name):
    # Use Close as Adj Close (common for Indian stock data)
    df = df[['Date', 'Open', 'High', 'Low', 'Close', 'Volume']].copy()
    df.columns = ['Date', f'{stock_name}_Open', f'{stock_name}_High',
                  f'{stock_name}_Low', f'{stock_name}_Close', f'{stock_name}_Volume']
    df['Date'] = pd.to_datetime(df['Date'], errors='coerce')
    return df

eicher_clean = prepare_stock_data(eicher, 'EICHERMOT')
hero_clean = prepare_stock_data(hero, 'HEROMOTOCO')
maruti_clean = prepare_stock_data(maruti, 'MARUTI')

# ---------- Merge on Common Dates ----------
print("Merging datasets on common dates...")
merged = eicher_clean.merge(hero_clean, on='Date', how='inner')
merged = merged.merge(maruti_clean, on='Date', how='inner')

# Sort chronologically
merged = merged.sort_values('Date').reset_index(drop=True)

# ---------- Remove Missing and Non-Numeric Values ----------
print("Cleaning merged dataset...")
# Convert all numeric columns to numeric, coercing errors to NaN
numeric_cols = [col for col in merged.columns if col != 'Date']
for col in numeric_cols:
    merged[col] = pd.to_numeric(merged[col], errors='coerce')

# Drop rows with any missing values
merged = merged.dropna()

print(f"Merged dataset shape: {merged.shape}")
print(f"Date range: {merged['Date'].min()} to {merged['Date'].max()}")

# ---------- Calculate Daily Returns (Log Returns) ----------
print("\nCalculating daily returns...")

def calculate_log_returns(df, stock_name):
    close_col = f'{stock_name}_Close'
    volume_col = f'{stock_name}_Volume'
    return_col = f'{stock_name}_Return'

    # Calculate log returns: ln(Close_t) - ln(Close_t-1)
    df[return_col] = np.log(df[close_col]) - np.log(df[close_col].shift(1))

    # Mark rows where Volume = 0 (will exclude from return-based computations)
    df[f'{stock_name}_ValidReturn'] = (df[volume_col] > 0) & (~df[return_col].isna())

    return df

merged = calculate_log_returns(merged, 'EICHERMOT')
merged = calculate_log_returns(merged, 'HEROMOTOCO')
merged = calculate_log_returns(merged, 'MARUTI')

# ---------- 1. Quantile Regression (90%) for HEROMOTOCO ----------
print("\n1. Quantile Regression (90%) for HEROMOTOCO...")

# Prepare data excluding zero volume
hero_qr = merged[merged['HEROMOTOCO_ValidReturn']].copy()
hero_qr['log_volume'] = np.log(hero_qr['HEROMOTOCO_Volume'])
hero_qr = hero_qr.dropna(subset=['HEROMOTOCO_Return', 'log_volume'])

X_hero = hero_qr[['log_volume']]
y_hero = hero_qr['HEROMOTOCO_Return']

# Fit 90% quantile regression
qr_hero = QuantReg(y_hero, sm.add_constant(X_hero)).fit(q=0.9)
hero_coef = qr_hero.params['log_volume']

print(f"   HEROMOTOCO 90% quantile regression coefficient (log Volume): {hero_coef:.4f}")

# ---------- Perform 90% Quantile Regression for All Stocks ----------
print("\nPerforming 90% quantile regression for all stocks...")

def fit_quantile_regression(df, stock_name, quantile=0.9):
    stock_data = df[df[f'{stock_name}_ValidReturn']].copy()
    stock_data['log_volume'] = np.log(stock_data[f'{stock_name}_Volume'])
    stock_data = stock_data.dropna(subset=[f'{stock_name}_Return', 'log_volume'])

    X = stock_data[['log_volume']]
    y = stock_data[f'{stock_name}_Return']

    qr = QuantReg(y, sm.add_constant(X)).fit(q=quantile)
    return qr.params['log_volume']

eicher_qr_coef = fit_quantile_regression(merged, 'EICHERMOT', 0.9)
hero_qr_coef = fit_quantile_regression(merged, 'HEROMOTOCO', 0.9)
maruti_qr_coef = fit_quantile_regression(merged, 'MARUTI', 0.9)

print(f"   EICHERMOT: {eicher_qr_coef:.4f}")
print(f"   HEROMOTOCO: {hero_qr_coef:.4f}")
print(f"   MARUTI: {maruti_qr_coef:.4f}")

# ---------- 2. Average Volatility (30-day Rolling Std Dev) ----------
print("\n2. Calculating 30-day rolling volatility...")

def calculate_avg_volatility(df, stock_name):
    stock_data = df[df[f'{stock_name}_ValidReturn']].copy()
    returns = stock_data[f'{stock_name}_Return']

    # 30-day rolling standard deviation
    rolling_std = returns.rolling(window=30, min_periods=30).std()

    # Average across full range
    avg_vol = rolling_std.mean()
    return avg_vol

eicher_vol = calculate_avg_volatility(merged, 'EICHERMOT')
hero_vol = calculate_avg_volatility(merged, 'HEROMOTOCO')
maruti_vol = calculate_avg_volatility(merged, 'MARUTI')

overall_avg_vol = (eicher_vol + hero_vol + maruti_vol) / 3

print(f"   EICHERMOT avg volatility: {eicher_vol:.6f}")
print(f"   HEROMOTOCO avg volatility: {hero_vol:.6f}")
print(f"   MARUTI avg volatility: {maruti_vol:.6f}")
print(f"   Overall average volatility: {overall_avg_vol:.4f}")

# ---------- 3. Spearman Correlation (Volume vs Absolute Return) ----------
print("\n3. Spearman correlation (Volume vs Absolute Return)...")

def calculate_spearman_corr(df, stock_name):
    stock_data = df[df[f'{stock_name}_ValidReturn']].copy()
    volume = stock_data[f'{stock_name}_Volume']
    abs_return = np.abs(stock_data[f'{stock_name}_Return'])

    corr, _ = spearmanr(volume, abs_return)
    return corr

eicher_spearman = calculate_spearman_corr(merged, 'EICHERMOT')
hero_spearman = calculate_spearman_corr(merged, 'HEROMOTOCO')
maruti_spearman = calculate_spearman_corr(merged, 'MARUTI')

avg_spearman = (eicher_spearman + hero_spearman + maruti_spearman) / 3

print(f"   EICHERMOT: {eicher_spearman:.6f}")
print(f"   HEROMOTOCO: {hero_spearman:.6f}")
print(f"   MARUTI: {maruti_spearman:.6f}")
print(f"   Average Spearman correlation: {avg_spearman:.4f}")

# ---------- 4. Logistic Regression for EICHERMOT ----------
print("\n4. Logistic regression for EICHERMOT...")

eicher_logit = merged[merged['EICHERMOT_ValidReturn']].copy()

# Response: 1 if |return| > 75th percentile
abs_return = np.abs(eicher_logit['EICHERMOT_Return'])
percentile_75 = abs_return.quantile(0.75)
eicher_logit['high_volatility'] = (abs_return > percentile_75).astype(int)

# Predictor: 1 if Volume > median
volume_median = eicher_logit['EICHERMOT_Volume'].median()
eicher_logit['high_volume'] = (eicher_logit['EICHERMOT_Volume'] > volume_median).astype(int)

# Fit logistic regression
X_logit = eicher_logit[['high_volume']].values
y_logit = eicher_logit['high_volatility'].values

logit_model = LogisticRegression(fit_intercept=True, solver='lbfgs')
logit_model.fit(X_logit, y_logit)

logit_coef = logit_model.coef_[0][0]

print(f"   High-volume indicator coefficient: {logit_coef:.4f}")

# ---------- 5. Quantile Spread (90% - 10%) ----------
print("\n5. Quantile spread (90% - 10%)...")

def calculate_quantile_spread(df, stock_name):
    stock_data = df[df[f'{stock_name}_ValidReturn']].copy()
    returns = stock_data[f'{stock_name}_Return']

    q90 = returns.quantile(0.9)
    q10 = returns.quantile(0.1)
    spread = q90 - q10
    return spread

eicher_spread = calculate_quantile_spread(merged, 'EICHERMOT')
hero_spread = calculate_quantile_spread(merged, 'HEROMOTOCO')
maruti_spread = calculate_quantile_spread(merged, 'MARUTI')

avg_spread = (eicher_spread + hero_spread + maruti_spread) / 3

print(f"   EICHERMOT: {eicher_spread:.6f}")
print(f"   HEROMOTOCO: {hero_spread:.6f}")
print(f"   MARUTI: {maruti_spread:.6f}")
print(f"   Average quantile spread: {avg_spread:.4f}")

# ---------- 6. Spiral Chart for MARUTI ----------
print("\n6. Creating spiral chart for MARUTI...")

maruti_spiral = merged[merged['MARUTI_ValidReturn']].copy()
maruti_spiral = maruti_spiral.reset_index(drop=True)

# Angular index (theta) - normalized to radians
n_days = len(maruti_spiral)
theta = np.linspace(0, 8 * np.pi, n_days)  # Multiple spirals

# Radial distance = absolute return
radial = np.abs(maruti_spiral['MARUTI_Return'])

# Color intensity = Volume
volume = maruti_spiral['MARUTI_Volume']

max_radial = radial.max()

print(f"   Maximum radial value (max absolute return): {max_radial:.4f}")

# Create spiral plot
fig, ax = plt.subplots(figsize=(10, 10), subplot_kw=dict(projection='polar'))

scatter = ax.scatter(theta, radial, c=volume, cmap='viridis',
                     alpha=0.6, s=10, edgecolors='none')

ax.set_title('MARUTI Spiral Chart: Absolute Return vs Trading Days',
             pad=20, fontsize=14, fontweight='bold')
ax.set_xlabel('Angular Index (Trading Days)', fontsize=10)
ax.set_ylabel('Radial Distance (Absolute Return)', fontsize=10)

# Add colorbar for volume
cbar = plt.colorbar(scatter, ax=ax, pad=0.1)
cbar.set_label('Volume', rotation=270, labelpad=20)

plt.tight_layout()
plt.savefig('spiral_chart.png', dpi=300, bbox_inches='tight')
print("   Spiral chart saved as 'spiral_chart.png'")

# ---------- 7. Largest 90% Quantile Coefficient ----------
print("\n7. Identifying stock with largest 90% quantile coefficient...")

coefficients = {
    'EICHERMOT': eicher_qr_coef,
    'HEROMOTOCO': hero_qr_coef,
    'MARUTI': maruti_qr_coef
}

largest_stock = max(coefficients, key=coefficients.get)

print(f"   Stock with largest 90% quantile coefficient: {largest_stock}")

# ==========================================
# KEY OUTPUTS
# ==========================================

print("\n" + "="*50)
print("KEY OUTPUTS")
print("="*50)

print(f"\n1. HEROMOTOCO 90% Quantile Regression Coefficient (log Volume): {hero_coef:.4f}")
print(f"2. Average Volatility (30-day rolling std): {overall_avg_vol:.4f}")
print(f"3. Average Spearman Correlation (Volume vs |Return|): {avg_spearman:.4f}")
print(f"4. EICHERMOT Logistic Regression Coefficient (High Volume): {logit_coef:.4f}")
print(f"5. Average Quantile Spread (90% - 10%): {avg_spread:.4f}")
print(f"6. Maximum Radial Value (MARUTI Spiral Chart): {max_radial:.4f}")
print(f"7. Stock with Largest 90% Quantile Coefficient: {largest_stock}")

print("\n" + "="*50)
print("ANALYSIS COMPLETE")
print("="*50)
