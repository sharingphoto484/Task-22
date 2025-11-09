# ==========================================
# Integrated Crypto Analysis Pipeline
# ==========================================
# Requirements: pandas, numpy, matplotlib, scipy, scikit-learn, statsmodels, ruptures
# Input files: Bitcoin 2024.csv, Bitcoin USD (01-05.2024).csv, Ethereum 2024.csv
# Output files: metrics.txt, volatility_volume_scatterplot.png
# ==========================================

import pandas as pd
import numpy as np
import matplotlib.pyplot as plt
from scipy import stats
from scipy.stats import spearmanr, ttest_ind
from sklearn.cluster import KMeans
from sklearn.metrics import silhouette_score
from sklearn.linear_model import QuantileRegressor
from statsmodels.tsa.stattools import adfuller, grangercausalitytests
from statsmodels.tsa.arima.model import ARIMA
from statsmodels.tsa.statespace.sarimax import SARIMAX
from sklearn.metrics import mean_squared_error
import ruptures as rpt
import warnings
warnings.filterwarnings('ignore')

# ---------- Load CSVs Robustly ----------
def load_and_preprocess(filepath):
    """Load CSV and perform comprehensive preprocessing"""
    df = pd.read_csv(filepath)
    return df

btc_2024 = load_and_preprocess('Bitcoin 2024.csv')
btc_partial = load_and_preprocess('Bitcoin USD (01-05.2024).csv')
eth_2024 = load_and_preprocess('Ethereum 2024.csv')

# ---------- Data Cleaning and Normalization ----------
def clean_dataset(df, price_col='Price'):
    """Clean and normalize dataset"""
    df_clean = df.copy()

    # Convert Date column
    df_clean['Date'] = pd.to_datetime(df_clean['Date'])

    # Function to clean numeric columns
    def clean_numeric(val):
        if pd.isna(val) or val == 'null':
            return np.nan
        if isinstance(val, str):
            val = val.replace(',', '').replace('%', '').replace('$', '')
        return float(val)

    # Function to expand K/M suffixes in Volume
    def expand_volume(val):
        if pd.isna(val) or val == 'null':
            return np.nan
        if isinstance(val, str):
            val = val.strip()
            if val.endswith('K'):
                return float(val.replace('K', '').replace(',', '')) * 1000
            elif val.endswith('M'):
                return float(val.replace('M', '').replace(',', '')) * 1000000
            else:
                return float(val.replace(',', ''))
        return float(val)

    # Clean numeric columns
    numeric_cols = ['Price', 'Open', 'High', 'Low']
    for col in numeric_cols:
        if col in df_clean.columns:
            df_clean[col] = df_clean[col].apply(clean_numeric)

    # Handle Volume column
    if 'Vol.' in df_clean.columns:
        df_clean['Volume'] = df_clean['Vol.'].apply(expand_volume)
        df_clean = df_clean.drop('Vol.', axis=1)
    elif 'Volume' in df_clean.columns:
        df_clean['Volume'] = df_clean['Volume'].apply(expand_volume)

    # Clean Change %
    if 'Change %' in df_clean.columns:
        df_clean['Change %'] = df_clean['Change %'].apply(clean_numeric)

    return df_clean

btc_2024_clean = clean_dataset(btc_2024)
eth_2024_clean = clean_dataset(eth_2024)

# ---------- Clean Bitcoin Partial Dataset ----------
btc_partial_clean = btc_partial.copy()
btc_partial_clean['Date'] = pd.to_datetime(btc_partial_clean['Date'])

def clean_numeric_partial(val):
    if pd.isna(val) or val == 'null':
        return np.nan
    return float(val)

for col in ['Open', 'High', 'Low', 'Close', 'Adj Close', 'Volume']:
    if col in btc_partial_clean.columns:
        btc_partial_clean[col] = btc_partial_clean[col].apply(clean_numeric_partial)

# Rename Close to Price for consistency
if 'Close' in btc_partial_clean.columns:
    btc_partial_clean['Price'] = btc_partial_clean['Close']

# ---------- Linear Interpolation for Missing Values ----------
btc_partial_clean = btc_partial_clean.sort_values('Date')
btc_partial_clean['Price'] = btc_partial_clean['Price'].interpolate(method='linear')
btc_partial_clean['Open'] = btc_partial_clean['Open'].interpolate(method='linear')
btc_partial_clean['High'] = btc_partial_clean['High'].interpolate(method='linear')
btc_partial_clean['Low'] = btc_partial_clean['Low'].interpolate(method='linear')
btc_partial_clean['Volume'] = btc_partial_clean['Volume'].interpolate(method='linear')

# ---------- Merge All Datasets ----------
# Prepare Bitcoin data - combine both sources
btc_all = pd.concat([
    btc_partial_clean[['Date', 'Price', 'Open', 'High', 'Low', 'Volume']],
    btc_2024_clean[['Date', 'Price', 'Open', 'High', 'Low', 'Volume']]
]).drop_duplicates(subset=['Date'], keep='first').sort_values('Date')

# Prepare Ethereum data
eth_all = eth_2024_clean[['Date', 'Price', 'Open', 'High', 'Low', 'Volume']].copy()

# Merge on Date
merged = pd.merge(
    btc_all.rename(columns={
        'Price': 'BTC_Price',
        'Open': 'BTC_Open',
        'High': 'BTC_High',
        'Low': 'BTC_Low',
        'Volume': 'BTC_Volume'
    }),
    eth_all.rename(columns={
        'Price': 'ETH_Price',
        'Open': 'ETH_Open',
        'High': 'ETH_High',
        'Low': 'ETH_Low',
        'Volume': 'ETH_Volume'
    }),
    on='Date',
    how='inner'
)

# ---------- Add Temporal Features ----------
merged['Month'] = merged['Date'].dt.month
merged['Weekday'] = merged['Date'].dt.dayofweek

# Sort by date
merged = merged.sort_values('Date').reset_index(drop=True)

# ---------- Calculate Daily Logarithmic Returns ----------
merged['BTC_Log_Return'] = np.log(merged['BTC_Price'] / merged['BTC_Price'].shift(1))
merged['ETH_Log_Return'] = np.log(merged['ETH_Price'] / merged['ETH_Price'].shift(1))

# ---------- Calculate 7-Day Rolling Volatility for Bitcoin ----------
merged['BTC_Rolling_Volatility_7d'] = merged['BTC_Log_Return'].rolling(window=7).std()

# ---------- Calculate Volume Percentage Change ----------
merged['BTC_Volume_Pct_Change'] = merged['BTC_Volume'].pct_change() * 100
merged['ETH_Volume_Pct_Change'] = merged['ETH_Volume'].pct_change() * 100

# Drop NaN values for analysis
analysis_df = merged.dropna().copy()

# ---------- Pearson Correlation (Bitcoin vs Ethereum Returns) ----------
pearson_corr, pearson_pval = stats.pearsonr(analysis_df['BTC_Log_Return'], analysis_df['ETH_Log_Return'])

# ---------- Spearman Correlation (ETH Absolute Returns vs Volume % Change) ----------
analysis_df['ETH_Abs_Return'] = analysis_df['ETH_Log_Return'].abs()
spearman_corr, spearman_pval = spearmanr(analysis_df['ETH_Abs_Return'], analysis_df['ETH_Volume_Pct_Change'])

# ---------- Augmented Dickey-Fuller Test on Bitcoin Returns ----------
adf_result = adfuller(analysis_df['BTC_Log_Return'])
adf_statistic = adf_result[0]
adf_pvalue = adf_result[1]

# ---------- Train ARIMA Model for Bitcoin Returns ----------
# Find optimal ARIMA order using AIC
best_aic = np.inf
best_order = None
for p in range(0, 3):
    for d in range(0, 2):
        for q in range(0, 3):
            try:
                model = ARIMA(analysis_df['BTC_Log_Return'], order=(p, d, q))
                fitted = model.fit()
                if fitted.aic < best_aic:
                    best_aic = fitted.aic
                    best_order = (p, d, q)
            except:
                continue

# Fit best ARIMA model
arima_model = ARIMA(analysis_df['BTC_Log_Return'], order=best_order)
arima_fitted = arima_model.fit()
arima_predictions = arima_fitted.fittedvalues
arima_aic = arima_fitted.aic
arima_rmse = np.sqrt(mean_squared_error(analysis_df['BTC_Log_Return'], arima_predictions))

# ---------- Train ARIMAX Model with Ethereum Lagged Returns ----------
# Create lagged Ethereum returns as exogenous variable
analysis_df['ETH_Log_Return_Lag1'] = analysis_df['ETH_Log_Return'].shift(1)
arimax_df = analysis_df.dropna().copy()

# Fit ARIMAX model
arimax_model = SARIMAX(
    arimax_df['BTC_Log_Return'],
    exog=arimax_df[['ETH_Log_Return_Lag1']],
    order=best_order
)
arimax_fitted = arimax_model.fit(disp=False)
arimax_predictions = arimax_fitted.fittedvalues
arimax_aic = arimax_fitted.aic
arimax_rmse = np.sqrt(mean_squared_error(arimax_df['BTC_Log_Return'], arimax_predictions))

# ---------- Quantile Regression (Median) ----------
# Bitcoin returns using Volume as independent variable
qr_df = analysis_df[['BTC_Log_Return', 'BTC_Volume']].dropna()
X_qr = qr_df['BTC_Volume'].values.reshape(-1, 1)
y_qr = qr_df['BTC_Log_Return'].values

qr_model = QuantileRegressor(quantile=0.5, alpha=0, solver='highs')
qr_model.fit(X_qr, y_qr)
qr_coefficient = qr_model.coef_[0]
qr_intercept = qr_model.intercept_

# ---------- Granger Causality Test (2 lags) ----------
# Test if Bitcoin returns precede Ethereum returns
granger_df = analysis_df[['BTC_Log_Return', 'ETH_Log_Return']].dropna()
granger_result = grangercausalitytests(granger_df[['ETH_Log_Return', 'BTC_Log_Return']], maxlag=2, verbose=False)
granger_pvalue_lag2 = granger_result[2][0]['ssr_ftest'][1]

# ---------- KMeans Clustering on Ethereum Returns ----------
eth_returns_for_clustering = analysis_df['ETH_Log_Return'].values.reshape(-1, 1)
kmeans = KMeans(n_clusters=3, random_state=42, n_init=10)
clusters = kmeans.fit_predict(eth_returns_for_clustering)
silhouette_score_value = silhouette_score(eth_returns_for_clustering, clusters)

# ---------- Two-Sample T-Test ----------
ttest_statistic, ttest_pvalue = ttest_ind(
    analysis_df['BTC_Log_Return'],
    analysis_df['ETH_Log_Return']
)

# ---------- Change Point Detection (PELT) on Bitcoin Rolling Volatility ----------
volatility_series = analysis_df['BTC_Rolling_Volatility_7d'].dropna().values
algo = rpt.Pelt(model="rbf").fit(volatility_series)
changepoints = algo.predict(pen=3)
num_changepoints = len(changepoints) - 1  # Last point is always the end

# ---------- Chi-Square Test for High-Volatility vs High-Volume Days ----------
# Define high volatility and high volume as above median
median_volatility = analysis_df['BTC_Rolling_Volatility_7d'].median()
median_volume = analysis_df['BTC_Volume'].median()

analysis_df['High_Volatility'] = (analysis_df['BTC_Rolling_Volatility_7d'] > median_volatility).astype(int)
analysis_df['High_Volume'] = (analysis_df['BTC_Volume'] > median_volume).astype(int)

contingency_table = pd.crosstab(analysis_df['High_Volatility'], analysis_df['High_Volume'])
chi2_statistic, chi2_pvalue, chi2_dof, chi2_expected = stats.chi2_contingency(contingency_table)

# ---------- Generate Connected Scatterplot ----------
plot_df = analysis_df[['BTC_Rolling_Volatility_7d', 'BTC_Volume_Pct_Change']].dropna()

plt.figure(figsize=(12, 8))
plt.plot(plot_df['BTC_Volume_Pct_Change'], plot_df['BTC_Rolling_Volatility_7d'],
         marker='o', markersize=3, linestyle='-', linewidth=0.5, alpha=0.6)
plt.xlabel('Percentage Change in Volume (%)', fontsize=12)
plt.ylabel('Bitcoin 7-Day Rolling Volatility', fontsize=12)
plt.title('Bitcoin Rolling Volatility vs Volume % Change (Connected Time Series)', fontsize=14)
plt.grid(True, alpha=0.3)
plt.tight_layout()
plt.savefig('volatility_volume_scatterplot.png', dpi=300, bbox_inches='tight')
plt.close()

# ---------- Output Metrics ----------
metrics = {
    'Pearson_Correlation_BTC_ETH_Returns': pearson_corr,
    'Pearson_Correlation_PValue': pearson_pval,
    'Spearman_Correlation_ETH_AbsReturns_VolumePctChange': spearman_corr,
    'Spearman_Correlation_PValue': spearman_pval,
    'ADF_Test_Statistic': adf_statistic,
    'ADF_Test_PValue': adf_pvalue,
    'ARIMA_Order': str(best_order),
    'ARIMA_AIC': arima_aic,
    'ARIMA_RMSE': arima_rmse,
    'ARIMAX_AIC': arimax_aic,
    'ARIMAX_RMSE': arimax_rmse,
    'Quantile_Regression_Median_Coefficient': qr_coefficient,
    'Quantile_Regression_Median_Intercept': qr_intercept,
    'Granger_Causality_PValue_Lag2': granger_pvalue_lag2,
    'KMeans_Silhouette_Score': silhouette_score_value,
    'TTest_Statistic': ttest_statistic,
    'TTest_PValue': ttest_pvalue,
    'Number_of_Changepoints_PELT': num_changepoints,
    'ChiSquare_Statistic': chi2_statistic,
    'ChiSquare_PValue': chi2_pvalue,
    'ChiSquare_DegreesOfFreedom': chi2_dof
}

# Print only metric names and values
print("\n" + "="*60)
for metric, value in metrics.items():
    if isinstance(value, (int, np.integer)):
        print(f"{metric}: {value}")
    elif isinstance(value, str):
        print(f"{metric}: {value}")
    else:
        print(f"{metric}: {value:.10f}")
print("="*60)

# Save metrics to file
with open('metrics.txt', 'w') as f:
    for metric, value in metrics.items():
        if isinstance(value, (int, np.integer)):
            f.write(f"{metric}: {value}\n")
        elif isinstance(value, str):
            f.write(f"{metric}: {value}\n")
        else:
            f.write(f"{metric}: {value:.10f}\n")

print("\nScatterplot saved as: volatility_volume_scatterplot.png")
