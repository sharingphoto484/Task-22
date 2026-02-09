# ==========================================
# Integrated Crypto Market Analysis Script
# Requirements: pandas, numpy, matplotlib, scipy, scikit-learn, statsmodels, scikit-posthocs
# Input files: Ethereum 2024.xlsx, Bitcoin 2024.xlsx, BTC-USD (2014-2024).xlsx
# Output files: gmm_clustering.png, arima_forecast.png, lasso_scatter.png,
#               kruskal_boxplot.png, cv_histogram.png, moving_avg_line.png,
#               qda_confusion_matrix.png
# ==========================================

import pandas as pd
import numpy as np
import matplotlib
matplotlib.use('Agg')
import matplotlib.pyplot as plt
import warnings
warnings.filterwarnings('ignore')

from sklearn.mixture import GaussianMixture
from sklearn.preprocessing import StandardScaler
from sklearn.metrics import (silhouette_score, mean_absolute_error,
                             accuracy_score, f1_score, precision_score,
                             confusion_matrix, ConfusionMatrixDisplay)
from sklearn.linear_model import Lasso
from sklearn.discriminant_analysis import (LinearDiscriminantAnalysis,
                                           QuadraticDiscriminantAnalysis)
from sklearn.naive_bayes import GaussianNB
from sklearn.linear_model import LinearRegression
from statsmodels.tsa.arima.model import ARIMA
from scipy.stats import kruskal, friedmanchisquare
import scikit_posthocs as sp

# ---------- Helper: Convert String Columns to Numeric ----------
def to_numeric_col(series):
    """Remove commas and percentage signs, convert to float."""
    return pd.to_numeric(series.astype(str).str.replace(',', '', regex=False).str.replace('%', '', regex=False), errors='coerce')

def convert_volume(series):
    """Convert volume strings like '81.30K' to numeric (multiply K by 1000)."""
    s = series.astype(str).str.strip()
    mask_k = s.str.upper().str.endswith('K')
    s_clean = s.str.replace('K', '', case=False, regex=False).str.replace('k', '', regex=False).str.replace(',', '', regex=False)
    vals = pd.to_numeric(s_clean, errors='coerce')
    vals[mask_k] = vals[mask_k] * 1000
    return vals

# ---------- Load Data ----------
print("=" * 60)
print("CRYPTO MARKET ANALYSIS - KEY OUTPUTS")
print("=" * 60)

eth_df = pd.read_excel('Ethereum 2024.xlsx')
btc_df = pd.read_excel('Bitcoin 2024.xlsx')
btc_usd_df = pd.read_excel('BTC-USD (2014-2024).xlsx')

# ---------- Section 1: Gaussian Mixture Model Clustering (Ethereum) ----------
print("\n--- 1. Gaussian Mixture Model Clustering (Ethereum 2024) ---")

gmm_df = eth_df.copy()
gmm_df['Price'] = to_numeric_col(gmm_df['Price'])
gmm_df['High'] = to_numeric_col(gmm_df['High'])

# Remove observations where Price is missing OR High is missing
gmm_df = gmm_df.dropna(subset=['Price', 'High'])

# Sort by Date ascending
gmm_df['Date'] = pd.to_datetime(gmm_df['Date'])
gmm_df = gmm_df.sort_values('Date').reset_index(drop=True)

# Standardize Price and High
scaler = StandardScaler()
gmm_features = scaler.fit_transform(gmm_df[['Price', 'High']])

# Fit GMM with 2 components
gmm = GaussianMixture(n_components=2, covariance_type='full', max_iter=100, random_state=42)
gmm_labels = gmm.fit_predict(gmm_features)

# Silhouette score
sil_score = silhouette_score(gmm_features, gmm_labels)
print(f"Silhouette Score: {sil_score:.4f}")

# ---------- Section 2: ARIMA Time Series Forecasting (BTC-USD) ----------
print("\n--- 2. ARIMA Forecasting (BTC-USD 2014-2024) ---")

arima_df = btc_usd_df.copy()
# Remove observations where Close is missing
arima_df = arima_df.dropna(subset=['Close'])

# Sort by Date ascending
arima_df['Date'] = pd.to_datetime(arima_df['Date'])
arima_df = arima_df.sort_values('Date').reset_index(drop=True)

close_series = arima_df['Close'].values

# Fit ARIMA(1,1,1)
model = ARIMA(close_series, order=(1, 1, 1))
arima_result = model.fit()

# Generate 10-step ahead forecast
forecast = arima_result.forecast(steps=10)

# MAPE between final 10 actual observations and forecast
actual_last10 = close_series[-10:]
mape = np.mean(np.abs((actual_last10 - forecast) / actual_last10)) * 100
print(f"MAPE: {mape:.3f}")

# ---------- Section 3: Lasso Regression (BTC-USD) ----------
print("\n--- 3. Lasso Regression (BTC-USD 2014-2024) ---")

lasso_df = btc_usd_df.copy()
# Remove observations where Close, Open, High, or Volume is missing
lasso_df = lasso_df.dropna(subset=['Close', 'Open', 'High', 'Volume'])

# Sort by Date ascending
lasso_df['Date'] = pd.to_datetime(lasso_df['Date'])
lasso_df = lasso_df.sort_values('Date').reset_index(drop=True)

# Split 75% train, 25% test
n_lasso = len(lasso_df)
split_lasso = int(n_lasso * 0.75)
train_lasso = lasso_df.iloc[:split_lasso]
test_lasso = lasso_df.iloc[split_lasso:]

X_train_lasso = train_lasso[['Open', 'High', 'Volume']]
y_train_lasso = train_lasso['Close']
X_test_lasso = test_lasso[['Open', 'High', 'Volume']]
y_test_lasso = test_lasso['Close']

# Fit Lasso
lasso_model = Lasso(alpha=0.1, max_iter=5000, random_state=42)
lasso_model.fit(X_train_lasso, y_train_lasso)
lasso_pred = lasso_model.predict(X_test_lasso)

lasso_mae = mean_absolute_error(y_test_lasso, lasso_pred)
print(f"Testing MAE: {lasso_mae:.2f}")

# ---------- Section 4: Linear Discriminant Analysis (Bitcoin 2024) ----------
print("\n--- 4. Linear Discriminant Analysis (Bitcoin 2024) ---")

lda_df = btc_df.copy()
lda_df['Price'] = to_numeric_col(lda_df['Price'])
lda_df['Open'] = to_numeric_col(lda_df['Open'])

# Remove observations where Price or Open is missing
lda_df = lda_df.dropna(subset=['Price', 'Open'])

# Create binary target: 1 if Price > Open, else 0
lda_df['direction'] = (lda_df['Price'] > lda_df['Open']).astype(int)

# Sort by Date ascending
lda_df['Date'] = pd.to_datetime(lda_df['Date'])
lda_df = lda_df.sort_values('Date').reset_index(drop=True)

# Create lagged Price differences at lag 1
lda_df['price_diff_lag1'] = lda_df['Price'].diff(1)

# Remove observations with missing lagged values
lda_df = lda_df.dropna(subset=['price_diff_lag1'])

# Split 70% train, 30% test
n_lda = len(lda_df)
split_lda = int(n_lda * 0.70)
train_lda = lda_df.iloc[:split_lda]
test_lda = lda_df.iloc[split_lda:]

X_train_lda = train_lda[['price_diff_lag1']]
y_train_lda = train_lda['direction']
X_test_lda = test_lda[['price_diff_lag1']]
y_test_lda = test_lda['direction']

# Fit LDA
lda_model = LinearDiscriminantAnalysis()
lda_model.fit(X_train_lda, y_train_lda)
lda_pred = lda_model.predict(X_test_lda)

lda_accuracy = accuracy_score(y_test_lda, lda_pred)
print(f"Testing Accuracy: {lda_accuracy:.4f}")

# ---------- Section 5: Gaussian Naive Bayes (Ethereum 2024) ----------
print("\n--- 5. Gaussian Naive Bayes (Ethereum 2024) ---")

gnb_df = eth_df.copy()
gnb_df['High'] = to_numeric_col(gnb_df['High'])
gnb_df['Low'] = to_numeric_col(gnb_df['Low'])

# Remove observations where High or Low is missing
gnb_df = gnb_df.dropna(subset=['High', 'Low'])

# Calculate daily range
gnb_df['range'] = gnb_df['High'] - gnb_df['Low']

# Compute median range
median_range = gnb_df['range'].median()

# Create binary target: 1 if range > median, else 0
gnb_df['volatility'] = (gnb_df['range'] > median_range).astype(int)

# Sort by Date ascending
gnb_df['Date'] = pd.to_datetime(gnb_df['Date'])
gnb_df = gnb_df.sort_values('Date').reset_index(drop=True)

# Split 65% train, 35% test
n_gnb = len(gnb_df)
split_gnb = int(n_gnb * 0.65)
train_gnb = gnb_df.iloc[:split_gnb]
test_gnb = gnb_df.iloc[split_gnb:]

X_train_gnb = train_gnb[['range']]
y_train_gnb = train_gnb['volatility']
X_test_gnb = test_gnb[['range']]
y_test_gnb = test_gnb['volatility']

# Fit Gaussian Naive Bayes
gnb_model = GaussianNB()
gnb_model.fit(X_train_gnb, y_train_gnb)
gnb_pred = gnb_model.predict(X_test_gnb)

gnb_f1 = f1_score(y_test_gnb, gnb_pred)
print(f"Testing F1 Score: {gnb_f1:.4f}")

# ---------- Section 6: Quadratic Discriminant Analysis (Bitcoin 2024) ----------
print("\n--- 6. Quadratic Discriminant Analysis (Bitcoin 2024) ---")

qda_df = btc_df.copy()
qda_df['Vol.'] = convert_volume(qda_df['Vol.'])

# Remove observations where Vol. is missing
qda_df = qda_df.dropna(subset=['Vol.'])

# Compute median volume
median_vol = qda_df['Vol.'].median()

# Create binary target: 1 if volume > median, else 0
qda_df['volume_class'] = (qda_df['Vol.'] > median_vol).astype(int)

# Sort by Date ascending
qda_df['Date'] = pd.to_datetime(qda_df['Date'])
qda_df = qda_df.sort_values('Date').reset_index(drop=True)

# Split 60% train, 40% test
n_qda = len(qda_df)
split_qda = int(n_qda * 0.60)
train_qda = qda_df.iloc[:split_qda]
test_qda = qda_df.iloc[split_qda:]

X_train_qda = train_qda[['Vol.']]
y_train_qda = train_qda['volume_class']
X_test_qda = test_qda[['Vol.']]
y_test_qda = test_qda['volume_class']

# Fit QDA
qda_model = QuadraticDiscriminantAnalysis()
qda_model.fit(X_train_qda, y_train_qda)
qda_pred = qda_model.predict(X_test_qda)

qda_precision = precision_score(y_test_qda, qda_pred)
print(f"Testing Precision: {qda_precision:.4f}")

# ---------- Section 7: Kruskal-Wallis Test (BTC-USD) ----------
print("\n--- 7. Kruskal-Wallis Test (BTC-USD 2014-2024) ---")

kw_df = btc_usd_df.copy()
kw_df = kw_df.dropna(subset=['Close'])
kw_df['Date'] = pd.to_datetime(kw_df['Date'])
kw_df['year'] = kw_df['Date'].dt.year

# Retain only years with at least 30 observations
year_counts = kw_df['year'].value_counts()
valid_years = year_counts[year_counts >= 30].index.tolist()
valid_years.sort()
kw_df = kw_df[kw_df['year'].isin(valid_years)]

# Extract Close values per year group
year_groups = [kw_df[kw_df['year'] == y]['Close'].values for y in valid_years]

# Kruskal-Wallis test
h_stat, kw_pval = kruskal(*year_groups)
print(f"H Statistic: {h_stat:.3f}")

# ---------- Section 8: Friedman Test (Three Cryptocurrencies) ----------
print("\n--- 8. Friedman Test (Three Cryptocurrencies, 2024) ---")

# Prepare Ethereum 2024 prices
fr_eth = eth_df.copy()
fr_eth['Price'] = to_numeric_col(fr_eth['Price'])
fr_eth['Date'] = pd.to_datetime(fr_eth['Date'])
fr_eth = fr_eth.dropna(subset=['Price'])
fr_eth = fr_eth[['Date', 'Price']].rename(columns={'Price': 'ETH_Price'})

# Prepare Bitcoin 2024 prices
fr_btc = btc_df.copy()
fr_btc['Price'] = to_numeric_col(fr_btc['Price'])
fr_btc['Date'] = pd.to_datetime(fr_btc['Date'])
fr_btc = fr_btc.dropna(subset=['Price'])
fr_btc = fr_btc[['Date', 'Price']].rename(columns={'Price': 'BTC_Price'})

# Prepare BTC-USD Close for 2024 dates
fr_btcusd = btc_usd_df.copy()
fr_btcusd['Date'] = pd.to_datetime(fr_btcusd['Date'])
fr_btcusd = fr_btcusd.dropna(subset=['Close'])
fr_btcusd_2024 = fr_btcusd[fr_btcusd['Date'].dt.year == 2024][['Date', 'Close']].rename(columns={'Close': 'BTCUSD_Close'})

# Merge on Date
fr_merged = fr_eth.merge(fr_btc, on='Date').merge(fr_btcusd_2024, on='Date')
fr_merged = fr_merged.sort_values('Date').reset_index(drop=True)

# Friedman test
friedman_stat, friedman_pval = friedmanchisquare(
    fr_merged['ETH_Price'].values,
    fr_merged['BTC_Price'].values,
    fr_merged['BTCUSD_Close'].values
)
print(f"Friedman Chi-Square Statistic: {friedman_stat:.3f}")

# ---------- Section 9: Coefficient of Variation Ratio ----------
print("\n--- 9. Coefficient of Variation Ratio (Bitcoin vs Ethereum) ---")

# Bitcoin CV
cv_btc_df = btc_df.copy()
cv_btc_df['Price'] = to_numeric_col(cv_btc_df['Price'])
cv_btc_df = cv_btc_df.dropna(subset=['Price'])
btc_cv = cv_btc_df['Price'].std() / cv_btc_df['Price'].mean()

# Ethereum CV
cv_eth_df = eth_df.copy()
cv_eth_df['Price'] = to_numeric_col(cv_eth_df['Price'])
cv_eth_df = cv_eth_df.dropna(subset=['Price'])
eth_cv = cv_eth_df['Price'].std() / cv_eth_df['Price'].mean()

cv_ratio = btc_cv / eth_cv
print(f"Volatility Ratio (BTC CV / ETH CV): {cv_ratio:.4f}")

# ---------- Section 10: Autoregressive Model (Ethereum) ----------
print("\n--- 10. Autoregressive Model (Ethereum 2024) ---")

ar_df = eth_df.copy()
ar_df['Price'] = to_numeric_col(ar_df['Price'])
ar_df = ar_df.dropna(subset=['Price'])

# Sort by Date ascending
ar_df['Date'] = pd.to_datetime(ar_df['Date'])
ar_df = ar_df.sort_values('Date').reset_index(drop=True)

# Create lagged features
ar_df['lag1'] = ar_df['Price'].shift(1)
ar_df['lag2'] = ar_df['Price'].shift(2)
ar_df['lag3'] = ar_df['Price'].shift(3)

# Remove observations with missing lagged values
ar_df = ar_df.dropna(subset=['lag1', 'lag2', 'lag3'])

# Split 70% train, 30% test
n_ar = len(ar_df)
split_ar = int(n_ar * 0.70)
train_ar = ar_df.iloc[:split_ar]
test_ar = ar_df.iloc[split_ar:]

X_train_ar = train_ar[['lag1', 'lag2', 'lag3']]
y_train_ar = train_ar['Price']
X_test_ar = test_ar[['lag1', 'lag2', 'lag3']]
y_test_ar = test_ar['Price']

# Fit OLS
ar_model = LinearRegression()
ar_model.fit(X_train_ar, y_train_ar)
ar_pred = ar_model.predict(X_test_ar)

# R-squared on test set
ar_r2 = ar_model.score(X_test_ar, y_test_ar)
print(f"Testing R-squared: {ar_r2:.4f}")

# ---------- Section 11: Dunn Test Post-Hoc (BTC-USD) ----------
print("\n--- 11. Dunn Test Post-Hoc (BTC-USD 2014-2024) ---")

# Reuse kw_df and valid_years from Kruskal-Wallis
dunn_df = kw_df[['Close', 'year']].copy()

# Conduct pairwise Dunn tests with Bonferroni correction
dunn_result = sp.posthoc_dunn(dunn_df, val_col='Close', group_col='year', p_adjust='bonferroni')

# Report adjusted p-value for 2020 vs 2021
dunn_pval_2020_2021 = dunn_result.loc[2020, 2021]
print(f"Adjusted p-value (2020 vs 2021): {dunn_pval_2020_2021:.4f}")

# ---------- Section 12: Moving Average Crossover Strategy (Bitcoin 2024) ----------
print("\n--- 12. Moving Average Crossover Strategy (Bitcoin 2024) ---")

ma_df = btc_df.copy()
ma_df['Price'] = to_numeric_col(ma_df['Price'])
ma_df = ma_df.dropna(subset=['Price'])

# Sort by Date ascending
ma_df['Date'] = pd.to_datetime(ma_df['Date'])
ma_df = ma_df.sort_values('Date').reset_index(drop=True)

# Calculate moving averages
ma_df['SMA5'] = ma_df['Price'].rolling(window=5).mean()
ma_df['SMA20'] = ma_df['Price'].rolling(window=20).mean()

# Generate binary signal: 1 if SMA5 > SMA20 (buy), else 0 (sell)
ma_df['signal'] = (ma_df['SMA5'] > ma_df['SMA20']).astype(int)

# Count buy signals (excluding NaN rows)
ma_valid = ma_df.dropna(subset=['SMA5', 'SMA20'])
buy_count = int(ma_valid['signal'].sum())
print(f"Buy Signals Count: {buy_count}")

# ==========================================
# VISUALIZATIONS
# ==========================================

# ---------- Plot 1: GMM Clustering Scatter ----------
fig, ax = plt.subplots(figsize=(8, 6))
scatter = ax.scatter(gmm_features[:, 0], gmm_features[:, 1], c=gmm_labels, cmap='viridis', alpha=0.7, edgecolors='k', linewidth=0.3)
ax.set_xlabel('Standardized Price')
ax.set_ylabel('Standardized High')
ax.set_title('Gaussian Mixture Model Clustering (Ethereum 2024)')
plt.colorbar(scatter, ax=ax, label='Component')
plt.tight_layout()
plt.savefig('gmm_clustering.png', dpi=150)
plt.close()

# ---------- Plot 2: ARIMA Forecast Line ----------
fig, ax = plt.subplots(figsize=(10, 6))
ax.plot(range(len(close_series)), close_series, label='Actual Close', color='steelblue')
forecast_idx = range(len(close_series), len(close_series) + 10)
ax.plot(forecast_idx, forecast, label='Forecast', color='red', linestyle='--', marker='o', markersize=4)
ax.set_xlabel('Time Index')
ax.set_ylabel('Close Price (USD)')
ax.set_title('ARIMA(1,1,1) Forecast (BTC-USD)')
ax.legend()
plt.tight_layout()
plt.savefig('arima_forecast.png', dpi=150)
plt.close()

# ---------- Plot 3: Lasso Regression Scatter ----------
fig, ax = plt.subplots(figsize=(8, 6))
ax.scatter(y_test_lasso, lasso_pred, alpha=0.5, edgecolors='k', linewidth=0.3, color='coral')
min_val = min(y_test_lasso.min(), lasso_pred.min())
max_val = max(y_test_lasso.max(), lasso_pred.max())
ax.plot([min_val, max_val], [min_val, max_val], 'k--', linewidth=1, label='Perfect Prediction')
ax.set_xlabel('Actual Close')
ax.set_ylabel('Predicted Close')
ax.set_title('Lasso Regression: Actual vs Predicted (BTC-USD)')
ax.legend()
plt.tight_layout()
plt.savefig('lasso_scatter.png', dpi=150)
plt.close()

# ---------- Plot 4: Kruskal-Wallis Box Plot ----------
fig, ax = plt.subplots(figsize=(12, 6))
box_data = [kw_df[kw_df['year'] == y]['Close'].values for y in valid_years]
bp = ax.boxplot(box_data, labels=[str(y) for y in valid_years], patch_artist=True)
colors = plt.cm.tab10(np.linspace(0, 1, len(valid_years)))
for patch, color in zip(bp['boxes'], colors):
    patch.set_facecolor(color)
ax.set_xlabel('Year')
ax.set_ylabel('Close Price (USD)')
ax.set_title('Kruskal-Wallis: Close Price Distribution by Year (BTC-USD)')
plt.tight_layout()
plt.savefig('kruskal_boxplot.png', dpi=150)
plt.close()

# ---------- Plot 5: Coefficient of Variation Histogram ----------
fig, axes = plt.subplots(1, 2, figsize=(12, 5))
axes[0].hist(cv_btc_df['Price'].values, bins=30, color='orange', edgecolor='black', alpha=0.7)
axes[0].set_xlabel('Price (USD)')
axes[0].set_ylabel('Frequency')
axes[0].set_title(f'Bitcoin 2024 Prices (CV={btc_cv:.4f})')
axes[1].hist(cv_eth_df['Price'].values, bins=30, color='skyblue', edgecolor='black', alpha=0.7)
axes[1].set_xlabel('Price (USD)')
axes[1].set_ylabel('Frequency')
axes[1].set_title(f'Ethereum 2024 Prices (CV={eth_cv:.4f})')
fig.suptitle(f'Coefficient of Variation Comparison (Ratio: {cv_ratio:.4f})', fontsize=13)
plt.tight_layout()
plt.savefig('cv_histogram.png', dpi=150)
plt.close()

# ---------- Plot 6: Moving Average Line Plot ----------
fig, ax = plt.subplots(figsize=(12, 6))
ax.plot(ma_df['Date'], ma_df['Price'], label='Price', color='steelblue', alpha=0.8)
ax.plot(ma_df['Date'], ma_df['SMA5'], label='SMA-5 (Short)', color='orange', linewidth=1.5)
ax.plot(ma_df['Date'], ma_df['SMA20'], label='SMA-20 (Long)', color='red', linewidth=1.5)
ax.set_xlabel('Date')
ax.set_ylabel('Price (USD)')
ax.set_title('Moving Average Crossover Strategy (Bitcoin 2024)')
ax.legend()
plt.xticks(rotation=45)
plt.tight_layout()
plt.savefig('moving_avg_line.png', dpi=150)
plt.close()

# ---------- Plot 7: QDA Confusion Matrix Heatmap ----------
fig, ax = plt.subplots(figsize=(6, 5))
cm = confusion_matrix(y_test_qda, qda_pred)
disp = ConfusionMatrixDisplay(confusion_matrix=cm, display_labels=['Low Volume', 'High Volume'])
disp.plot(ax=ax, cmap='Blues', colorbar=True)
ax.set_xlabel('Predicted Volume Class')
ax.set_ylabel('Actual Volume Class')
ax.set_title('QDA Confusion Matrix (Bitcoin 2024)')
plt.tight_layout()
plt.savefig('qda_confusion_matrix.png', dpi=150)
plt.close()

print("\n" + "=" * 60)
print("All plots saved successfully.")
print("=" * 60)
