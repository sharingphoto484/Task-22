# ==========================================
# Integrated Tesla Stock Analysis Script
# Requirements: pandas, numpy, matplotlib, scikit-learn, statsmodels, openpyxl
# Input files: tsla_2025.xlsx, TSLA-2.xlsx, HistoricalData_1726367135218.xlsx
# Output files: rf_scatter.png, gb_line.png, dbscan_box.png,
#               exp_smoothing_line.png, daily_returns_hist.png
# ==========================================

import warnings
warnings.filterwarnings("ignore")

import numpy as np
import pandas as pd
import matplotlib
matplotlib.use("Agg")
import matplotlib.pyplot as plt
from sklearn.ensemble import (
    RandomForestRegressor,
    GradientBoostingRegressor,
    ExtraTreesClassifier,
    IsolationForest,
)
from sklearn.neighbors import KNeighborsRegressor
from sklearn.cluster import DBSCAN
from sklearn.preprocessing import StandardScaler
from sklearn.metrics import (
    mean_absolute_error,
    r2_score,
    matthews_corrcoef,
    mean_squared_error,
)
from statsmodels.tsa.vector_ar.var_model import VAR
from statsmodels.tsa.holtwinters import ExponentialSmoothing


# ---------- 1. Random Forest Regression (tsla_2025.xlsx) ----------

df_rf = pd.read_excel("tsla_2025.xlsx")

# Remove observations where Close, Open, High, Low, or Volume is missing
df_rf = df_rf.dropna(subset=["Close", "Open", "High", "Low", "Volume"])

# Sort by Date ascending
df_rf["Date"] = pd.to_datetime(df_rf["Date"])
df_rf = df_rf.sort_values("Date").reset_index(drop=True)

# Create lagged features
df_rf["lag_close_1"] = df_rf["Close"].shift(1)
df_rf["lag_volume_1"] = df_rf["Volume"].shift(1)
df_rf["lag_high_1"] = df_rf["High"].shift(1)

# Remove observations with missing lagged values
df_rf = df_rf.dropna(subset=["lag_close_1", "lag_volume_1", "lag_high_1"]).reset_index(drop=True)

# Split 75/25 preserving temporal order
split_idx_rf = int(len(df_rf) * 0.75)
train_rf = df_rf.iloc[:split_idx_rf]
test_rf = df_rf.iloc[split_idx_rf:]

features_rf = ["lag_close_1", "lag_volume_1", "lag_high_1", "Open", "High", "Low"]
X_train_rf = train_rf[features_rf]
y_train_rf = train_rf["Close"]
X_test_rf = test_rf[features_rf]
y_test_rf = test_rf["Close"]

# Fit Random Forest
rf_model = RandomForestRegressor(
    n_estimators=100, max_depth=10, min_samples_split=5, random_state=42
)
rf_model.fit(X_train_rf, y_train_rf)
y_pred_rf = rf_model.predict(X_test_rf)

rf_mae = mean_absolute_error(y_test_rf, y_pred_rf)
print(f"Random Forest MAE (test): {rf_mae:.4f}")


# ---------- 2. Gradient Boosting Regression (TSLA-2.xlsx) ----------

df_gb = pd.read_excel("TSLA-2.xlsx")

# Remove observations where Adj Close, Open, or Volume is missing
df_gb = df_gb.dropna(subset=["Adj Close", "Open", "Volume"])

# Sort by Date ascending
df_gb["Date"] = pd.to_datetime(df_gb["Date"])
df_gb = df_gb.sort_values("Date").reset_index(drop=True)

# Split 80/20 preserving temporal order
split_idx_gb = int(len(df_gb) * 0.80)
train_gb = df_gb.iloc[:split_idx_gb]
test_gb = df_gb.iloc[split_idx_gb:]

features_gb = ["Open", "Volume"]
X_train_gb = train_gb[features_gb]
y_train_gb = train_gb["Adj Close"]
X_test_gb = test_gb[features_gb]
y_test_gb = test_gb["Adj Close"]

# Fit Gradient Boosting
gb_model = GradientBoostingRegressor(
    learning_rate=0.1,
    n_estimators=150,
    max_depth=4,
    subsample=0.8,
    random_state=42,
)
gb_model.fit(X_train_gb, y_train_gb)
y_pred_gb = gb_model.predict(X_test_gb)

gb_r2 = r2_score(y_test_gb, y_pred_gb)
print(f"Gradient Boosting R-squared (test): {gb_r2:.4f}")


# ---------- 3. Extra Trees Classification (HistoricalData_1726367135218.xlsx) ----------

df_et = pd.read_excel("HistoricalData_1726367135218.xlsx")

# Remove observations where Close/Last or Open is missing
df_et = df_et.dropna(subset=["Close/Last", "Open"])

# Convert from string to numeric by removing dollar sign prefix
df_et["Close/Last"] = df_et["Close/Last"].astype(str).str.replace("$", "", regex=False).astype(float)
df_et["Open"] = df_et["Open"].astype(str).str.replace("$", "", regex=False).astype(float)

# Create binary target: 1 if Close/Last > Open, else 0
df_et["direction"] = (df_et["Close/Last"] > df_et["Open"]).astype(int)

# Sort by Date ascending
df_et["Date"] = pd.to_datetime(df_et["Date"])
df_et = df_et.sort_values("Date").reset_index(drop=True)

# Split 65/35 preserving temporal order
split_idx_et = int(len(df_et) * 0.65)
train_et = df_et.iloc[:split_idx_et]
test_et = df_et.iloc[split_idx_et:]

X_train_et = train_et[["Open"]]
y_train_et = train_et["direction"]
X_test_et = test_et[["Open"]]
y_test_et = test_et["direction"]

# Fit Extra Trees Classifier
et_model = ExtraTreesClassifier(
    n_estimators=200, max_depth=8, min_samples_leaf=3, random_state=42
)
et_model.fit(X_train_et, y_train_et)
y_pred_et = et_model.predict(X_test_et)

et_mcc = matthews_corrcoef(y_test_et, y_pred_et)
print(f"Extra Trees MCC (test): {et_mcc:.4f}")


# ---------- 4. Isolation Forest Anomaly Detection (tsla_2025.xlsx) ----------

df_if = pd.read_excel("tsla_2025.xlsx")

# Remove observations where Volume is missing
df_if = df_if.dropna(subset=["Volume"])

# Sort by Date ascending
df_if["Date"] = pd.to_datetime(df_if["Date"])
df_if = df_if.sort_values("Date").reset_index(drop=True)

# Fit Isolation Forest
iso_model = IsolationForest(
    contamination=0.05, n_estimators=100, random_state=42
)
df_if["anomaly"] = iso_model.fit_predict(df_if[["Volume"]])

# Count anomalies (label == -1)
anomaly_count = int((df_if["anomaly"] == -1).sum())
print(f"Isolation Forest anomaly count: {anomaly_count}")


# ---------- 5. DBSCAN Density Clustering (TSLA-2.xlsx) ----------

df_db = pd.read_excel("TSLA-2.xlsx")

# Remove observations where High, Low, or Volume is missing
df_db = df_db.dropna(subset=["High", "Low", "Volume"])

# Sort by Date ascending
df_db["Date"] = pd.to_datetime(df_db["Date"])
df_db = df_db.sort_values("Date").reset_index(drop=True)

# Calculate daily range
df_db["range"] = df_db["High"] - df_db["Low"]

# Standardize range and Volume to zero mean unit variance
scaler = StandardScaler()
df_db[["std_range", "std_volume"]] = scaler.fit_transform(df_db[["range", "Volume"]])

# Fit DBSCAN
dbscan_model = DBSCAN(eps=0.5, min_samples=10)
df_db["cluster"] = dbscan_model.fit_predict(df_db[["std_range", "std_volume"]])

# Count unique clusters excluding noise (-1)
cluster_count = int(df_db[df_db["cluster"] != -1]["cluster"].nunique())
print(f"DBSCAN cluster count (excl. noise): {cluster_count}")


# ---------- 6. Vector Autoregression (tsla_2025.xlsx) ----------

df_var = pd.read_excel("tsla_2025.xlsx")

# Remove observations where Close or Volume is missing
df_var = df_var.dropna(subset=["Close", "Volume"])

# Sort by Date ascending
df_var["Date"] = pd.to_datetime(df_var["Date"])
df_var = df_var.sort_values("Date").reset_index(drop=True)

# Extract most recent 500 observations
df_var_sub = df_var.tail(500).reset_index(drop=True)

# Fit VAR model with lag order 2
var_data = df_var_sub[["Close", "Volume"]].astype(float)
var_model = VAR(var_data)
var_result = var_model.fit(maxlags=2, ic=None)

# Extract coefficient for Close at lag 1 predicting Close
# In the VAR coefficient matrix, the equation for 'Close' row, 'L1.Close' column
var_coef = var_result.params.loc["L1.Close", "Close"]
print(f"VAR coefficient (Close L1 -> Close): {var_coef:.4f}")


# ---------- 7. Triple Exponential Smoothing (HistoricalData_1726367135218.xlsx) ----------

df_hw = pd.read_excel("HistoricalData_1726367135218.xlsx")

# Remove observations where Close/Last is missing
df_hw = df_hw.dropna(subset=["Close/Last"])

# Convert Close/Last from string to numeric by removing dollar sign
df_hw["Close/Last"] = df_hw["Close/Last"].astype(str).str.replace("$", "", regex=False).astype(float)

# Sort by Date ascending
df_hw["Date"] = pd.to_datetime(df_hw["Date"])
df_hw = df_hw.sort_values("Date").reset_index(drop=True)

# Fit Holt-Winters with additive trend, additive seasonality, seasonal period 7
hw_model = ExponentialSmoothing(
    df_hw["Close/Last"],
    trend="add",
    seasonal="add",
    seasonal_periods=7,
)
hw_result = hw_model.fit()

# Generate 7-step ahead forecast
hw_forecast = hw_result.forecast(steps=7)
hw_forecast_mean = hw_forecast.mean()
print(f"Holt-Winters 7-step forecast mean: {hw_forecast_mean:.4f}")


# ---------- 8. KNN Regression (TSLA-2.xlsx) ----------

df_knn = pd.read_excel("TSLA-2.xlsx")

# Remove observations where Close, Open, or High is missing
df_knn = df_knn.dropna(subset=["Close", "Open", "High"])

# Sort by Date ascending
df_knn["Date"] = pd.to_datetime(df_knn["Date"])
df_knn = df_knn.sort_values("Date").reset_index(drop=True)

# Split 70/30 preserving temporal order
split_idx_knn = int(len(df_knn) * 0.70)
train_knn = df_knn.iloc[:split_idx_knn]
test_knn = df_knn.iloc[split_idx_knn:]

features_knn = ["Open", "High"]
X_train_knn = train_knn[features_knn]
y_train_knn = train_knn["Close"]
X_test_knn = test_knn[features_knn]
y_test_knn = test_knn["Close"]

# Fit KNN regressor
knn_model = KNeighborsRegressor(n_neighbors=5, weights="uniform")
knn_model.fit(X_train_knn, y_train_knn)
y_pred_knn = knn_model.predict(X_test_knn)

knn_mse = mean_squared_error(y_test_knn, y_pred_knn)
print(f"KNN MSE (test): {knn_mse:.4f}")


# ---------- 9. Sharpe Ratio (tsla_2025.xlsx) ----------

df_sr = pd.read_excel("tsla_2025.xlsx")

# Remove observations where Close is missing
df_sr = df_sr.dropna(subset=["Close"])

# Sort by Date ascending
df_sr["Date"] = pd.to_datetime(df_sr["Date"])
df_sr = df_sr.sort_values("Date").reset_index(drop=True)

# Calculate daily returns as percentage change in Close
df_sr["daily_return"] = df_sr["Close"].pct_change()

# Remove first observation with undefined return
df_sr = df_sr.iloc[1:].reset_index(drop=True)

# Compute Sharpe ratio
mean_return = df_sr["daily_return"].mean()
std_return = df_sr["daily_return"].std()
sharpe_ratio = (mean_return / std_return) * np.sqrt(252)
print(f"Annualized Sharpe Ratio: {sharpe_ratio:.4f}")


# ---------- 10. Optimal Buy-Hold Period (tsla_2025.xlsx) ----------

df_bh = pd.read_excel("tsla_2025.xlsx")

# Extract Close prices, sort by Date
df_bh = df_bh.dropna(subset=["Close"])
df_bh["Date"] = pd.to_datetime(df_bh["Date"])
df_bh = df_bh.sort_values("Date").reset_index(drop=True)

# Calculate rolling returns for different timeframes
df_bh["roll_30"] = df_bh["Close"].pct_change(periods=30)
df_bh["roll_90"] = df_bh["Close"].pct_change(periods=90)
df_bh["roll_180"] = df_bh["Close"].pct_change(periods=180)

# Compute mean return for each timeframe
mean_30 = df_bh["roll_30"].dropna().mean()
mean_90 = df_bh["roll_90"].dropna().mean()
mean_180 = df_bh["roll_180"].dropna().mean()

# Identify timeframe with maximum mean return
returns_map = {30: mean_30, 90: mean_90, 180: mean_180}
optimal_period = max(returns_map, key=returns_map.get)
print(f"Optimal holding period: {optimal_period}")


# ---------- Plot 1: Random Forest Scatter (Actual vs Predicted Close) ----------

fig, ax = plt.subplots(figsize=(8, 6))
ax.scatter(y_test_rf, y_pred_rf, alpha=0.5, edgecolors="k", linewidth=0.3, s=20)
ax.plot(
    [y_test_rf.min(), y_test_rf.max()],
    [y_test_rf.min(), y_test_rf.max()],
    "r--",
    lw=1.5,
    label="Perfect prediction",
)
ax.set_xlabel("Actual Close")
ax.set_ylabel("Predicted Close")
ax.set_title("Random Forest: Actual vs Predicted Close")
ax.legend()
plt.tight_layout()
plt.savefig("rf_scatter.png", dpi=150)
plt.close()


# ---------- Plot 2: Gradient Boosting Line (Actual vs Predicted Adj Close) ----------

fig, ax = plt.subplots(figsize=(10, 5))
ax.plot(test_gb["Date"].values, y_test_gb.values, label="Actual Adj Close", linewidth=1)
ax.plot(test_gb["Date"].values, y_pred_gb, label="Predicted Adj Close", linewidth=1, linestyle="--")
ax.set_xlabel("Date")
ax.set_ylabel("Adj Close")
ax.set_title("Gradient Boosting: Actual vs Predicted Adj Close")
ax.legend()
plt.tight_layout()
plt.savefig("gb_line.png", dpi=150)
plt.close()


# ---------- Plot 3: DBSCAN Box Plot (Cluster vs Standardized Volume) ----------

df_db_plot = df_db[df_db["cluster"] != -1]
clusters_unique = sorted(df_db_plot["cluster"].unique())
box_data = [df_db_plot[df_db_plot["cluster"] == c]["std_volume"].values for c in clusters_unique]

fig, ax = plt.subplots(figsize=(8, 6))
ax.boxplot(box_data, tick_labels=[str(c) for c in clusters_unique])
ax.set_xlabel("Cluster Label")
ax.set_ylabel("Standardized Volume")
ax.set_title("DBSCAN: Standardized Volume by Cluster")
plt.tight_layout()
plt.savefig("dbscan_box.png", dpi=150)
plt.close()


# ---------- Plot 4: Exponential Smoothing Line (Historical + Forecast) ----------

fig, ax = plt.subplots(figsize=(10, 5))
ax.plot(range(len(df_hw)), df_hw["Close/Last"].values, label="Historical Close/Last", linewidth=0.8)
forecast_idx = range(len(df_hw), len(df_hw) + 7)
ax.plot(forecast_idx, hw_forecast.values, label="7-step Forecast", linewidth=2, color="red", marker="o")
ax.set_xlabel("Time Index")
ax.set_ylabel("Close/Last")
ax.set_title("Holt-Winters: Historical Data and 7-step Forecast")
ax.legend()
plt.tight_layout()
plt.savefig("exp_smoothing_line.png", dpi=150)
plt.close()


# ---------- Plot 5: Daily Returns Histogram ----------

fig, ax = plt.subplots(figsize=(8, 6))
ax.hist(df_sr["daily_return"].values, bins=80, edgecolor="black", linewidth=0.3, alpha=0.75)
ax.set_xlabel("Daily Return (%)")
ax.set_ylabel("Frequency")
ax.set_title("Distribution of Daily Returns (tsla_2025)")
plt.tight_layout()
plt.savefig("daily_returns_hist.png", dpi=150)
plt.close()

print("\nAll plots saved successfully.")
