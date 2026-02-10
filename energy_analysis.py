# ==========================================
# Integrated Energy Analysis Script
# Requirements: pandas, numpy, matplotlib, scipy, scikit-learn, statsmodels
# Input files: State_Region_corrected.xlsx, long_data_.xlsx, dataset_tk.xlsx
# Output files: scatter_latitude_usage.png, roc_curve.png, violin_regional_usage.png,
#               contour_interaction.png, decomposition_tamil_nadu.png
# ==========================================

import pandas as pd
import numpy as np
import matplotlib.pyplot as plt
from sklearn.model_selection import train_test_split
from sklearn.linear_model import LinearRegression, LogisticRegression, HuberRegressor
from sklearn.svm import SVR
from sklearn.metrics import roc_auc_score, roc_curve, mean_squared_error
from scipy.stats import chi2_contingency, median_test
from statsmodels.tsa.seasonal import STL
from scipy.stats import norm
import warnings
warnings.filterwarnings('ignore')

# ---------- Helper Function for Runs Test ----------
def runs_test(binary_sequence):
    """Calculate runs test z-statistic"""
    n1 = np.sum(binary_sequence)
    n2 = len(binary_sequence) - n1

    # Count runs
    runs = 1
    for i in range(1, len(binary_sequence)):
        if binary_sequence[i] != binary_sequence[i-1]:
            runs += 1

    # Calculate expected runs and standard deviation
    n = n1 + n2
    runs_expected = ((2 * n1 * n2) / n) + 1
    runs_var = (2 * n1 * n2 * (2 * n1 * n2 - n)) / (n**2 * (n - 1))
    runs_std = np.sqrt(runs_var)

    # Calculate z-statistic
    z_stat = (runs - runs_expected) / runs_std if runs_std > 0 else 0
    return z_stat

# ---------- Load Datasets ----------
print("Loading datasets...")
state_region = pd.read_excel('State_Region_corrected.xlsx')
long_data = pd.read_excel('long_data_.xlsx')
dataset_tk = pd.read_excel('dataset_tk.xlsx')

# ==========================================
# Task 1: Multiple Linear Regression (MLR) on long_data_.xlsx
# Predicting Usage from latitude and longitude
# ==========================================
print("\n" + "="*60)
print("Task 1: Multiple Linear Regression (Latitude, Longitude -> Usage)")
print("="*60)

# ---------- Clean and Prepare Data ----------
df1 = long_data.copy()
df1 = df1.dropna(subset=['Usage', 'latitude', 'longitude'])

# ---------- Sort Data ----------
df1 = df1.sort_values(by=['Dates', 'States'], ascending=[True, True])

# ---------- Split Data ----------
X1 = df1[['latitude', 'longitude']]
y1 = df1['Usage']
X1_train, X1_test, y1_train, y1_test = train_test_split(X1, y1, train_size=0.7, random_state=42, shuffle=True)

# ---------- Fit OLS Model ----------
mlr1 = LinearRegression()
mlr1.fit(X1_train, y1_train)

# ---------- Calculate Adjusted R-squared ----------
n = X1_train.shape[0]
p = X1_train.shape[1]
r2 = mlr1.score(X1_train, y1_train)
adj_r2 = 1 - (1 - r2) * (n - 1) / (n - p - 1)
print(f"Training Set Adjusted R-squared: {adj_r2:.6f}")

# ---------- Generate Scatter Plot ----------
plt.figure(figsize=(8, 6))
plt.scatter(X1_train['latitude'], y1_train, alpha=0.5, s=10)
plt.xlabel('Latitude')
plt.ylabel('Usage')
plt.title('Regression Fit: Latitude vs Usage')
plt.tight_layout()
plt.savefig('scatter_latitude_usage.png', dpi=300)
plt.close()

# ==========================================
# Task 2: Logistic Regression for High vs Low Consumption
# ==========================================
print("\n" + "="*60)
print("Task 2: Logistic Regression (High vs Low Consumption)")
print("="*60)

# ---------- Clean and Prepare Data ----------
df2 = long_data.copy()
df2 = df2.dropna(subset=['Usage'])

# ---------- Create Binary Target ----------
median_usage = df2['Usage'].median()
df2['high_usage'] = (df2['Usage'] > median_usage).astype(int)

# ---------- Sort Data ----------
df2 = df2.sort_values(by='States', ascending=True)

# ---------- Split Data ----------
X2 = df2[['Usage']]
y2 = df2['high_usage']
X2_train, X2_test, y2_train, y2_test = train_test_split(X2, y2, train_size=0.65, random_state=42, shuffle=True)

# ---------- Fit Logistic Regression ----------
log_reg = LogisticRegression(penalty='l2', C=1.0, solver='lbfgs', random_state=42, max_iter=1000)
log_reg.fit(X2_train, y2_train)

# ---------- Predict and Calculate AUC ----------
y2_pred_proba = log_reg.predict_proba(X2_test)[:, 1]
auc_roc = roc_auc_score(y2_test, y2_pred_proba)
print(f"Testing Set AUC-ROC: {auc_roc:.6f}")

# ---------- Generate ROC Curve ----------
fpr, tpr, _ = roc_curve(y2_test, y2_pred_proba)
plt.figure(figsize=(8, 6))
plt.plot(fpr, tpr, linewidth=2, label=f'AUC = {auc_roc:.4f}')
plt.plot([0, 1], [0, 1], 'k--', linewidth=1)
plt.xlabel('False Positive Rate')
plt.ylabel('True Positive Rate')
plt.title('ROC Curve for Classifier Performance')
plt.legend()
plt.tight_layout()
plt.savefig('roc_curve.png', dpi=300)
plt.close()

# ==========================================
# Task 3: Support Vector Regression (Punjab from Haryana and Delhi)
# ==========================================
print("\n" + "="*60)
print("Task 3: Support Vector Regression (Punjab from Haryana, Delhi)")
print("="*60)

# ---------- Clean and Prepare Data ----------
df3 = dataset_tk.copy()
df3 = df3.dropna(subset=['Punjab', 'Haryana', 'Delhi'])

# ---------- Sort Data ----------
df3 = df3.sort_values(by='Unnamed: 0', ascending=True)

# ---------- Split Data (80-20 Temporal) ----------
split_idx = int(len(df3) * 0.8)
df3_train = df3.iloc[:split_idx]
df3_test = df3.iloc[split_idx:]

X3_train = df3_train[['Haryana', 'Delhi']]
y3_train = df3_train['Punjab']
X3_test = df3_test[['Haryana', 'Delhi']]
y3_test = df3_test['Punjab']

# ---------- Fit SVR ----------
svr = SVR(kernel='rbf', C=10.0, gamma='scale')
svr.fit(X3_train, y3_train)

# ---------- Predict and Calculate MAPE ----------
y3_pred = svr.predict(X3_test)
mape = np.mean(np.abs((y3_test - y3_pred) / y3_test)) * 100
print(f"Testing Set Mean Absolute Percentage Error: {mape:.6f}%")

# ==========================================
# Task 4: Mood Median Test (Regional Usage Comparison)
# ==========================================
print("\n" + "="*60)
print("Task 4: Mood Median Test (Regional Usage Comparison)")
print("="*60)

# ---------- Clean and Prepare Data ----------
df4 = long_data.copy()
df4 = df4.dropna(subset=['Usage', 'Regions'])

# ---------- Extract Regional Data ----------
nr_usage = df4[df4['Regions'] == 'NR']['Usage'].values
sr_usage = df4[df4['Regions'] == 'SR']['Usage'].values
er_usage = df4[df4['Regions'] == 'ER']['Usage'].values
wr_usage = df4[df4['Regions'] == 'WR']['Usage'].values

# ---------- Compute Mood Median Test ----------
stat, p_value, med, contingency_table = median_test(nr_usage, sr_usage, er_usage, wr_usage)
print(f"Chi-square Statistic: {stat:.6f}")

# ---------- Generate Violin Plot ----------
plt.figure(figsize=(10, 6))
regions_data = [nr_usage, sr_usage, er_usage, wr_usage]
positions = [1, 2, 3, 4]
parts = plt.violinplot(regions_data, positions=positions, showmeans=True, showmedians=True)
plt.xticks(positions, ['NR', 'SR', 'ER', 'WR'])
plt.xlabel('Regions')
plt.ylabel('Usage')
plt.title('Violin Plot: Regional Usage Distributions')
plt.tight_layout()
plt.savefig('violin_regional_usage.png', dpi=300)
plt.close()

# ==========================================
# Task 5: Runs Test for Randomness (Delhi Consumption)
# ==========================================
print("\n" + "="*60)
print("Task 5: Runs Test for Randomness (Delhi Consumption)")
print("="*60)

# ---------- Clean and Prepare Data ----------
df5 = dataset_tk.copy()
df5 = df5.dropna(subset=['Delhi'])

# ---------- Sort Data ----------
df5 = df5.sort_values(by='Unnamed: 0', ascending=True)

# ---------- Create Binary Sequence ----------
delhi_median = df5['Delhi'].median()
binary_seq = (df5['Delhi'] > delhi_median).astype(int)

# ---------- Calculate Runs Test ----------
z_stat = runs_test(binary_seq.values)
print(f"Z-score Statistic: {z_stat:.6f}")

# ==========================================
# Task 6: Sign Test (Maharashtra vs Gujarat)
# ==========================================
print("\n" + "="*60)
print("Task 6: Sign Test (Maharashtra vs Gujarat)")
print("="*60)

# ---------- Clean and Prepare Data ----------
df6 = dataset_tk.copy()
df6 = df6.dropna(subset=['Maharashtra', 'Gujarat'])

# ---------- Sort Data ----------
df6 = df6.sort_values(by='Unnamed: 0', ascending=True)

# ---------- Calculate Differences ----------
differences = df6['Maharashtra'] - df6['Gujarat']
M = (differences > 0).sum()
N = (differences < 0).sum()
test_statistic = min(M, N)
print(f"Sign Test Statistic (min of M, N): {test_statistic}")

# ==========================================
# Task 7: Partial Correlation (Punjab and Haryana controlling for Rajasthan)
# ==========================================
print("\n" + "="*60)
print("Task 7: Partial Correlation (Punjab-Haryana | Rajasthan)")
print("="*60)

# ---------- Clean and Prepare Data ----------
df7 = dataset_tk.copy()
df7 = df7.dropna(subset=['Punjab', 'Haryana', 'Rajasthan'])

# ---------- Fit Linear Regressions ----------
# Punjab ~ Rajasthan
lr_punjab = LinearRegression()
lr_punjab.fit(df7[['Rajasthan']], df7['Punjab'])
punjab_resid = df7['Punjab'] - lr_punjab.predict(df7[['Rajasthan']])

# Haryana ~ Rajasthan
lr_haryana = LinearRegression()
lr_haryana.fit(df7[['Rajasthan']], df7['Haryana'])
haryana_resid = df7['Haryana'] - lr_haryana.predict(df7[['Rajasthan']])

# ---------- Compute Partial Correlation ----------
partial_corr = np.corrcoef(punjab_resid, haryana_resid)[0, 1]
print(f"Partial Correlation Coefficient: {partial_corr:.6f}")

# ==========================================
# Task 8: MLR with Interaction Terms (Usage from Lat, Lon, Lat*Lon)
# ==========================================
print("\n" + "="*60)
print("Task 8: MLR with Interaction Terms")
print("="*60)

# ---------- Clean and Prepare Data ----------
df8 = long_data.copy()
df8 = df8.dropna(subset=['Usage', 'latitude', 'longitude'])

# ---------- Create Interaction Term ----------
df8['lat_lon_interaction'] = df8['latitude'] * df8['longitude']

# ---------- Sort Data ----------
df8 = df8.sort_values(by='Dates', ascending=True)

# ---------- Split Data (75-25 Temporal) ----------
split_idx8 = int(len(df8) * 0.75)
df8_train = df8.iloc[:split_idx8]
df8_test = df8.iloc[split_idx8:]

X8_train = df8_train[['latitude', 'longitude', 'lat_lon_interaction']]
y8_train = df8_train['Usage']
X8_test = df8_test[['latitude', 'longitude', 'lat_lon_interaction']]
y8_test = df8_test['Usage']

# ---------- Fit MLR with Interaction ----------
mlr8 = LinearRegression()
mlr8.fit(X8_train, y8_train)

interaction_coef = mlr8.coef_[2]
print(f"Interaction Term Coefficient: {interaction_coef:.6f}")

# ---------- Generate Contour Plot ----------
lat_range = np.linspace(df8['latitude'].min(), df8['latitude'].max(), 50)
lon_range = np.linspace(df8['longitude'].min(), df8['longitude'].max(), 50)
lat_grid, lon_grid = np.meshgrid(lat_range, lon_range)
interaction_grid = lat_grid * lon_grid

X_grid = np.c_[lat_grid.ravel(), lon_grid.ravel(), interaction_grid.ravel()]
usage_pred = mlr8.predict(X_grid).reshape(lat_grid.shape)

plt.figure(figsize=(10, 6))
contour = plt.contourf(lat_grid, lon_grid, usage_pred, levels=20, cmap='viridis')
plt.colorbar(contour, label='Predicted Usage')
plt.xlabel('Latitude')
plt.ylabel('Longitude')
plt.title('Contour Plot: Interaction Effect (Lat x Lon)')
plt.tight_layout()
plt.savefig('contour_interaction.png', dpi=300)
plt.close()

# ==========================================
# Task 9: Robust Regression using Huber Loss
# ==========================================
print("\n" + "="*60)
print("Task 9: Robust Regression using Huber Loss")
print("="*60)

# ---------- Clean and Prepare Data ----------
df9 = long_data.copy()
df9 = df9.dropna(subset=['Usage', 'latitude'])

# ---------- Sort Data ----------
df9 = df9.sort_values(by='States', ascending=True)

X9 = df9[['latitude']]
y9 = df9['Usage']

# ---------- Fit Huber Regressor ----------
huber = HuberRegressor(epsilon=1.35, alpha=0.001, max_iter=100)
huber.fit(X9, y9)
y9_pred_huber = huber.predict(X9)
mse_huber = mean_squared_error(y9, y9_pred_huber)

# ---------- Fit OLS for Comparison ----------
ols9 = LinearRegression()
ols9.fit(X9, y9)
y9_pred_ols = ols9.predict(X9)
mse_ols = mean_squared_error(y9, y9_pred_ols)

# ---------- Compute Difference ----------
mse_diff = mse_ols - mse_huber
print(f"MSE Difference (OLS MSE - Huber MSE): {mse_diff:.6f}")

# ==========================================
# Task 10: Time Series Decomposition (STL for Tamil Nadu)
# ==========================================
print("\n" + "="*60)
print("Task 10: Time Series Decomposition (STL for Tamil Nadu)")
print("="*60)

# ---------- Clean and Prepare Data ----------
df10 = dataset_tk.copy()
df10 = df10.dropna(subset=['Tamil Nadu'])

# ---------- Sort Data ----------
df10 = df10.sort_values(by='Unnamed: 0', ascending=True)

# ---------- Apply STL Decomposition ----------
ts_tn = df10['Tamil Nadu'].values
stl = STL(ts_tn, period=7)
result = stl.fit()

trend = result.trend
seasonal = result.seasonal

# ---------- Calculate Seasonality Strength ----------
seasonal_range = seasonal.max() - seasonal.min()
trend_range = trend.max() - trend.min()
seasonality_strength = seasonal_range / trend_range if trend_range != 0 else 0
print(f"Seasonality Strength Ratio: {seasonality_strength:.6f}")

# ---------- Generate Decomposition Plot ----------
fig, axes = plt.subplots(4, 1, figsize=(12, 10))

axes[0].plot(ts_tn)
axes[0].set_ylabel('Original')
axes[0].set_title('Time Series Decomposition: Tamil Nadu')

axes[1].plot(trend)
axes[1].set_ylabel('Trend')

axes[2].plot(seasonal)
axes[2].set_ylabel('Seasonal')

axes[3].plot(result.resid)
axes[3].set_ylabel('Residual')
axes[3].set_xlabel('Time')

plt.tight_layout()
plt.savefig('decomposition_tamil_nadu.png', dpi=300)
plt.close()

# ==========================================
# Task 11: Coefficient of Determination (Eta-squared)
# ==========================================
print("\n" + "="*60)
print("Task 11: Coefficient of Determination (Eta-squared)")
print("="*60)

# ---------- Compute Mean Usage per State ----------
mean_usage_by_state = long_data.groupby('States')['Usage'].mean().reset_index()
mean_usage_by_state.columns = ['State', 'Mean_Usage']

# ---------- Merge with State_Region ----------
# Clean state names for matching
state_region['State_Clean'] = state_region['State / Union territory (UT)'].str.strip()
mean_usage_by_state['State_Clean'] = mean_usage_by_state['State'].str.strip()

merged = pd.merge(mean_usage_by_state, state_region[['State_Clean', 'Region']],
                  left_on='State_Clean', right_on='State_Clean', how='inner')

# ---------- Remove Missing Values ----------
merged = merged.dropna(subset=['Mean_Usage', 'Region'])

# ---------- One-way ANOVA Calculation ----------
# Group by Region
groups = merged.groupby('Region')['Mean_Usage'].apply(list)

# Calculate grand mean
grand_mean = merged['Mean_Usage'].mean()

# Calculate between-group SS
ss_between = sum([len(group) * (np.mean(group) - grand_mean)**2 for group in groups])

# Calculate total SS
ss_total = sum([(x - grand_mean)**2 for x in merged['Mean_Usage']])

# Calculate eta-squared
eta_squared = ss_between / ss_total if ss_total != 0 else 0
print(f"Eta-squared Effect Size: {eta_squared:.6f}")

print("\n" + "="*60)
print("All analyses completed successfully!")
print("="*60)
