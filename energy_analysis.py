# ==========================================
# Integrated Energy Consumption Analysis Script
# Requirements: pandas, numpy, matplotlib, scipy, scikit-learn, statsmodels, openpyxl
# Input files: State_Region_corrected.xlsx, long_data_.xlsx, dataset_tk.xlsx
# Output files: scatter_lat_usage.png, roc_curve.png, svr_predictions.png,
#               regional_violin.png, interaction_contour.png,
#               tamilnadu_decomposition.png
# ==========================================

import pandas as pd
import numpy as np
import matplotlib.pyplot as plt
from sklearn.linear_model import LinearRegression, LogisticRegression, HuberRegressor
from sklearn.svm import SVR
from sklearn.model_selection import train_test_split
from sklearn.metrics import roc_auc_score, roc_curve, mean_absolute_percentage_error, mean_squared_error
from scipy import stats
from scipy.stats import chi2_contingency
from statsmodels.tsa.seasonal import STL
import warnings
warnings.filterwarnings('ignore')

# ---------- Load Datasets ----------
print("Loading datasets...")
state_region_df = pd.read_excel('State_Region_corrected.xlsx')
long_data_df = pd.read_excel('long_data_.xlsx')
dataset_tk_df = pd.read_excel('dataset_tk.xlsx')

# ==========================================
# ANALYSIS 1: Multiple Linear Regression (Usage from Coordinates)
# ==========================================
print("\n1. Multiple Linear Regression - Predicting Usage from Coordinates")

# Remove observations with missing Usage, latitude, or longitude
mlr1_df = long_data_df.dropna(subset=['Usage', 'latitude', 'longitude'])

# Sort by Dates then States
mlr1_df = mlr1_df.sort_values(['Dates', 'States'])

# Split 70-30 with random seed 42 and shuffle enabled
X1 = mlr1_df[['latitude', 'longitude']]
y1 = mlr1_df['Usage']
X1_train, X1_test, y1_train, y1_test = train_test_split(X1, y1, train_size=0.70, random_state=42, shuffle=True)

# Fit OLS regression
mlr1_model = LinearRegression()
mlr1_model.fit(X1_train, y1_train)

# Calculate adjusted R-squared for training set
n1 = len(X1_train)
p1 = X1_train.shape[1]
r2_1 = mlr1_model.score(X1_train, y1_train)
adj_r2_1 = 1 - (1 - r2_1) * (n1 - 1) / (n1 - p1 - 1)

print(f"Training Set Adjusted R-squared: {adj_r2_1:.6f}")

# ==========================================
# ANALYSIS 2: Logistic Regression (High vs Low Usage Classification)
# ==========================================
print("\n2. Logistic Regression - Classifying High vs Low Usage")

# Remove observations with missing Usage
lr_df = long_data_df.dropna(subset=['Usage'])

# Compute median Usage
median_usage = lr_df['Usage'].median()

# Create binary target
lr_df['high_usage'] = (lr_df['Usage'] > median_usage).astype(int)

# Sort by States
lr_df = lr_df.sort_values('States')

# Split 65-35 with random seed 42 and shuffle enabled
X2 = lr_df[['Usage']]
y2 = lr_df['high_usage']
X2_train, X2_test, y2_train, y2_test = train_test_split(X2, y2, train_size=0.65, random_state=42, shuffle=True)

# Fit logistic regression
lr_model = LogisticRegression(penalty='l2', C=1.0, solver='lbfgs', random_state=42, max_iter=1000)
lr_model.fit(X2_train, y2_train)

# Predict probabilities and calculate ROC AUC on test set
y2_pred_proba = lr_model.predict_proba(X2_test)[:, 1]
roc_auc_2 = roc_auc_score(y2_test, y2_pred_proba)

print(f"Testing Set Area Under ROC Curve: {roc_auc_2:.6f}")

# ==========================================
# ANALYSIS 3: Support Vector Regression (Punjab Prediction)
# ==========================================
print("\n3. Support Vector Regression - Predicting Punjab Usage")

# Remove observations with missing Punjab, Haryana, or Delhi
svr_df = dataset_tk_df.dropna(subset=['Punjab', 'Haryana', 'Delhi'])

# Sort by Unnamed: 0 chronologically
svr_df = svr_df.sort_values('Unnamed: 0')

# Split 80-20 temporal without shuffling
split_idx = int(len(svr_df) * 0.80)
svr_train = svr_df.iloc[:split_idx]
svr_test = svr_df.iloc[split_idx:]

X3_train = svr_train[['Haryana', 'Delhi']]
y3_train = svr_train['Punjab']
X3_test = svr_test[['Haryana', 'Delhi']]
y3_test = svr_test['Punjab']

# Fit SVR with RBF kernel
svr_model = SVR(kernel='rbf', C=10.0, gamma='scale')
svr_model.fit(X3_train, y3_train)

# Predict and calculate MAPE on test set
y3_pred = svr_model.predict(X3_test)
mape_3 = mean_absolute_percentage_error(y3_test, y3_pred)

print(f"Testing Set Mean Absolute Percentage Error: {mape_3:.6f}")

# ==========================================
# ANALYSIS 4: Mood Median Test (Regional Usage Comparison)
# ==========================================
print("\n4. Mood Median Test - Comparing Regional Usage Distributions")

# Remove observations with missing Usage or Regions
mood_df = long_data_df.dropna(subset=['Usage', 'Regions'])

# Extract regional data
nr_usage = mood_df[mood_df['Regions'] == 'NR']['Usage'].values
sr_usage = mood_df[mood_df['Regions'] == 'SR']['Usage'].values
er_usage = mood_df[mood_df['Regions'] == 'ER']['Usage'].values
wr_usage = mood_df[mood_df['Regions'] == 'WR']['Usage'].values

# Compute overall median
all_usage = np.concatenate([nr_usage, sr_usage, er_usage, wr_usage])
overall_median = np.median(all_usage)

# Create contingency table
above_nr = np.sum(nr_usage > overall_median)
below_nr = np.sum(nr_usage <= overall_median)
above_sr = np.sum(sr_usage > overall_median)
below_sr = np.sum(sr_usage <= overall_median)
above_er = np.sum(er_usage > overall_median)
below_er = np.sum(er_usage <= overall_median)
above_wr = np.sum(wr_usage > overall_median)
below_wr = np.sum(wr_usage <= overall_median)

contingency_table = np.array([
    [above_nr, below_nr],
    [above_sr, below_sr],
    [above_er, below_er],
    [above_wr, below_wr]
])

# Calculate chi-square statistic
chi2_stat, _, _, _ = chi2_contingency(contingency_table)

print(f"Chi-square Statistic: {chi2_stat:.6f}")

# ==========================================
# ANALYSIS 5: Runs Test (Delhi Consumption Randomness)
# ==========================================
print("\n5. Runs Test - Delhi Consumption Sequence Randomness")

# Remove observations with missing Delhi
runs_df = dataset_tk_df.dropna(subset=['Delhi'])

# Sort by Unnamed: 0 chronologically
runs_df = runs_df.sort_values('Unnamed: 0')

# Compute median Delhi value
delhi_median = runs_df['Delhi'].median()

# Create binary sequence
binary_seq = (runs_df['Delhi'] > delhi_median).astype(int).values

# Calculate runs
runs = 1
for i in range(1, len(binary_seq)):
    if binary_seq[i] != binary_seq[i-1]:
        runs += 1

# Calculate expected runs and standard deviation
n1 = np.sum(binary_seq == 1)
n0 = np.sum(binary_seq == 0)
n_total = n1 + n0

expected_runs = (2 * n1 * n0) / n_total + 1
variance_runs = (2 * n1 * n0 * (2 * n1 * n0 - n_total)) / (n_total**2 * (n_total - 1))
std_runs = np.sqrt(variance_runs)

# Calculate z-score
z_score_5 = (runs - expected_runs) / std_runs

print(f"Z-score Statistic: {z_score_5:.6f}")

# ==========================================
# ANALYSIS 6: Sign Test (Maharashtra vs Gujarat)
# ==========================================
print("\n6. Sign Test - Maharashtra vs Gujarat Comparison")

# Remove observations with missing Maharashtra or Gujarat
sign_df = dataset_tk_df.dropna(subset=['Maharashtra', 'Gujarat'])

# Sort by Unnamed: 0 chronologically
sign_df = sign_df.sort_values('Unnamed: 0')

# Calculate differences
differences = sign_df['Maharashtra'] - sign_df['Gujarat']

# Count positive and negative differences (exclude zeros)
M = np.sum(differences > 0)
N = np.sum(differences < 0)

# Report smaller of M or N
sign_test_stat = min(M, N)

print(f"Sign Test Statistic: {sign_test_stat}")

# ==========================================
# ANALYSIS 7: Partial Correlation (Punjab-Haryana controlling Rajasthan)
# ==========================================
print("\n7. Partial Correlation - Punjab-Haryana Controlling for Rajasthan")

# Remove observations with missing Punjab, Haryana, or Rajasthan
partial_df = dataset_tk_df.dropna(subset=['Punjab', 'Haryana', 'Rajasthan'])

# Fit regression Punjab ~ Rajasthan
X7a = partial_df[['Rajasthan']]
y7a = partial_df['Punjab']
reg_punjab = LinearRegression()
reg_punjab.fit(X7a, y7a)
punjab_resid = y7a - reg_punjab.predict(X7a)

# Fit regression Haryana ~ Rajasthan
y7b = partial_df['Haryana']
reg_haryana = LinearRegression()
reg_haryana.fit(X7a, y7b)
haryana_resid = y7b - reg_haryana.predict(X7a)

# Compute Pearson correlation between residuals
partial_corr = np.corrcoef(punjab_resid, haryana_resid)[0, 1]

print(f"Partial Correlation Coefficient: {partial_corr:.6f}")

# ==========================================
# ANALYSIS 8: Multiple Linear Regression with Interaction Terms
# ==========================================
print("\n8. Multiple Linear Regression with Interaction Terms")

# Remove observations with missing Usage, latitude, or longitude
mlr2_df = long_data_df.dropna(subset=['Usage', 'latitude', 'longitude'])

# Create interaction term
mlr2_df['lat_lon_interaction'] = mlr2_df['latitude'] * mlr2_df['longitude']

# Sort by Dates chronologically
mlr2_df = mlr2_df.sort_values('Dates')

# Split 75-25 temporal without shuffling
split_idx2 = int(len(mlr2_df) * 0.75)
mlr2_train = mlr2_df.iloc[:split_idx2]
mlr2_test = mlr2_df.iloc[split_idx2:]

X8_train = mlr2_train[['latitude', 'longitude', 'lat_lon_interaction']]
y8_train = mlr2_train['Usage']

# Fit multiple linear regression
mlr2_model = LinearRegression()
mlr2_model.fit(X8_train, y8_train)

# Get interaction term coefficient
interaction_coef = mlr2_model.coef_[2]

print(f"Interaction Term Coefficient: {interaction_coef:.6f}")

# ==========================================
# ANALYSIS 9: Robust Regression (Huber Loss)
# ==========================================
print("\n9. Robust Regression - Huber vs OLS Comparison")

# Remove observations with missing Usage or latitude
robust_df = long_data_df.dropna(subset=['Usage', 'latitude'])

# Sort by States alphabetically
robust_df = robust_df.sort_values('States')

X9 = robust_df[['latitude']]
y9 = robust_df['Usage']

# Fit Huber regressor
huber_model = HuberRegressor(epsilon=1.35, alpha=0.001, max_iter=100)
huber_model.fit(X9, y9)
y9_pred_huber = huber_model.predict(X9)
mse_huber = mean_squared_error(y9, y9_pred_huber)

# Fit OLS on same data
ols_model = LinearRegression()
ols_model.fit(X9, y9)
y9_pred_ols = ols_model.predict(X9)
mse_ols = mean_squared_error(y9, y9_pred_ols)

# Calculate difference
mse_diff = mse_ols - mse_huber

print(f"MSE Difference (OLS MSE - Huber MSE): {mse_diff:.6f}")

# ==========================================
# ANALYSIS 10: Time Series Decomposition (STL for Tamil Nadu)
# ==========================================
print("\n10. Time Series Decomposition - Tamil Nadu STL")

# Remove observations with missing Tamil Nadu
stl_df = dataset_tk_df.dropna(subset=['Tamil Nadu'])

# Sort by Unnamed: 0 chronologically
stl_df = stl_df.sort_values('Unnamed: 0')

# Apply STL decomposition with period 7 (weekly seasonality)
stl = STL(stl_df['Tamil Nadu'], period=7, seasonal=13)
result = stl.fit()

# Extract components
trend = result.trend
seasonal = result.seasonal

# Calculate seasonality strength
seasonal_range = seasonal.max() - seasonal.min()
trend_range = trend.max() - trend.min()
seasonality_strength = seasonal_range / trend_range if trend_range != 0 else 0

print(f"Seasonality Strength Ratio: {seasonality_strength:.6f}")

# ==========================================
# ANALYSIS 11: Eta-squared Effect Size (Region on Usage)
# ==========================================
print("\n11. Eta-squared - Regional Effect on Usage")

# Compute mean Usage for each state from long_data
state_means = long_data_df.groupby('States')['Usage'].mean().reset_index()
state_means.columns = ['State', 'Mean_Usage']

# Merge with state_region data
# Need to match state names carefully
state_region_df['State'] = state_region_df['State / Union territory (UT)'].str.strip()
merged_df = state_means.merge(state_region_df[['State', 'Region']], on='State', how='inner')

# Remove missing values
merged_df = merged_df.dropna()

# Fit one-way ANOVA
groups = [group['Mean_Usage'].values for name, group in merged_df.groupby('Region')]
f_stat, p_val = stats.f_oneway(*groups)

# Calculate eta-squared
overall_mean = merged_df['Mean_Usage'].mean()
ss_between = sum([len(group) * (group['Mean_Usage'].mean() - overall_mean)**2
                   for name, group in merged_df.groupby('Region')])
ss_total = sum((merged_df['Mean_Usage'] - overall_mean)**2)
eta_squared = ss_between / ss_total if ss_total != 0 else 0

print(f"Eta-squared Effect Size: {eta_squared:.6f}")

# ==========================================
# PLOT 1: Scatter Plot - Latitude vs Usage (Regression Fit)
# ==========================================
print("\n12. Generating Plots...")

plt.figure(figsize=(8, 6))
plt.scatter(X1_train['latitude'], y1_train, alpha=0.5, s=10, label='Training Data')
# Create prediction line
lat_range = np.linspace(X1_train['latitude'].min(), X1_train['latitude'].max(), 100)
lon_mean = X1_train['longitude'].mean()
X_pred = np.column_stack([lat_range, np.full(100, lon_mean)])
y_pred_line = mlr1_model.predict(X_pred)
plt.plot(lat_range, y_pred_line, 'r-', linewidth=2, label='Regression Fit')
plt.xlabel('Latitude')
plt.ylabel('Usage')
plt.title('Regression Fit: Latitude vs Usage')
plt.legend()
plt.tight_layout()
plt.savefig('scatter_lat_usage.png', dpi=300)
plt.close()

# ==========================================
# PLOT 2: ROC Curve for Classifier Performance
# ==========================================
fpr, tpr, _ = roc_curve(y2_test, y2_pred_proba)

plt.figure(figsize=(8, 6))
plt.plot(fpr, tpr, linewidth=2, label=f'ROC Curve (AUC = {roc_auc_2:.3f})')
plt.plot([0, 1], [0, 1], 'k--', linewidth=1, label='Random Classifier')
plt.xlabel('False Positive Rate')
plt.ylabel('True Positive Rate')
plt.title('ROC Curve - High vs Low Usage Classification')
plt.legend()
plt.grid(alpha=0.3)
plt.tight_layout()
plt.savefig('roc_curve.png', dpi=300)
plt.close()

# ==========================================
# PLOT 3: Line Plot - SVR Predictions vs Actual
# ==========================================
plt.figure(figsize=(10, 6))
test_indices = range(len(y3_test))
plt.plot(test_indices, y3_test.values, 'b-', linewidth=2, label='Actual Punjab', alpha=0.7)
plt.plot(test_indices, y3_pred, 'r--', linewidth=2, label='Predicted Punjab', alpha=0.7)
plt.xlabel('Observation Index')
plt.ylabel('Punjab Usage')
plt.title('SVR Predictions vs Actual Punjab Usage')
plt.legend()
plt.grid(alpha=0.3)
plt.tight_layout()
plt.savefig('svr_predictions.png', dpi=300)
plt.close()

# ==========================================
# PLOT 4: Violin Plot - Regional Usage Distributions
# ==========================================
fig, ax = plt.subplots(figsize=(10, 6))
regions_list = ['NR', 'SR', 'ER', 'WR']
data_list = [nr_usage, sr_usage, er_usage, wr_usage]

parts = ax.violinplot(data_list, positions=range(len(regions_list)), showmeans=True, showmedians=True)
ax.set_xticks(range(len(regions_list)))
ax.set_xticklabels(regions_list)
ax.set_xlabel('Regions')
ax.set_ylabel('Usage')
ax.set_title('Regional Usage Distributions')
ax.grid(alpha=0.3, axis='y')
plt.tight_layout()
plt.savefig('regional_violin.png', dpi=300)
plt.close()

# ==========================================
# PLOT 5: Contour Plot - Interaction Effect
# ==========================================
lat_grid = np.linspace(mlr2_df['latitude'].min(), mlr2_df['latitude'].max(), 50)
lon_grid = np.linspace(mlr2_df['longitude'].min(), mlr2_df['longitude'].max(), 50)
LAT, LON = np.meshgrid(lat_grid, lon_grid)
INTERACTION = LAT * LON

X_grid = np.column_stack([LAT.ravel(), LON.ravel(), INTERACTION.ravel()])
Z = mlr2_model.predict(X_grid).reshape(LAT.shape)

plt.figure(figsize=(10, 6))
contour = plt.contourf(LAT, LON, Z, levels=20, cmap='viridis')
plt.colorbar(contour, label='Predicted Usage')
plt.xlabel('Latitude')
plt.ylabel('Longitude')
plt.title('Interaction Effect: Latitude × Longitude on Usage')
plt.tight_layout()
plt.savefig('interaction_contour.png', dpi=300)
plt.close()

# ==========================================
# PLOT 6: Decomposition Plot - Tamil Nadu Time Series
# ==========================================
fig, axes = plt.subplots(4, 1, figsize=(12, 10))

axes[0].plot(stl_df['Unnamed: 0'].values, stl_df['Tamil Nadu'].values, linewidth=1)
axes[0].set_ylabel('Original')
axes[0].set_title('Tamil Nadu Usage Decomposition')
axes[0].grid(alpha=0.3)

axes[1].plot(stl_df['Unnamed: 0'].values, result.trend.values, linewidth=1, color='orange')
axes[1].set_ylabel('Trend')
axes[1].grid(alpha=0.3)

axes[2].plot(stl_df['Unnamed: 0'].values, result.seasonal.values, linewidth=1, color='green')
axes[2].set_ylabel('Seasonal')
axes[2].grid(alpha=0.3)

axes[3].plot(stl_df['Unnamed: 0'].values, result.resid.values, linewidth=1, color='red')
axes[3].set_ylabel('Residual')
axes[3].set_xlabel('Time')
axes[3].grid(alpha=0.3)

plt.tight_layout()
plt.savefig('tamilnadu_decomposition.png', dpi=300)
plt.close()

print("\n✓ All analyses completed successfully!")
print("✓ All plots generated successfully!")
