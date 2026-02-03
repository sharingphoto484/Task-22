# ==========================================
# Integrated Athletics Field Competition Analysis Script
# Requirements: pandas, numpy, matplotlib, scipy, scikit-learn, openpyxl
# Input files: Shot Put Men.xlsx, Javelin Throw Men.xlsx, Discus Throw Men.xlsx
# Output files: svr_scatter.png, gbr_residual.png, qda_confusion.png,
#               adaboost_scatter.png, lda_projection.png, lda_loadings.png,
#               knn_line.png, spearman_scatter.png, mannwhitney_box.png,
#               paired_ttest_hist.png
# ==========================================

import pandas as pd
import numpy as np
import matplotlib.pyplot as plt
from scipy import stats
from sklearn.model_selection import train_test_split, StratifiedShuffleSplit
from sklearn.svm import SVR
from sklearn.ensemble import GradientBoostingRegressor, AdaBoostRegressor
from sklearn.discriminant_analysis import QuadraticDiscriminantAnalysis, LinearDiscriminantAnalysis
from sklearn.neighbors import KNeighborsRegressor
from sklearn.preprocessing import StandardScaler
from sklearn.metrics import mean_squared_error, mean_absolute_error, r2_score, matthews_corrcoef, confusion_matrix
import warnings
warnings.filterwarnings('ignore')

# ---------- Load Datasets Robustly ----------
print("Loading datasets...")
shot_put_df = pd.read_excel('Shot Put Men.xlsx')
javelin_df = pd.read_excel('Javelin Throw Men.xlsx')
discus_df = pd.read_excel('Discus Throw Men.xlsx')

# Rename columns for consistency (handle space in "Result Score")
shot_put_df = shot_put_df.rename(columns={'Result Score': 'Result_Score'})
javelin_df = javelin_df.rename(columns={'Result Score': 'Result_Score'})
discus_df = discus_df.rename(columns={'Result Score': 'Result_Score'})

print("=" * 60)
print("ATHLETICS FIELD COMPETITION ANALYSIS - KEY OUTPUTS")
print("=" * 60)

# ==========================================
# PROCEDURE 1: Support Vector Regression (Polynomial Kernel) - Shot Put
# ==========================================
print("\n" + "-" * 60)
print("PROCEDURE 1: SVR Polynomial Kernel - Shot Put")
print("-" * 60)

# Load fresh dataset independently
df1 = pd.read_excel('Shot Put Men.xlsx').rename(columns={'Result Score': 'Result_Score'})

# Discard observations where Mark is missing OR Rank is missing OR Result_Score is missing
df1_filtered = df1.dropna(subset=['Mark', 'Rank', 'Result_Score'])

# Arrange by original row index (Unnamed: 0) in ascending order
df1_filtered = df1_filtered.sort_values('Unnamed: 0').reset_index(drop=True)

# Create derived feature: Rank * Result_Score
df1_filtered['Rank_x_ResultScore'] = df1_filtered['Rank'] * df1_filtered['Result_Score']

# Designate predictors and target
X1 = df1_filtered[['Rank', 'Result_Score', 'Rank_x_ResultScore']].values
y1 = df1_filtered['Mark'].values

# Allocate 70% train, 30% test with random_state 42
X1_train, X1_test, y1_train, y1_test = train_test_split(X1, y1, test_size=0.30, random_state=42)

# Configure SVR with polynomial kernel
svr_model = SVR(kernel='poly', degree=3, C=1.0, epsilon=0.1, gamma='scale', coef0=1.0)
svr_model.fit(X1_train, y1_train)

# Predict and calculate MSE
y1_pred = svr_model.predict(X1_test)
mse_svr = mean_squared_error(y1_test, y1_pred)
print(f"mean_squared_error: {round(mse_svr, 3)}")

# Generate scatter plot
plt.figure(figsize=(8, 6))
plt.scatter(y1_test, y1_pred, alpha=0.7, edgecolors='k')
plt.plot([y1_test.min(), y1_test.max()], [y1_test.min(), y1_test.max()], 'r--', lw=2)
plt.xlabel('Mark Actual Values')
plt.ylabel('Mark Predicted Values')
plt.title('SVR Polynomial Kernel - Shot Put Performance')
plt.tight_layout()
plt.savefig('svr_scatter.png', dpi=150)
plt.close()

# ==========================================
# PROCEDURE 2: Gradient Boosting Regression - Javelin Throw
# ==========================================
print("\n" + "-" * 60)
print("PROCEDURE 2: Gradient Boosting Regression - Javelin Throw")
print("-" * 60)

# Load fresh dataset independently
df2 = pd.read_excel('Javelin Throw Men.xlsx').rename(columns={'Result Score': 'Result_Score'})

# Discard observations where Mark is missing OR Rank is missing OR Result_Score is missing
df2_filtered = df2.dropna(subset=['Mark', 'Rank', 'Result_Score'])

# Arrange by original row index in ascending order
df2_filtered = df2_filtered.sort_values('Unnamed: 0').reset_index(drop=True)

# Designate predictors and target
X2 = df2_filtered[['Rank', 'Result_Score']].values
y2 = df2_filtered['Mark'].values

# Allocate 65% train, 35% test with random_state 42
X2_train, X2_test, y2_train, y2_test = train_test_split(X2, y2, test_size=0.35, random_state=42)

# Configure Gradient Boosting Regressor
gbr_model = GradientBoostingRegressor(n_estimators=100, max_depth=4, learning_rate=0.1,
                                       subsample=0.8, random_state=42)
gbr_model.fit(X2_train, y2_train)

# Predict and calculate RMSE
y2_pred = gbr_model.predict(X2_test)
rmse_gbr = np.sqrt(mean_squared_error(y2_test, y2_pred))
print(f"root_mean_squared_error: {round(rmse_gbr, 4)}")

# Generate residual plot
residuals = y2_test - y2_pred
plt.figure(figsize=(8, 6))
plt.scatter(y2_pred, residuals, alpha=0.7, edgecolors='k')
plt.axhline(y=0, color='r', linestyle='--', lw=2)
plt.xlabel('Predicted Mark Values')
plt.ylabel('Residual Errors')
plt.title('Gradient Boosting Regression - Residual Plot (Javelin)')
plt.tight_layout()
plt.savefig('gbr_residual.png', dpi=150)
plt.close()

# ==========================================
# PROCEDURE 3: Quadratic Discriminant Analysis - Discus Throw
# ==========================================
print("\n" + "-" * 60)
print("PROCEDURE 3: Quadratic Discriminant Analysis - Discus Throw")
print("-" * 60)

# Load fresh dataset independently
df3 = pd.read_excel('Discus Throw Men.xlsx').rename(columns={'Result Score': 'Result_Score'})

# Discard observations where Result_Score is missing OR Mark is missing OR Rank is missing
df3_filtered = df3.dropna(subset=['Result_Score', 'Mark', 'Rank'])

# Arrange by original row index in ascending order
df3_filtered = df3_filtered.sort_values('Unnamed: 0').reset_index(drop=True)

# Create binary target using MEDIAN threshold for deterministic class split
# Median provides balanced classes from actual data range (1244-1320)
median_threshold = df3_filtered['Result_Score'].median()
df3_filtered['Elite_Status'] = (df3_filtered['Result_Score'] >= median_threshold).astype(int)
print(f"Binary threshold (median): {median_threshold}")

# Designate discriminating features
X3 = df3_filtered[['Mark', 'Rank']].values
y3 = df3_filtered['Elite_Status'].values

# Stratified split: 75% train, 25% test using StratifiedShuffleSplit
sss = StratifiedShuffleSplit(n_splits=1, test_size=0.25, random_state=42)
for train_idx, test_idx in sss.split(X3, y3):
    X3_train, X3_test = X3[train_idx], X3[test_idx]
    y3_train, y3_test = y3[train_idx], y3[test_idx]

# Configure QDA with reg_param 0.01
qda_model = QuadraticDiscriminantAnalysis(reg_param=0.01)
qda_model.fit(X3_train, y3_train)

# Predict and calculate Matthews Correlation Coefficient
y3_pred = qda_model.predict(X3_test)
mcc = matthews_corrcoef(y3_test, y3_pred)
print(f"matthews_corrcoef: {round(mcc, 4)}")

# Generate confusion matrix heatmap
cm = confusion_matrix(y3_test, y3_pred)
plt.figure(figsize=(6, 5))
plt.imshow(cm, interpolation='nearest', cmap=plt.cm.Blues)
plt.title('QDA Confusion Matrix - Discus Elite Classification')
plt.colorbar()
tick_marks = np.arange(2)
plt.xticks(tick_marks, ['Class 0', 'Class 1'])
plt.yticks(tick_marks, ['Class 0', 'Class 1'])
for i in range(2):
    for j in range(2):
        plt.text(j, i, str(cm[i, j]), ha='center', va='center', fontsize=14,
                color='white' if cm[i, j] > cm.max()/2 else 'black')
plt.xlabel('Predicted Class Labels')
plt.ylabel('Actual Class Labels')
plt.tight_layout()
plt.savefig('qda_confusion.png', dpi=150)
plt.close()

# ==========================================
# PROCEDURE 4: AdaBoost Regression - Combined Shot Put & Discus
# ==========================================
print("\n" + "-" * 60)
print("PROCEDURE 4: AdaBoost Regression - Combined Shot Put & Discus")
print("-" * 60)

# Load fresh datasets independently
df4a = pd.read_excel('Shot Put Men.xlsx').rename(columns={'Result Score': 'Result_Score'})
df4b = pd.read_excel('Discus Throw Men.xlsx').rename(columns={'Result Score': 'Result_Score'})

# Discard observations where Mark is missing OR Rank is missing from each dataset independently
df4a_filtered = df4a.dropna(subset=['Mark', 'Rank'])
df4b_filtered = df4b.dropna(subset=['Mark', 'Rank'])

# Vertically concatenate filtered datasets maintaining concatenation order
df4_combined = pd.concat([df4a_filtered[['Mark', 'Rank']], df4b_filtered[['Mark', 'Rank']]],
                          ignore_index=True)

# Designate predictors and target
X4 = df4_combined[['Rank']].values
y4 = df4_combined['Mark'].values

# Allocate 80% train, 20% test with random_state 42
X4_train, X4_test, y4_train, y4_test = train_test_split(X4, y4, test_size=0.20, random_state=42)

# Configure AdaBoost Regressor
ada_model = AdaBoostRegressor(n_estimators=50, learning_rate=1.0, loss='linear', random_state=42)
ada_model.fit(X4_train, y4_train)

# Predict and calculate MAE
y4_pred = ada_model.predict(X4_test)
mae_ada = mean_absolute_error(y4_test, y4_pred)
print(f"mean_absolute_error: {round(mae_ada, 3)}")

# Generate prediction scatter plot
plt.figure(figsize=(8, 6))
plt.scatter(y4_test, y4_pred, alpha=0.7, edgecolors='k')
plt.plot([y4_test.min(), y4_test.max()], [y4_test.min(), y4_test.max()], 'r--', lw=2)
plt.xlabel('Actual Mark Values')
plt.ylabel('Predicted Mark Values')
plt.title('AdaBoost Regression - Combined Shot Put & Discus')
plt.tight_layout()
plt.savefig('adaboost_scatter.png', dpi=150)
plt.close()

# ==========================================
# PROCEDURE 5: Linear Discriminant Analysis - Javelin Throw
# ==========================================
print("\n" + "-" * 60)
print("PROCEDURE 5: Linear Discriminant Analysis - Javelin Throw")
print("-" * 60)

# Load fresh dataset independently
df5 = pd.read_excel('Javelin Throw Men.xlsx').rename(columns={'Result Score': 'Result_Score'})

# Discard observations where Result_Score is missing OR Mark is missing OR Rank is missing
df5_filtered = df5.dropna(subset=['Result_Score', 'Mark', 'Rank'])

# Arrange by original row index in ascending order
df5_filtered = df5_filtered.sort_values('Unnamed: 0').reset_index(drop=True)

# Create three-class target using TERTILES for deterministic class split
# Tertiles (33rd and 67th percentiles) ensure balanced class distribution
tertile_33 = df5_filtered['Result_Score'].quantile(0.33)
tertile_67 = df5_filtered['Result_Score'].quantile(0.67)
print(f"Tertile boundaries: 33rd={tertile_33}, 67th={tertile_67}")

def bin_result_score(score):
    if score <= tertile_33:
        return 0
    elif score <= tertile_67:
        return 1
    else:
        return 2

df5_filtered['Performance_Tier'] = df5_filtered['Result_Score'].apply(bin_result_score)

# Create derived feature: Mark / Rank
df5_filtered['Mark_div_Rank'] = df5_filtered['Mark'] / df5_filtered['Rank']

# Designate projection features
X5 = df5_filtered[['Mark', 'Rank', 'Mark_div_Rank']].values
y5 = df5_filtered['Performance_Tier'].values

# Standardize features (zero mean, unit variance)
scaler = StandardScaler()
X5_scaled = scaler.fit_transform(X5)

# Configure LDA with n_components=2, solver=eigen for deterministic eigenvalue access
lda_model = LinearDiscriminantAnalysis(n_components=2, solver='eigen')
lda_model.fit(X5_scaled, y5)

# Transform to get discriminant scores
X5_lda = lda_model.transform(X5_scaled)

# Calculate proportion of variance explained by first discriminant component
# Using explicit eigenvalue ratio: eigvals_[0] / sum(eigvals_) for determinism
eigenvalues = lda_model.explained_variance_ratio_
variance_proportion = eigenvalues[0] * 100
print(f"variance_proportion: {round(variance_proportion, 2)}%")

# Generate scatter plot for discriminant projection
plt.figure(figsize=(8, 6))
scatter = plt.scatter(X5_lda[:, 0], X5_lda[:, 1], c=y5, cmap='viridis', alpha=0.7, edgecolors='k')
plt.colorbar(scatter, label='Performance Tier')
plt.xlabel('First Discriminant Component')
plt.ylabel('Second Discriminant Component')
plt.title('LDA Projection - Javelin Performance Tiers')
plt.tight_layout()
plt.savefig('lda_projection.png', dpi=150)
plt.close()

# Generate LDA feature loadings bar chart (scalings_ coefficients)
feature_names = ['Mark', 'Rank', 'Mark/Rank']
scalings = lda_model.scalings_
plt.figure(figsize=(10, 5))
x_pos = np.arange(len(feature_names))
width = 0.35
plt.bar(x_pos - width/2, np.abs(scalings[:, 0]), width, label='LD1', color='steelblue')
plt.bar(x_pos + width/2, np.abs(scalings[:, 1]), width, label='LD2', color='darkorange')
plt.xlabel('Features')
plt.ylabel('Absolute Coefficient Magnitude')
plt.title('LDA Feature Loadings (Scalings) - Javelin Performance')
plt.xticks(x_pos, feature_names)
plt.legend()
plt.tight_layout()
plt.savefig('lda_loadings.png', dpi=150)
plt.close()

# ==========================================
# PROCEDURE 6: K-Nearest Neighbors Regression - Discus Throw
# ==========================================
print("\n" + "-" * 60)
print("PROCEDURE 6: K-Nearest Neighbors Regression - Discus Throw")
print("-" * 60)

# Load fresh dataset independently
df6 = pd.read_excel('Discus Throw Men.xlsx').rename(columns={'Result Score': 'Result_Score'})

# Discard observations where Mark is missing OR Result_Score is missing
df6_filtered = df6.dropna(subset=['Mark', 'Result_Score'])

# Arrange by original row index in ascending order
df6_filtered = df6_filtered.sort_values('Unnamed: 0').reset_index(drop=True)

# Designate predictor and target
X6 = df6_filtered[['Result_Score']].values
y6 = df6_filtered['Mark'].values

# Allocate 70% train, 30% test with random_state 42
X6_train, X6_test, y6_train, y6_test = train_test_split(X6, y6, test_size=0.30, random_state=42)

# Configure KNN Regressor
knn_model = KNeighborsRegressor(n_neighbors=5, weights='distance', metric='euclidean')
knn_model.fit(X6_train, y6_train)

# Predict and calculate R-squared
y6_pred = knn_model.predict(X6_test)
r2_knn = r2_score(y6_test, y6_pred)
print(f"r_squared: {round(r2_knn, 4)}")

# Generate line plot with Result_Score sorted ascending
# Use all data for visualization
X6_sorted_idx = np.argsort(X6.flatten())
X6_sorted = X6[X6_sorted_idx]
y6_pred_all = knn_model.predict(X6_sorted)

plt.figure(figsize=(8, 6))
plt.plot(X6_sorted.flatten(), y6_pred_all, 'b-', lw=2, label='KNN Predictions')
plt.scatter(X6.flatten(), y6, alpha=0.5, edgecolors='k', label='Actual Data')
plt.xlabel('Result_Score Values (Sorted Ascending)')
plt.ylabel('Predicted Mark Values')
plt.title('KNN Regression - Discus Throw')
plt.legend()
plt.tight_layout()
plt.savefig('knn_line.png', dpi=150)
plt.close()

# ==========================================
# PROCEDURE 7: Spearman Rank Correlation - Shot Put
# ==========================================
print("\n" + "-" * 60)
print("PROCEDURE 7: Spearman Rank Correlation - Shot Put")
print("-" * 60)

# Load fresh dataset independently
df7 = pd.read_excel('Shot Put Men.xlsx').rename(columns={'Result Score': 'Result_Score'})

# Discard observations where Mark is missing OR Rank is missing
df7_filtered = df7.dropna(subset=['Mark', 'Rank'])

# Arrange by original row index in ascending order
df7_filtered = df7_filtered.sort_values('Unnamed: 0').reset_index(drop=True)

# Compute Spearman correlation
spearman_corr, p_value = stats.spearmanr(df7_filtered['Mark'], df7_filtered['Rank'])
print(f"spearman_correlation: {round(spearman_corr, 4)}")

# Generate scatter plot
plt.figure(figsize=(8, 6))
plt.scatter(df7_filtered['Rank'], df7_filtered['Mark'], alpha=0.7, edgecolors='k')
plt.xlabel('Rank Values')
plt.ylabel('Mark Values')
plt.title(f'Spearman Rank Correlation - Shot Put (r = {round(spearman_corr, 4)})')
plt.tight_layout()
plt.savefig('spearman_scatter.png', dpi=150)
plt.close()

# ==========================================
# PROCEDURE 8: Mann-Whitney U Test - Shot Put vs Javelin Mark
# ==========================================
print("\n" + "-" * 60)
print("PROCEDURE 8: Mann-Whitney U Test - Shot Put vs Javelin")
print("-" * 60)

# Load fresh datasets independently
df8a = pd.read_excel('Shot Put Men.xlsx').rename(columns={'Result Score': 'Result_Score'})
df8b = pd.read_excel('Javelin Throw Men.xlsx').rename(columns={'Result Score': 'Result_Score'})

# Extract Mark variable discarding missing observations
shot_put_mark = df8a.dropna(subset=['Mark'])['Mark'].values
javelin_mark = df8b.dropna(subset=['Mark'])['Mark'].values

# Compute Mann-Whitney U statistic
u_stat, p_value = stats.mannwhitneyu(shot_put_mark, javelin_mark, alternative='two-sided')
print(f"u_statistic: {round(u_stat, 2)}")

# Generate box plot
plt.figure(figsize=(8, 6))
plt.boxplot([shot_put_mark, javelin_mark], labels=['Shot Put', 'Javelin Throw'])
plt.xlabel('Event Type Labels')
plt.ylabel('Mark Values')
plt.title('Mark Distribution Comparison - Mann-Whitney U Test')
plt.tight_layout()
plt.savefig('mannwhitney_box.png', dpi=150)
plt.close()

# ==========================================
# PROCEDURE 9: Paired Samples t-Test - Discus Throw
# ==========================================
print("\n" + "-" * 60)
print("PROCEDURE 9: Paired Samples t-Test - Discus Throw")
print("-" * 60)

# Load fresh dataset independently
df9 = pd.read_excel('Discus Throw Men.xlsx').rename(columns={'Result Score': 'Result_Score'})

# Discard observations where Mark is missing OR Result_Score is missing
df9_filtered = df9.dropna(subset=['Mark', 'Result_Score'])

# Arrange by original row index in ascending order
df9_filtered = df9_filtered.sort_values('Unnamed: 0').reset_index(drop=True)

# Standardize both variables to z-scores independently
mark_zscore = stats.zscore(df9_filtered['Mark'])
result_score_zscore = stats.zscore(df9_filtered['Result_Score'])

# Compute paired differences
paired_diff = mark_zscore - result_score_zscore

# Execute paired t-test
t_stat, p_value = stats.ttest_rel(mark_zscore, result_score_zscore)
print(f"t_statistic: {round(t_stat, 3)}")

# Generate histogram for paired differences
plt.figure(figsize=(8, 6))
plt.hist(paired_diff, bins=15, edgecolor='black', alpha=0.7)
plt.axvline(x=0, color='r', linestyle='--', lw=2)
plt.xlabel('Difference Values (Standardized Mark - Standardized Result_Score)')
plt.ylabel('Frequency Counts')
plt.title('Paired Differences Distribution - Discus Throw')
plt.tight_layout()
plt.savefig('paired_ttest_hist.png', dpi=150)
plt.close()

# ==========================================
# PROCEDURE 10: Coefficient of Variation Ratio - Shot Put vs Javelin
# ==========================================
print("\n" + "-" * 60)
print("PROCEDURE 10: Coefficient of Variation Ratio")
print("-" * 60)

# Load fresh datasets independently
df10a = pd.read_excel('Shot Put Men.xlsx').rename(columns={'Result Score': 'Result_Score'})
df10b = pd.read_excel('Javelin Throw Men.xlsx').rename(columns={'Result Score': 'Result_Score'})

# Extract Mark discarding missing observations
shot_put_mark_cv = df10a.dropna(subset=['Mark'])['Mark']
javelin_mark_cv = df10b.dropna(subset=['Mark'])['Mark']

# Calculate CV for shot put
shot_put_cv = (shot_put_mark_cv.std() / shot_put_mark_cv.mean()) * 100

# Calculate CV for javelin
javelin_cv = (javelin_mark_cv.std() / javelin_mark_cv.mean()) * 100

# Compute ratio
cv_ratio = shot_put_cv / javelin_cv
print(f"CV_ratio: {round(cv_ratio, 3)}")

# ==========================================
# PROCEDURE 11: Gradient Boosting Explanation
# ==========================================
print("\n" + "-" * 60)
print("PROCEDURE 11: Gradient Boosting Explanation")
print("-" * 60)

# One sentence explanation
explanation = ("Gradient boosting achieves superior predictive accuracy compared to single estimators "
               "because it sequentially trains weak learners where each subsequent learner focuses on "
               "correcting the residual errors of the ensemble, thereby iteratively reducing bias and "
               "capturing complex nonlinear patterns that individual models cannot learn alone.")
print(f"Explanation: {explanation}")

print("\n" + "=" * 60)
print("ANALYSIS COMPLETE - All visualizations saved as PNG files")
print("=" * 60)
