# ==========================================
# Cricket Statistics Analysis Script
# Requirements: pandas, numpy, matplotlib, scipy, scikit-learn, openpyxl
# Input files: odb.xlsx, odbo.xlsx, twb.xlsx
# Output files: Multiple PNG plots for each analytical procedure
# ==========================================

import pandas as pd
import numpy as np
import matplotlib.pyplot as plt
from sklearn.model_selection import train_test_split
from sklearn.linear_model import LinearRegression, LogisticRegression
from sklearn.preprocessing import StandardScaler, PolynomialFeatures
from sklearn.cluster import AgglomerativeClustering
from sklearn.discriminant_analysis import LinearDiscriminantAnalysis
from sklearn.metrics import mean_squared_error, accuracy_score, silhouette_score
from scipy import stats
import warnings
warnings.filterwarnings('ignore')

# ---------- Open-ended Question Answer ----------
# Ward linkage hierarchical clustering guarantees deterministic cluster formation independent of random initialization because it uses a greedy bottom-up agglomerative approach that merges clusters based solely on minimizing within-cluster variance at each step, with no random starting points or stochastic elements involved in the algorithm.

# ---------- Procedure 1: Multiple Linear Regression with Polynomial Features ----------
print("="*60)
print("Procedure 1: Polynomial Linear Regression - ODI Batting")
print("="*60)

# Load original unfiltered dataset
odb_p1 = pd.read_excel('odb.xlsx')

# Remove records where Ave OR Mat OR Runs is missing
mask_p1 = odb_p1['Ave'].notna() & odb_p1['Mat'].notna() & odb_p1['Runs'].notna()
odb_p1_clean = odb_p1[mask_p1].copy()

# Designate variables
y_p1 = odb_p1_clean['Ave'].values
X_base_p1 = odb_p1_clean[['Mat', 'Runs']].values

# Generate polynomial features: Mat, Runs, Mat^2, Runs^2, Mat*Runs
poly = PolynomialFeatures(degree=2, include_bias=False)
X_poly_p1 = poly.fit_transform(X_base_p1)

# Partition: 65% training, 35% validation, random_state=42
X_train_p1, X_val_p1, y_train_p1, y_val_p1 = train_test_split(
    X_poly_p1, y_p1, test_size=0.35, random_state=42
)

# Fit linear regression using closed-form normal equation
# LinearRegression uses normal equation (X'X)^-1 X'y when fit_intercept=True
lr_p1 = LinearRegression()
lr_p1.fit(X_train_p1, y_train_p1)

# Predict on validation partition
y_pred_p1 = lr_p1.predict(X_val_p1)

# Calculate RMSE on validation partition
rmse_p1 = np.sqrt(mean_squared_error(y_val_p1, y_pred_p1))
print(f"Root Mean Squared Error: {rmse_p1:.3f}")

# Generate scatter plot: Actual vs Predicted Ave
plt.figure(figsize=(8, 6))
plt.scatter(y_val_p1, y_pred_p1, alpha=0.7, edgecolors='k', linewidth=0.5)
plt.plot([y_val_p1.min(), y_val_p1.max()], [y_val_p1.min(), y_val_p1.max()], 'r--', lw=2)
plt.xlabel('Actual Ave')
plt.ylabel('Predicted Ave')
plt.title('Procedure 1: Polynomial Regression - Actual vs Predicted Ave')
plt.tight_layout()
plt.savefig('p1_regression_scatter.png', dpi=150)
plt.close()
print("Plot saved: p1_regression_scatter.png\n")

# ---------- Procedure 2: Binary Logistic Regression ----------
print("="*60)
print("Procedure 2: Logistic Regression - ODI Bowling")
print("="*60)

# Load original unfiltered dataset
odbo_p2 = pd.read_excel('odbo.xlsx')

# Column 5 is the '5' column (5-wicket hauls) - using column name directly
# Create binary target: >=10 -> 1, <10 -> 0
odbo_p2['target'] = (odbo_p2['5'] >= 10).astype(int)

# Remove records where Mat OR Wkts OR SR is missing
mask_p2 = odbo_p2['Mat'].notna() & odbo_p2['Wkts'].notna() & odbo_p2['SR'].notna()
odbo_p2_clean = odbo_p2[mask_p2].copy()

# Designate predictor variables
X_p2 = odbo_p2_clean[['Mat', 'Wkts', 'SR']].values
y_p2 = odbo_p2_clean['target'].values

# Partition: 70% training, 30% testing, stratified, random_state=42
X_train_p2, X_test_p2, y_train_p2, y_test_p2 = train_test_split(
    X_p2, y_p2, test_size=0.30, random_state=42, stratify=y_p2
)

# Fit logistic regression with lbfgs solver, max_iter=1000
logreg_p2 = LogisticRegression(solver='lbfgs', max_iter=1000)
logreg_p2.fit(X_train_p2, y_train_p2)

# Predict on testing partition
y_pred_p2 = logreg_p2.predict(X_test_p2)

# Calculate accuracy
accuracy_p2 = accuracy_score(y_test_p2, y_pred_p2)
print(f"Classification Accuracy: {accuracy_p2:.4f}")

# Generate bar plot for predicted class outcomes
class_counts_p2 = pd.Series(y_pred_p2).value_counts().sort_index()
plt.figure(figsize=(8, 6))
plt.bar(['Class 0', 'Class 1'], [class_counts_p2.get(0, 0), class_counts_p2.get(1, 0)],
        color=['steelblue', 'coral'], edgecolor='black')
plt.xlabel('Class Labels')
plt.ylabel('Observation Counts')
plt.title('Procedure 2: Logistic Regression - Predicted Class Outcomes')
plt.tight_layout()
plt.savefig('p2_logistic_bar.png', dpi=150)
plt.close()
print("Plot saved: p2_logistic_bar.png\n")

# ---------- Procedure 3: Hierarchical Agglomerative Clustering ----------
print("="*60)
print("Procedure 3: Hierarchical Clustering - T20 Batting")
print("="*60)

# Load original unfiltered dataset
twb_p3 = pd.read_excel('twb.xlsx')

# Remove records where Runs OR Ave OR SR is missing
mask_p3 = twb_p3['Runs'].notna() & twb_p3['Ave'].notna() & twb_p3['SR'].notna()
twb_p3_clean = twb_p3[mask_p3].copy()

# Designate clustering attributes
X_p3 = twb_p3_clean[['Runs', 'Ave', 'SR']].values

# Standardize after missing value removal
scaler_p3 = StandardScaler()
X_p3_std = scaler_p3.fit_transform(X_p3)

# Apply AgglomerativeClustering with n_clusters=4, linkage='ward'
agg_clust = AgglomerativeClustering(n_clusters=4, linkage='ward')
cluster_labels_p3 = agg_clust.fit_predict(X_p3_std)

# Calculate silhouette coefficient on standardized features
silhouette_p3 = silhouette_score(X_p3_std, cluster_labels_p3)
print(f"Silhouette Coefficient: {silhouette_p3:.4f}")

# Generate scatter plot: Runs vs Ave colored by cluster
plt.figure(figsize=(8, 6))
scatter = plt.scatter(twb_p3_clean['Runs'].values, twb_p3_clean['Ave'].values,
                      c=cluster_labels_p3, cmap='viridis', alpha=0.7, edgecolors='k', linewidth=0.5)
plt.colorbar(scatter, label='Cluster')
plt.xlabel('Runs')
plt.ylabel('Ave')
plt.title('Procedure 3: Hierarchical Clustering - T20 Batting Profiles')
plt.tight_layout()
plt.savefig('p3_clustering_scatter.png', dpi=150)
plt.close()
print("Plot saved: p3_clustering_scatter.png\n")

# ---------- Procedure 4: Linear Discriminant Analysis ----------
print("="*60)
print("Procedure 4: Linear Discriminant Analysis - ODI Batting")
print("="*60)

# Load original unfiltered dataset
odb_p4 = pd.read_excel('odb.xlsx')

# Create categorical target by binning column '100' into three classes
# 0-5 -> class 0, 6-15 -> class 1, 16+ -> class 2
def bin_centuries(val):
    if pd.isna(val):
        return np.nan
    elif val <= 5:
        return 0
    elif val <= 15:
        return 1
    else:
        return 2

odb_p4['century_class'] = odb_p4['100'].apply(bin_centuries)

# Remove records where Runs OR BF OR SR is missing
mask_p4 = odb_p4['Runs'].notna() & odb_p4['BF'].notna() & odb_p4['SR'].notna()
odb_p4_clean = odb_p4[mask_p4].copy()

# Designate discriminant predictors
X_p4 = odb_p4_clean[['Runs', 'BF', 'SR']].values
y_p4 = odb_p4_clean['century_class'].values

# Standardize predictors after missing value removal
scaler_p4 = StandardScaler()
X_p4_std = scaler_p4.fit_transform(X_p4)

# Fit LDA with solver='svd' and n_components=2
lda_p4 = LinearDiscriminantAnalysis(solver='svd', n_components=2)
X_lda_p4 = lda_p4.fit_transform(X_p4_std, y_p4)

# Calculate proportion of between-class variance explained by first component
variance_explained_p4 = lda_p4.explained_variance_ratio_[0] * 100
print(f"Variance Explained by First Component: {variance_explained_p4:.2f}%")

# Generate scatter plot: LD1 vs LD2
plt.figure(figsize=(8, 6))
scatter = plt.scatter(X_lda_p4[:, 0], X_lda_p4[:, 1], c=y_p4, cmap='viridis',
                      alpha=0.7, edgecolors='k', linewidth=0.5)
plt.colorbar(scatter, label='Century Class')
plt.xlabel('First Discriminant Component')
plt.ylabel('Second Discriminant Component')
plt.title('Procedure 4: LDA - Discriminant Space')
plt.tight_layout()
plt.savefig('p4_lda_scatter.png', dpi=150)
plt.close()
print("Plot saved: p4_lda_scatter.png\n")

# ---------- Procedure 5: One-way ANOVA ----------
print("="*60)
print("Procedure 5: One-way ANOVA - T20 Batting")
print("="*60)

# Load original unfiltered dataset
twb_p5 = pd.read_excel('twb.xlsx')

# Create categorical grouping by binning column '50' into three categories
# 0-15 -> 'Low', 16-30 -> 'Medium', 31+ -> 'High'
def bin_fifties(val):
    if pd.isna(val):
        return np.nan
    elif val <= 15:
        return 'Low (0-15)'
    elif val <= 30:
        return 'Medium (16-30)'
    else:
        return 'High (31+)'

twb_p5['fifty_category'] = twb_p5['50'].apply(bin_fifties)

# Remove records where SR OR column 50 is missing
mask_p5 = twb_p5['SR'].notna() & twb_p5['50'].notna()
twb_p5_clean = twb_p5[mask_p5].copy()

# Designate dependent variable
sr_values_p5 = twb_p5_clean['SR'].values
categories_p5 = twb_p5_clean['fifty_category'].values

# Group data by category - only include non-empty groups
groups_p5 = {}
all_categories = ['Low (0-15)', 'Medium (16-30)', 'High (31+)']
non_empty_cats = []
for cat in all_categories:
    group_data = twb_p5_clean[twb_p5_clean['fifty_category'] == cat]['SR'].values
    if len(group_data) > 0:
        groups_p5[cat] = group_data
        non_empty_cats.append(cat)

# Calculate F-statistic using scipy one-way ANOVA (only non-empty groups)
group_arrays = [groups_p5[cat] for cat in non_empty_cats]
f_stat_p5, p_value_p5 = stats.f_oneway(*group_arrays)
print(f"F-Statistic: {f_stat_p5:.3f}")

# Generate box plot for non-empty groups
plt.figure(figsize=(8, 6))
box_data_p5 = [groups_p5[cat] for cat in non_empty_cats]
bp = plt.boxplot(box_data_p5, labels=non_empty_cats, patch_artist=True)
colors = ['lightblue', 'lightgreen', 'lightsalmon'][:len(non_empty_cats)]
for patch, color in zip(bp['boxes'], colors):
    patch.set_facecolor(color)
plt.xlabel('Fifty Category')
plt.ylabel('SR')
plt.title('Procedure 5: ANOVA - Strike Rate by Fifty Category')
plt.tight_layout()
plt.savefig('p5_anova_boxplot.png', dpi=150)
plt.close()
print("Plot saved: p5_anova_boxplot.png\n")

# ---------- Procedure 6: Chi-square Test of Independence ----------
print("="*60)
print("Procedure 6: Chi-square Test - ODI Bowling")
print("="*60)

# Load original unfiltered dataset
odbo_p6 = pd.read_excel('odbo.xlsx')

# Create economy efficiency categories: Econ < 4.5 -> LowEcon, >= 4.5 -> HighEcon
odbo_p6['econ_category'] = odbo_p6['Econ'].apply(
    lambda x: 'LowEcon' if x < 4.5 else 'HighEcon' if pd.notna(x) else np.nan
)

# Create haul achievement categories: column '5' threshold
# Assuming HasFiveWicket if any 5-wicket hauls exist (>=1), NoFiveWicket if 0
odbo_p6['haul_category'] = odbo_p6['5'].apply(
    lambda x: 'HasFiveWicket' if x >= 1 else 'NoFiveWicket' if pd.notna(x) else np.nan
)

# Remove records where Econ OR column 5 is missing
mask_p6 = odbo_p6['Econ'].notna() & odbo_p6['5'].notna()
odbo_p6_clean = odbo_p6[mask_p6].copy()

# Construct contingency table
contingency_p6 = pd.crosstab(odbo_p6_clean['econ_category'], odbo_p6_clean['haul_category'])

# Calculate chi-square statistic (without Yates correction for standard chi-square)
chi2_p6, p_val_p6, dof_p6, expected_p6 = stats.chi2_contingency(contingency_p6, correction=False)
print(f"Chi-square Statistic: {chi2_p6:.4f}")

# Generate stacked bar plot
plt.figure(figsize=(8, 6))
econ_cats = ['LowEcon', 'HighEcon']
no_five = [contingency_p6.loc[cat, 'NoFiveWicket'] if cat in contingency_p6.index else 0 for cat in econ_cats]
has_five = [contingency_p6.loc[cat, 'HasFiveWicket'] if cat in contingency_p6.index else 0 for cat in econ_cats]

x = np.arange(len(econ_cats))
width = 0.6
plt.bar(x, no_five, width, label='NoFiveWicket', color='steelblue')
plt.bar(x, has_five, width, bottom=no_five, label='HasFiveWicket', color='coral')
plt.xlabel('Economy Categories')
plt.ylabel('Count Frequencies')
plt.title('Procedure 6: Chi-square - Economy vs Five-Wicket Hauls')
plt.xticks(x, econ_cats)
plt.legend()
plt.tight_layout()
plt.savefig('p6_chisquare_stacked_bar.png', dpi=150)
plt.close()
print("Plot saved: p6_chisquare_stacked_bar.png\n")

# ---------- Procedure 7: Coefficient of Variation Comparison ----------
print("="*60)
print("Procedure 7: Coefficient of Variation - ODI vs T20 Runs")
print("="*60)

# Load original unfiltered datasets
odb_p7 = pd.read_excel('odb.xlsx')
twb_p7 = pd.read_excel('twb.xlsx')

# Extract Runs removing missing values
odi_runs_p7 = odb_p7['Runs'].dropna().values
t20_runs_p7 = twb_p7['Runs'].dropna().values

# Calculate CV for each format: (std/mean)*100
cv_odi_p7 = (np.std(odi_runs_p7, ddof=0) / np.mean(odi_runs_p7)) * 100
cv_t20_p7 = (np.std(t20_runs_p7, ddof=0) / np.mean(t20_runs_p7)) * 100

# Compute absolute difference
abs_diff_p7 = abs(cv_odi_p7 - cv_t20_p7)
print(f"Absolute Difference in CV: {abs_diff_p7:.2f}")

# Generate bar plot
plt.figure(figsize=(8, 6))
formats = ['ODI', 'T20']
cvs = [cv_odi_p7, cv_t20_p7]
plt.bar(formats, cvs, color=['steelblue', 'coral'], edgecolor='black')
plt.xlabel('Format')
plt.ylabel('Coefficient of Variation (%)')
plt.title('Procedure 7: CV Comparison - ODI vs T20 Runs')
plt.tight_layout()
plt.savefig('p7_cv_bar.png', dpi=150)
plt.close()
print("Plot saved: p7_cv_bar.png\n")

# ---------- Procedure 8: Median Strike Rate Comparison ----------
print("="*60)
print("Procedure 8: Median Strike Rate - ODI vs T20")
print("="*60)

# Load original unfiltered datasets
odb_p8 = pd.read_excel('odb.xlsx')
twb_p8 = pd.read_excel('twb.xlsx')

# Extract SR removing missing values
odi_sr_p8 = odb_p8['SR'].dropna().values
t20_sr_p8 = twb_p8['SR'].dropna().values

# Calculate median strike rate for each format
median_odi_p8 = np.median(odi_sr_p8)
median_t20_p8 = np.median(t20_sr_p8)

# Determine which format has higher median
if median_t20_p8 > median_odi_p8:
    higher_format_p8 = "T20"
else:
    higher_format_p8 = "ODI"

print(f"Format with Higher Median Strike Rate: {higher_format_p8}")

# Generate box plot comparing ODI and T20 SR distributions
plt.figure(figsize=(8, 6))
box_data_p8 = [odi_sr_p8, t20_sr_p8]
bp8 = plt.boxplot(box_data_p8, labels=['ODI', 'T20'], patch_artist=True)
colors8 = ['steelblue', 'coral']
for patch, color in zip(bp8['boxes'], colors8):
    patch.set_facecolor(color)
plt.xlabel('Format')
plt.ylabel('SR')
plt.title('Procedure 8: Strike Rate Distribution - ODI vs T20')
plt.tight_layout()
plt.savefig('p8_sr_boxplot.png', dpi=150)
plt.close()
print("Plot saved: p8_sr_boxplot.png\n")

# ---------- Summary of Key Results ----------
print("="*60)
print("SUMMARY OF KEY RESULTS")
print("="*60)
print(f"P1 - RMSE: {rmse_p1:.3f}")
print(f"P2 - Accuracy: {accuracy_p2:.4f}")
print(f"P3 - Silhouette Coefficient: {silhouette_p3:.4f}")
print(f"P4 - Variance Explained by First Component: {variance_explained_p4:.2f}%")
print(f"P5 - F-Statistic: {f_stat_p5:.3f}")
print(f"P6 - Chi-square Statistic: {chi2_p6:.4f}")
print(f"P7 - Absolute Difference in CV: {abs_diff_p7:.2f}")
print(f"P8 - Format with Higher Median SR: {higher_format_p8}")
print("="*60)
