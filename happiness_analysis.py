# ==========================================
# Happiness Analysis Script (2015-2017)
# ==========================================
# Requirements: pandas, numpy, matplotlib, scipy, scikit-learn, statsmodels
# Input files: 2015.csv, 2016.csv, 2017.csv (in same directory)
# Output files: mean_happiness_by_year.png
# Key outputs: Printed to console
# ==========================================

import pandas as pd
import numpy as np
import matplotlib.pyplot as plt
from scipy import stats
from scipy.stats import kendalltau
from sklearn.decomposition import PCA
from sklearn.preprocessing import StandardScaler
import warnings
warnings.filterwarnings('ignore')

# ---------- Load CSVs Robustly ----------
print("Loading data files...")
df_2015 = pd.read_csv('2015.csv')
df_2016 = pd.read_csv('2016.csv')
df_2017 = pd.read_csv('2017.csv')

# ---------- Standardize Column Names ----------
print("Standardizing column names across years...")

# 2015: Rename columns to standard names
df_2015 = df_2015.rename(columns={
    'Happiness Score': 'Happiness_Score',
    'Economy (GDP per Capita)': 'Economy_GDP_per_Capita',
    'Family': 'Family',
    'Health (Life Expectancy)': 'Health_Life_Expectancy',
    'Freedom': 'Freedom',
    'Generosity': 'Generosity',
    'Trust (Government Corruption)': 'Trust_Government_Corruption',
    'Happiness Rank': 'Happiness_Rank',
    'Dystopia Residual': 'Dystopia_Residual'
})

# 2016: Rename columns to standard names (identical to 2015)
df_2016 = df_2016.rename(columns={
    'Happiness Score': 'Happiness_Score',
    'Economy (GDP per Capita)': 'Economy_GDP_per_Capita',
    'Family': 'Family',
    'Health (Life Expectancy)': 'Health_Life_Expectancy',
    'Freedom': 'Freedom',
    'Generosity': 'Generosity',
    'Trust (Government Corruption)': 'Trust_Government_Corruption',
    'Happiness Rank': 'Happiness_Rank',
    'Dystopia Residual': 'Dystopia_Residual'
})

# 2017: Rename columns to standard names (different naming convention)
df_2017 = df_2017.rename(columns={
    'Happiness.Score': 'Happiness_Score',
    'Economy..GDP.per.Capita.': 'Economy_GDP_per_Capita',
    'Family': 'Family',
    'Health..Life.Expectancy.': 'Health_Life_Expectancy',
    'Freedom': 'Freedom',
    'Generosity': 'Generosity',
    'Trust..Government.Corruption.': 'Trust_Government_Corruption',
    'Happiness.Rank': 'Happiness_Rank',
    'Dystopia.Residual': 'Dystopia_Residual'
})

# Add year identifiers
df_2015['Year'] = 2015
df_2016['Year'] = 2016
df_2017['Year'] = 2017

# Select required columns
cols = ['Country', 'Happiness_Score', 'Economy_GDP_per_Capita', 'Family',
        'Health_Life_Expectancy', 'Freedom', 'Generosity',
        'Trust_Government_Corruption', 'Happiness_Rank', 'Dystopia_Residual', 'Year']

df_2015 = df_2015[cols]
df_2016 = df_2016[cols]
df_2017 = df_2017[cols]

# ---------- Pool Data for Regression ----------
print("Pooling data for analysis...")
df_pooled = pd.concat([df_2015, df_2016, df_2017], ignore_index=True)

# ---------- Two-Way Fixed Effects Regression ----------
print("\n" + "="*60)
print("ANALYSIS 1: Two-Way Fixed Effects Regression")
print("="*60)

# Prepare data for regression
predictors = ['Economy_GDP_per_Capita', 'Family', 'Health_Life_Expectancy',
              'Freedom', 'Generosity', 'Trust_Government_Corruption']

# Filter rows with required columns for regression
regression_data = df_pooled[['Country', 'Year', 'Happiness_Score'] + predictors].dropna()

# Create country dummies and year dummies
from statsmodels.regression.linear_model import OLS
from statsmodels.tools.tools import add_constant

# Create dummy variables
country_dummies = pd.get_dummies(regression_data['Country'], prefix='Country', drop_first=True).astype(float)
year_dummies = pd.get_dummies(regression_data['Year'], prefix='Year', drop_first=True).astype(float)

# Combine predictors with dummies
X = pd.concat([regression_data[predictors].reset_index(drop=True),
               country_dummies.reset_index(drop=True),
               year_dummies.reset_index(drop=True)], axis=1).astype(float)
y = regression_data['Happiness_Score'].reset_index(drop=True)

# Fit OLS model
model = OLS(y, add_constant(X)).fit(cov_type='cluster', cov_kwds={'groups': regression_data['Country'].reset_index(drop=True).values})

# Calculate within R-squared
# Get fitted values
y_pred = model.fittedvalues

# Calculate within R-squared (demeaned by country)
regression_data['fitted'] = y_pred
country_means_y = regression_data.groupby('Country')['Happiness_Score'].transform('mean')
country_means_fitted = regression_data.groupby('Country')['fitted'].transform('mean')

y_demeaned = regression_data['Happiness_Score'] - country_means_y
y_fitted_demeaned = regression_data['fitted'] - country_means_fitted

ss_res = np.sum(y_demeaned ** 2 - y_fitted_demeaned ** 2)
ss_tot = np.sum(y_demeaned ** 2)
within_r2 = 1 - (ss_res / ss_tot)

print(f"Within R-squared: {within_r2:.4f}")

# ---------- Pearson Correlation ----------
print("\n" + "="*60)
print("ANALYSIS 2: Pearson Correlation (Economy vs Happiness)")
print("="*60)

# Filter for complete cases on these two variables
corr_data = df_pooled[['Economy_GDP_per_Capita', 'Happiness_Score']].dropna()

if len(corr_data) >= 2:
    pearson_corr, _ = stats.pearsonr(corr_data['Economy_GDP_per_Capita'],
                                       corr_data['Happiness_Score'])
else:
    pearson_corr = 0.0

print(f"Pearson Correlation: {pearson_corr:.4f}")

# ---------- Kendall Rank Correlation ----------
print("\n" + "="*60)
print("ANALYSIS 3: Kendall Rank Correlation (2016 vs 2017 Ranks)")
print("="*60)

# Merge 2016 and 2017 on Country
kendall_data = pd.merge(
    df_2016[['Country', 'Happiness_Rank']],
    df_2017[['Country', 'Happiness_Rank']],
    on='Country',
    suffixes=('_2016', '_2017')
).dropna()

if len(kendall_data) >= 2:
    kendall_tau, _ = kendalltau(kendall_data['Happiness_Rank_2016'],
                                 kendall_data['Happiness_Rank_2017'])
else:
    kendall_tau = 0.0

print(f"Kendall Tau: {kendall_tau:.4f}")

# ---------- Median Absolute Year-over-Year Change ----------
print("\n" + "="*60)
print("ANALYSIS 4: Median Absolute YoY Change in Happiness Score")
print("="*60)

# Merge 2015-2016
change_15_16 = pd.merge(
    df_2015[['Country', 'Happiness_Score']],
    df_2016[['Country', 'Happiness_Score']],
    on='Country',
    suffixes=('_2015', '_2016')
).dropna()

# Merge 2016-2017
change_16_17 = pd.merge(
    df_2016[['Country', 'Happiness_Score']],
    df_2017[['Country', 'Happiness_Score']],
    on='Country',
    suffixes=('_2016', '_2017')
).dropna()

# Calculate absolute changes
if len(change_15_16) > 0:
    abs_change_15_16 = np.abs(change_15_16['Happiness_Score_2016'] - change_15_16['Happiness_Score_2015'])
else:
    abs_change_15_16 = pd.Series([])

if len(change_16_17) > 0:
    abs_change_16_17 = np.abs(change_16_17['Happiness_Score_2017'] - change_16_17['Happiness_Score_2016'])
else:
    abs_change_16_17 = pd.Series([])

# Pool changes
all_changes = pd.concat([abs_change_15_16, abs_change_16_17], ignore_index=True)

if len(all_changes) > 0:
    median_change = np.median(all_changes)
else:
    median_change = 0.0

print(f"Median Absolute Change: {median_change:.4f}")

# ---------- Count of Countries in All Three Years ----------
print("\n" + "="*60)
print("ANALYSIS 5: Countries Present in All Three Years")
print("="*60)

# Find countries present in all three years
countries_2015 = set(df_2015['Country'].unique())
countries_2016 = set(df_2016['Country'].unique())
countries_2017 = set(df_2017['Country'].unique())

countries_all_years = countries_2015.intersection(countries_2016).intersection(countries_2017)
count_all_years = len(countries_all_years)

print(f"Count: {count_all_years}")

# ---------- PCA on Standardized Predictors ----------
print("\n" + "="*60)
print("ANALYSIS 6: PCA - First Component Variance Explained")
print("="*60)

# Select predictors and filter complete cases
pca_data = df_pooled[predictors].dropna()

if len(pca_data) > 0 and pca_data.shape[1] > 0:
    # Standardize
    scaler = StandardScaler()
    pca_data_scaled = scaler.fit_transform(pca_data)

    # Fit PCA
    pca = PCA()
    pca.fit(pca_data_scaled)

    # Get variance explained by first component
    first_pc_variance = pca.explained_variance_ratio_[0] * 100
else:
    first_pc_variance = 0.0

print(f"First PC Variance Explained: {first_pc_variance:.4f}%")

# ---------- Mean Happiness by Year Figure ----------
print("\n" + "="*60)
print("ANALYSIS 7: Mean Happiness Score by Year")
print("="*60)

# Calculate mean happiness by year
mean_2015 = df_2015['Happiness_Score'].mean()
mean_2016 = df_2016['Happiness_Score'].mean()
mean_2017 = df_2017['Happiness_Score'].mean()

highest_mean = max(mean_2015, mean_2016, mean_2017)

print(f"Mean 2015: {mean_2015:.4f}")
print(f"Mean 2016: {mean_2016:.4f}")
print(f"Mean 2017: {mean_2017:.4f}")
print(f"Highest Mean: {highest_mean:.4f}")

# Create figure
plt.figure(figsize=(10, 6))
years = [2015, 2016, 2017]
means = [mean_2015, mean_2016, mean_2017]
plt.plot(years, means, marker='o', linewidth=2, markersize=10, color='#2E86AB')
plt.xlabel('Year', fontsize=12)
plt.ylabel('Mean Happiness Score', fontsize=12)
plt.title('Mean Happiness Score by Year (2015-2017)', fontsize=14, fontweight='bold')
plt.grid(True, alpha=0.3)
plt.xticks(years)
plt.tight_layout()
plt.savefig('mean_happiness_by_year.png', dpi=300)
print("Figure saved: mean_happiness_by_year.png")

# ---------- Paired t-test (2015 vs 2017) ----------
print("\n" + "="*60)
print("ANALYSIS 8: Paired t-test (2015 vs 2017)")
print("="*60)

# Merge 2015 and 2017
paired_data = pd.merge(
    df_2015[['Country', 'Happiness_Score']],
    df_2017[['Country', 'Happiness_Score']],
    on='Country',
    suffixes=('_2015', '_2017')
).dropna()

if len(paired_data) > 0:
    # Calculate difference (2017 - 2015)
    differences = paired_data['Happiness_Score_2017'] - paired_data['Happiness_Score_2015']
    mean_diff = differences.mean()

    # Perform paired t-test
    t_stat, p_value = stats.ttest_rel(paired_data['Happiness_Score_2017'],
                                        paired_data['Happiness_Score_2015'])

    # Binary indicator: 1 if mean_diff > 0 AND p < 0.05
    verdict = 1 if (mean_diff > 0 and p_value < 0.05) else 0
else:
    mean_diff = 0.0
    p_value = 1.0
    verdict = 0

print(f"Mean Difference (2017-2015): {mean_diff:.4f}")
print(f"P-value: {p_value:.4f}")
print(f"Verdict (1=significant increase, 0=otherwise): {verdict}")

# ---------- Answer to Open-Ended Question ----------
print("\n" + "="*60)
print("QUESTION: What impacts PELT penalty choice most?")
print("="*60)
print("""
ANSWER:
The PELT (Pruning Exact Linear Time) penalty choice is most significantly impacted
by the trade-off between model complexity and goodness-of-fit. A higher penalty
favors simpler models with fewer changepoints, reducing false positives but
potentially missing true changes. A lower penalty allows more changepoints,
increasing sensitivity but risking overfitting. The optimal penalty depends on
the signal-to-noise ratio in the data: noisy data benefits from higher penalties
to avoid spurious detections, while clean data with genuine structural breaks
requires lower penalties to capture all meaningful changes. Domain knowledge about
expected changepoint frequency and validation against ground truth are critical
for tuning this hyperparameter effectively.
""")

print("\n" + "="*60)
print("ANALYSIS COMPLETE")
print("="*60)
