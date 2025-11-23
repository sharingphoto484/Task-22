# ==========================================
# Integrated Well-being Patterns Analysis Script
# ==========================================
# Requirements: pandas, numpy, matplotlib, scipy, scikit-learn
# Input files: 2015.csv, 2017.csv, 2019.csv (in same directory)
# Output files: GDP_Score_Trend_Lines_Chart.png
# ==========================================

import pandas as pd
import numpy as np
import matplotlib.pyplot as plt
from scipy.stats import pearsonr
from sklearn.preprocessing import StandardScaler
from sklearn.linear_model import LinearRegression
from sklearn.decomposition import PCA
import warnings
warnings.filterwarnings('ignore')

# ---------- Load CSVs Robustly ----------
print("Loading datasets...")

# Load 2015 data
df_2015 = pd.read_csv('2015.csv')
df_2015 = df_2015.rename(columns={
    'Country': 'Country',
    'Happiness Score': 'Score',
    'Economy (GDP per Capita)': 'GDP',
    'Family': 'SocialSupport',
    'Health (Life Expectancy)': 'Health',
    'Freedom': 'Freedom',
    'Generosity': 'Generosity',
    'Trust (Government Corruption)': 'Corruption'
})
df_2015 = df_2015[['Country', 'Score', 'GDP', 'SocialSupport', 'Health', 'Freedom', 'Generosity', 'Corruption']]
df_2015['Year'] = 2015

# Load 2017 data
df_2017 = pd.read_csv('2017.csv')
df_2017 = df_2017.rename(columns={
    'Country': 'Country',
    'Happiness.Score': 'Score',
    'Economy..GDP.per.Capita.': 'GDP',
    'Family': 'SocialSupport',
    'Health..Life.Expectancy.': 'Health',
    'Freedom': 'Freedom',
    'Generosity': 'Generosity',
    'Trust..Government.Corruption.': 'Corruption'
})
df_2017 = df_2017[['Country', 'Score', 'GDP', 'SocialSupport', 'Health', 'Freedom', 'Generosity', 'Corruption']]
df_2017['Year'] = 2017

# Load 2019 data
df_2019 = pd.read_csv('2019.csv')
df_2019 = df_2019.rename(columns={
    'Country or region': 'Country',
    'Score': 'Score',
    'GDP per capita': 'GDP',
    'Social support': 'SocialSupport',
    'Healthy life expectancy': 'Health',
    'Freedom to make life choices': 'Freedom',
    'Generosity': 'Generosity',
    'Perceptions of corruption': 'Corruption'
})
df_2019 = df_2019[['Country', 'Score', 'GDP', 'SocialSupport', 'Health', 'Freedom', 'Generosity', 'Corruption']]
df_2019['Year'] = 2019

print(f"Initial counts - 2015: {len(df_2015)}, 2017: {len(df_2017)}, 2019: {len(df_2019)}")

# ---------- Remove Missing and Non-numeric Values ----------
print("\nCleaning datasets...")

numeric_cols = ['Score', 'GDP', 'SocialSupport', 'Health', 'Freedom', 'Generosity', 'Corruption']

for df in [df_2015, df_2017, df_2019]:
    # Convert to numeric, coercing errors to NaN
    for col in numeric_cols:
        df[col] = pd.to_numeric(df[col], errors='coerce')

# Remove rows with missing values
df_2015_clean = df_2015.dropna(subset=numeric_cols).reset_index(drop=True)
df_2017_clean = df_2017.dropna(subset=numeric_cols).reset_index(drop=True)
df_2019_clean = df_2019.dropna(subset=numeric_cols).reset_index(drop=True)

print(f"After cleaning - 2015: {len(df_2015_clean)}, 2017: {len(df_2017_clean)}, 2019: {len(df_2019_clean)}")

# ---------- Keep Only Common Countries ----------
print("\nIdentifying common countries across all years...")

countries_2015 = set(df_2015_clean['Country'].unique())
countries_2017 = set(df_2017_clean['Country'].unique())
countries_2019 = set(df_2019_clean['Country'].unique())

common_countries = countries_2015 & countries_2017 & countries_2019
print(f"Common countries: {len(common_countries)}")

df_2015_common = df_2015_clean[df_2015_clean['Country'].isin(common_countries)].reset_index(drop=True)
df_2017_common = df_2017_clean[df_2017_clean['Country'].isin(common_countries)].reset_index(drop=True)
df_2019_common = df_2019_clean[df_2019_clean['Country'].isin(common_countries)].reset_index(drop=True)

# ---------- Merge Multi-year Dataset ----------
print("\nMerging datasets...")

df_merged = pd.concat([df_2015_common, df_2017_common, df_2019_common], ignore_index=True)
print(f"Merged dataset size: {len(df_merged)}")

# ---------- Multiple Linear Regression on 2019 Data ----------
print("\n" + "="*60)
print("ANALYSIS 1: Multiple Linear Regression (2019)")
print("="*60)

X_2019 = df_2019_common[['GDP', 'SocialSupport', 'Health', 'Freedom', 'Generosity', 'Corruption']].values
y_2019 = df_2019_common['Score'].values

# Standardize predictors
scaler_2019 = StandardScaler()
X_2019_std = scaler_2019.fit_transform(X_2019)

# Fit regression model
model_2019 = LinearRegression()
model_2019.fit(X_2019_std, y_2019)

r_squared = model_2019.score(X_2019_std, y_2019)
print(f"R² (2019 regression): {r_squared:.3f}")

# Store coefficients for later
predictor_names = ['GDP', 'SocialSupport', 'Health', 'Freedom', 'Generosity', 'Corruption']
coefficients_2019 = dict(zip(predictor_names, model_2019.coef_))

# ---------- Average Pearson Correlation (GDP vs Score) ----------
print("\n" + "="*60)
print("ANALYSIS 2: Average Pearson Correlation (GDP vs Score)")
print("="*60)

corr_2015, _ = pearsonr(df_2015_common['GDP'], df_2015_common['Score'])
corr_2017, _ = pearsonr(df_2017_common['GDP'], df_2017_common['Score'])
corr_2019, _ = pearsonr(df_2019_common['GDP'], df_2019_common['Score'])

avg_correlation = (corr_2015 + corr_2017 + corr_2019) / 3
print(f"2015 correlation: {corr_2015:.4f}")
print(f"2017 correlation: {corr_2017:.4f}")
print(f"2019 correlation: {corr_2019:.4f}")
print(f"Average correlation: {avg_correlation:.4f}")

# ---------- Mean Coefficient of Variation (Score) ----------
print("\n" + "="*60)
print("ANALYSIS 3: Mean Coefficient of Variation (Score)")
print("="*60)

cv_2015 = df_2015_common['Score'].std() / df_2015_common['Score'].mean()
cv_2017 = df_2017_common['Score'].std() / df_2017_common['Score'].mean()
cv_2019 = df_2019_common['Score'].std() / df_2019_common['Score'].mean()

mean_cv = (cv_2015 + cv_2017 + cv_2019) / 3
print(f"2015 CV: {cv_2015:.4f}")
print(f"2017 CV: {cv_2017:.4f}")
print(f"2019 CV: {cv_2019:.4f}")
print(f"Mean CV: {mean_cv:.4f}")

# ---------- Principal Component Analysis ----------
print("\n" + "="*60)
print("ANALYSIS 4: Principal Component Analysis")
print("="*60)

X_all = df_merged[['GDP', 'SocialSupport', 'Health', 'Freedom', 'Generosity', 'Corruption']].values

# Standardize across all years
scaler_pca = StandardScaler()
X_all_std = scaler_pca.fit_transform(X_all)

# Perform PCA
pca = PCA()
pca.fit(X_all_std)

pc1_variance = pca.explained_variance_ratio_[0]
print(f"PC1 variance explained: {pc1_variance:.3f}")

# ---------- Year-over-Year Percentage Change ----------
print("\n" + "="*60)
print("ANALYSIS 5: Mean Year-over-Year Percentage Change")
print("="*60)

# Merge data by country to compute changes
df_2015_pivot = df_2015_common.set_index('Country')['Score']
df_2017_pivot = df_2017_common.set_index('Country')['Score']
df_2019_pivot = df_2019_common.set_index('Country')['Score']

# Calculate percentage changes
pct_change_2015_2017 = ((df_2017_pivot - df_2015_pivot) / df_2015_pivot * 100)
pct_change_2017_2019 = ((df_2019_pivot - df_2017_pivot) / df_2017_pivot * 100)

# Average the two periods for each country, then take overall mean
mean_pct_changes = (pct_change_2015_2017 + pct_change_2017_2019) / 2
overall_mean_pct_change = mean_pct_changes.mean()

print(f"Mean YoY percentage change: {overall_mean_pct_change:.3f}")

# ---------- GDP-Score Trend Lines Chart ----------
print("\n" + "="*60)
print("ANALYSIS 6: GDP-Score Trend Lines Chart")
print("="*60)

fig, ax = plt.subplots(figsize=(10, 6))

# Prepare data and fit regression for each year
years_data = {
    2015: df_2015_common,
    2017: df_2017_common,
    2019: df_2019_common
}

colors = {2015: 'blue', 2017: 'green', 2019: 'red'}
slopes = {}

for year, df in years_data.items():
    X_gdp = df['GDP'].values.reshape(-1, 1)
    y_score = df['Score'].values

    # Fit simple linear regression
    reg = LinearRegression()
    reg.fit(X_gdp, y_score)

    # Store slope
    slopes[year] = reg.coef_[0]

    # Plot scatter
    ax.scatter(df['GDP'], df['Score'], alpha=0.3, color=colors[year], label=f'{year} data', s=30)

    # Plot regression line
    gdp_range = np.linspace(df['GDP'].min(), df['GDP'].max(), 100).reshape(-1, 1)
    score_pred = reg.predict(gdp_range)
    ax.plot(gdp_range, score_pred, color=colors[year], linewidth=2, label=f'{year} trend')

ax.set_xlabel('GDP', fontsize=12)
ax.set_ylabel('Score', fontsize=12)
ax.set_title('GDP–Score Trend Lines Chart', fontsize=14, fontweight='bold')
ax.legend(loc='lower right', fontsize=9)
ax.grid(True, alpha=0.3)

plt.tight_layout()
plt.savefig('GDP_Score_Trend_Lines_Chart.png', dpi=300)
print("Chart saved: GDP_Score_Trend_Lines_Chart.png")

print(f"2019 regression slope: {slopes[2019]:.4f}")

# ---------- Predictor with Highest Absolute Standardized Coefficient ----------
print("\n" + "="*60)
print("ANALYSIS 7: Top Predictor (Highest Absolute Coefficient)")
print("="*60)

max_predictor = max(coefficients_2019.items(), key=lambda x: abs(x[1]))
print(f"Predictor with highest absolute standardized coefficient: {max_predictor[0]}")
print(f"Coefficient value: {max_predictor[1]:.4f}")

# ---------- Key Output Summary ----------
print("\n" + "="*60)
print("KEY OUTPUT SUMMARY")
print("="*60)
print(f"R² (2019 regression): {r_squared:.3f}")
print(f"Average correlation (GDP vs Score): {avg_correlation:.4f}")
print(f"Mean coefficient of variation (Score): {mean_cv:.4f}")
print(f"PC1 variance explained: {pc1_variance:.3f}")
print(f"Mean YoY percentage change: {overall_mean_pct_change:.3f}")
print(f"2019 regression slope: {slopes[2019]:.4f}")
print(f"Top predictor: {max_predictor[0]}")
print("="*60)
