# ==========================================
# Global Well-Being Quantitative Analysis
# ==========================================
# Requirements: pandas, numpy, matplotlib, scipy, scikit-learn
# Input files: 2017.csv, 2018.csv, 2019.csv (in same directory)
# Output files: aligned_dataset.csv, wellbeing_regression_plot.png,
#               analysis_results.json
# ==========================================

import pandas as pd
import numpy as np
import matplotlib.pyplot as plt
from scipy import stats
from sklearn.linear_model import LinearRegression
from sklearn.preprocessing import StandardScaler
import json
import warnings
warnings.filterwarnings('ignore')

# ---------- Load CSVs Robustly ----------
print("Loading datasets...")
df_2017 = pd.read_csv('2017.csv')
df_2018 = pd.read_csv('2018.csv')
df_2019 = pd.read_csv('2019.csv')

print(f"2017 dataset: {df_2017.shape[0]} countries")
print(f"2018 dataset: {df_2018.shape[0]} countries")
print(f"2019 dataset: {df_2019.shape[0]} countries")

# ---------- Standardize Column Names ----------
print("\nStandardizing column names...")

# Rename columns in 2017 dataset to match 2018/2019 format
df_2017_clean = df_2017.rename(columns={
    'Country': 'Country or region',
    'Happiness.Score': 'Score',
    'Economy..GDP.per.Capita.': 'GDP per capita',
    'Family': 'Social support',
    'Health..Life.Expectancy.': 'Healthy life expectancy',
    'Freedom': 'Freedom to make life choices',
    'Trust..Government.Corruption.': 'Perceptions of corruption'
})

# Add year column to each dataset
df_2017_clean['Year'] = 2017
df_2018['Year'] = 2018
df_2019['Year'] = 2019

# Select relevant columns for analysis
columns_to_keep = ['Country or region', 'Year', 'Score', 'GDP per capita',
                   'Social support', 'Healthy life expectancy',
                   'Freedom to make life choices', 'Generosity',
                   'Perceptions of corruption']

df_2017_final = df_2017_clean[columns_to_keep]
df_2018_final = df_2018[columns_to_keep]
df_2019_final = df_2019[columns_to_keep]

# ---------- Align Countries Across All Years ----------
print("\nAligning countries across all three years...")

# Clean country names (strip whitespace)
df_2017_final['Country or region'] = df_2017_final['Country or region'].str.strip()
df_2018_final['Country or region'] = df_2018_final['Country or region'].str.strip()
df_2019_final['Country or region'] = df_2019_final['Country or region'].str.strip()

# Find common countries across all three years
countries_2017 = set(df_2017_final['Country or region'])
countries_2018 = set(df_2018_final['Country or region'])
countries_2019 = set(df_2019_final['Country or region'])

common_countries = countries_2017.intersection(countries_2018).intersection(countries_2019)
print(f"Common countries across all three years: {len(common_countries)}")

# Filter datasets to include only common countries
df_2017_filtered = df_2017_final[df_2017_final['Country or region'].isin(common_countries)].copy()
df_2018_filtered = df_2018_final[df_2018_final['Country or region'].isin(common_countries)].copy()
df_2019_filtered = df_2019_final[df_2019_final['Country or region'].isin(common_countries)].copy()

# ---------- Handle Missing Values ----------
print("\nHandling missing values...")

# Convert 'N/A' strings to NaN
for df in [df_2017_filtered, df_2018_filtered, df_2019_filtered]:
    df.replace('N/A', np.nan, inplace=True)
    # Convert numeric columns to float
    numeric_cols = ['Score', 'GDP per capita', 'Social support',
                    'Healthy life expectancy', 'Freedom to make life choices',
                    'Generosity', 'Perceptions of corruption']
    for col in numeric_cols:
        df[col] = pd.to_numeric(df[col], errors='coerce')

# Remove rows with missing values in key columns
key_columns = ['Score', 'GDP per capita', 'Social support',
               'Healthy life expectancy', 'Freedom to make life choices']

df_2017_clean_final = df_2017_filtered.dropna(subset=key_columns)
df_2018_clean_final = df_2018_filtered.dropna(subset=key_columns)
df_2019_clean_final = df_2019_filtered.dropna(subset=key_columns)

# Further filter to ensure same countries after removing nulls
countries_after_clean = (set(df_2017_clean_final['Country or region'])
                        .intersection(set(df_2018_clean_final['Country or region']))
                        .intersection(set(df_2019_clean_final['Country or region'])))

df_2017_clean_final = df_2017_clean_final[df_2017_clean_final['Country or region'].isin(countries_after_clean)]
df_2018_clean_final = df_2018_clean_final[df_2018_clean_final['Country or region'].isin(countries_after_clean)]
df_2019_clean_final = df_2019_clean_final[df_2019_clean_final['Country or region'].isin(countries_after_clean)]

print(f"Final dataset size after removing nulls: {len(countries_after_clean)} countries")

# Combine all three years into one dataset
df_combined = pd.concat([df_2017_clean_final, df_2018_clean_final, df_2019_clean_final],
                        ignore_index=True)

# Save aligned dataset
df_combined.to_csv('aligned_dataset.csv', index=False)
print("\nAligned dataset saved to 'aligned_dataset.csv'")

# ---------- Multiple Regression Analysis ----------
print("\n" + "="*60)
print("MULTIPLE REGRESSION ANALYSIS")
print("="*60)

# Prepare data for regression
X = df_combined[['GDP per capita', 'Social support',
                 'Healthy life expectancy', 'Freedom to make life choices']]
y = df_combined['Score']

# Fit the regression model
reg_model = LinearRegression()
reg_model.fit(X, y)

# Get coefficients and R²
coefficients = reg_model.coef_
r_squared = reg_model.score(X, y)

print("\nRegression Model: Happiness Score = f(GDP, Social Support, Life Expectancy, Freedom)")
print(f"\nRegression Coefficients (rounded to 4 decimals):")
print(f"  GDP per capita: {coefficients[0]:.4f}")
print(f"  Social support: {coefficients[1]:.4f}")
print(f"  Healthy life expectancy: {coefficients[2]:.4f}")
print(f"  Freedom to make life choices: {coefficients[3]:.4f}")
print(f"  Intercept: {reg_model.intercept_:.4f}")
print(f"\nR² value (rounded to 3 decimals): {r_squared:.3f}")

# ---------- Pearson Correlation Analysis ----------
print("\n" + "="*60)
print("PEARSON CORRELATION: GDP vs HAPPINESS")
print("="*60)

corr_2017 = df_2017_clean_final['GDP per capita'].corr(df_2017_clean_final['Score'])
corr_2018 = df_2018_clean_final['GDP per capita'].corr(df_2018_clean_final['Score'])
corr_2019 = df_2019_clean_final['GDP per capita'].corr(df_2019_clean_final['Score'])

print(f"\nPearson correlation coefficient (GDP vs Happiness):")
print(f"  2017: {corr_2017:.4f}")
print(f"  2018: {corr_2018:.4f}")
print(f"  2019: {corr_2019:.4f}")

# ---------- Average Annual Percentage Change ----------
print("\n" + "="*60)
print("HAPPINESS SCORE STABILITY OVER TIME")
print("="*60)

# Merge datasets to calculate percentage change
df_2017_merge = df_2017_clean_final[['Country or region', 'Score']].rename(columns={'Score': 'Score_2017'})
df_2018_merge = df_2018_clean_final[['Country or region', 'Score']].rename(columns={'Score': 'Score_2018'})
df_2019_merge = df_2019_clean_final[['Country or region', 'Score']].rename(columns={'Score': 'Score_2019'})

df_change = df_2017_merge.merge(df_2018_merge, on='Country or region').merge(df_2019_merge, on='Country or region')

# Calculate total percentage change from 2017 to 2019
df_change['Total_Percentage_Change'] = ((df_change['Score_2019'] - df_change['Score_2017']) /
                                         df_change['Score_2017']) * 100

# Calculate average annual percentage change (2 years period)
df_change['Annual_Percentage_Change'] = df_change['Total_Percentage_Change'] / 2

mean_annual_change = df_change['Annual_Percentage_Change'].mean()

print(f"\nMean annual percentage change in Happiness Score (2017-2019):")
print(f"  {mean_annual_change:.3f}%")

# ---------- Variance and Standard Deviation Analysis ----------
print("\n" + "="*60)
print("FREEDOM TO MAKE LIFE CHOICES - DISPERSION (2019)")
print("="*60)

freedom_variance = df_2019_clean_final['Freedom to make life choices'].var()
freedom_std = df_2019_clean_final['Freedom to make life choices'].std()

print(f"\nVariance: {freedom_variance:.4f}")
print(f"Standard Deviation: {freedom_std:.4f}")

# ---------- Top 5 Countries Analysis ----------
print("\n" + "="*60)
print("TOP 5 HAPPIEST COUNTRIES - AVERAGE RANK")
print("="*60)

# Get top 5 countries by happiness score for each year
df_2017_sorted = df_2017_clean_final.sort_values('Score', ascending=False).reset_index(drop=True)
df_2018_sorted = df_2018_clean_final.sort_values('Score', ascending=False).reset_index(drop=True)
df_2019_sorted = df_2019_clean_final.sort_values('Score', ascending=False).reset_index(drop=True)

df_2017_sorted['Rank_2017'] = range(1, len(df_2017_sorted) + 1)
df_2018_sorted['Rank_2018'] = range(1, len(df_2018_sorted) + 1)
df_2019_sorted['Rank_2019'] = range(1, len(df_2019_sorted) + 1)

# Get top 5 from each year
top5_2017 = df_2017_sorted.head(5)['Country or region'].tolist()
top5_2018 = df_2018_sorted.head(5)['Country or region'].tolist()
top5_2019 = df_2019_sorted.head(5)['Country or region'].tolist()

# Combine all top 5 countries
all_top5 = list(set(top5_2017 + top5_2018 + top5_2019))

# Calculate average rank for each top country
rank_data_2017 = df_2017_sorted[['Country or region', 'Rank_2017']]
rank_data_2018 = df_2018_sorted[['Country or region', 'Rank_2018']]
rank_data_2019 = df_2019_sorted[['Country or region', 'Rank_2019']]

rank_combined = rank_data_2017.merge(rank_data_2018, on='Country or region').merge(rank_data_2019, on='Country or region')
rank_combined['Average_Rank'] = rank_combined[['Rank_2017', 'Rank_2018', 'Rank_2019']].mean(axis=1)

top_countries_avg_rank = rank_combined[rank_combined['Country or region'].isin(all_top5)].sort_values('Average_Rank')

print("\nTop countries appearing in top 5 across 2017-2019:")
for idx, row in top_countries_avg_rank.iterrows():
    print(f"  {row['Country or region']}: Average Rank = {row['Average_Rank']:.3f}")

# Overall average rank of top 5 countries
overall_avg_rank = top_countries_avg_rank['Average_Rank'].mean()
print(f"\nOverall average rank of top countries: {overall_avg_rank:.3f}")

# ---------- Visualization: Regression Trend Lines ----------
print("\n" + "="*60)
print("CREATING VISUALIZATION")
print("="*60)

fig, ax = plt.subplots(figsize=(12, 8))

colors = {'2017': '#FF6B6B', '2018': '#4ECDC4', '2019': '#45B7D1'}
slopes = {}

# Plot data and regression lines for each year
for year, df, color in [(2017, df_2017_clean_final, colors['2017']),
                         (2018, df_2018_clean_final, colors['2018']),
                         (2019, df_2019_clean_final, colors['2019'])]:

    X_year = df['GDP per capita'].values.reshape(-1, 1)
    y_year = df['Score'].values

    # Scatter plot
    ax.scatter(X_year, y_year, alpha=0.5, label=f'{year} Data', color=color, s=60)

    # Fit regression line
    reg_year = LinearRegression()
    reg_year.fit(X_year, y_year)
    slope = reg_year.coef_[0]
    slopes[year] = slope

    # Plot regression line
    X_line = np.linspace(X_year.min(), X_year.max(), 100).reshape(-1, 1)
    y_pred = reg_year.predict(X_line)
    ax.plot(X_line, y_pred, linewidth=2.5, label=f'{year} Trend (slope={slope:.4f})',
            color=color, linestyle='--')

ax.set_xlabel('GDP per capita', fontsize=14, fontweight='bold')
ax.set_ylabel('Happiness Score', fontsize=14, fontweight='bold')
ax.set_title('GDP per Capita vs Happiness Score: Regression Trend Lines (2017-2019)',
             fontsize=16, fontweight='bold', pad=20)
ax.legend(fontsize=11, loc='lower right')
ax.grid(True, alpha=0.3, linestyle='--')

plt.tight_layout()
plt.savefig('wellbeing_regression_plot.png', dpi=300, bbox_inches='tight')
print("\nVisualization saved to 'wellbeing_regression_plot.png'")

print(f"\nRegression line slopes for each year:")
print(f"  2017: {slopes[2017]:.4f}")
print(f"  2018: {slopes[2018]:.4f}")
print(f"  2019: {slopes[2019]:.4f}")

# ---------- Identify Strongest Predictor ----------
print("\n" + "="*60)
print("STRONGEST HAPPINESS DETERMINANT")
print("="*60)

# Standardize predictors to get standardized coefficients
scaler = StandardScaler()
X_standardized = scaler.fit_transform(X)

# Fit regression with standardized data
reg_standardized = LinearRegression()
reg_standardized.fit(X_standardized, y)

standardized_coefs = reg_standardized.coef_
predictor_names = ['GDP per capita', 'Social support',
                   'Healthy life expectancy', 'Freedom to make life choices']

# Find strongest predictor
abs_coefs = np.abs(standardized_coefs)
strongest_idx = np.argmax(abs_coefs)
strongest_predictor = predictor_names[strongest_idx]
strongest_coef = standardized_coefs[strongest_idx]

print("\nStandardized regression coefficients:")
for name, coef in zip(predictor_names, standardized_coefs):
    print(f"  {name}: {coef:.4f}")

print(f"\nStrongest predictor: {strongest_predictor}")
print(f"Coefficient magnitude: {abs(strongest_coef):.4f}")

# ---------- Save Analysis Results to JSON ----------
results = {
    'regression_analysis': {
        'coefficients': {
            'GDP_per_capita': round(coefficients[0], 4),
            'Social_support': round(coefficients[1], 4),
            'Healthy_life_expectancy': round(coefficients[2], 4),
            'Freedom_to_make_life_choices': round(coefficients[3], 4),
            'Intercept': round(reg_model.intercept_, 4)
        },
        'R_squared': round(r_squared, 3)
    },
    'pearson_correlations': {
        '2017': round(corr_2017, 4),
        '2018': round(corr_2018, 4),
        '2019': round(corr_2019, 4)
    },
    'happiness_stability': {
        'mean_annual_percentage_change': round(mean_annual_change, 3)
    },
    'freedom_dispersion_2019': {
        'variance': round(freedom_variance, 4),
        'standard_deviation': round(freedom_std, 4)
    },
    'top_countries_average_rank': round(overall_avg_rank, 3),
    'regression_slopes': {
        '2017': round(slopes[2017], 4),
        '2018': round(slopes[2018], 4),
        '2019': round(slopes[2019], 4)
    },
    'strongest_predictor': {
        'variable': strongest_predictor,
        'coefficient_magnitude': round(abs(strongest_coef), 4)
    }
}

with open('analysis_results.json', 'w') as f:
    json.dump(results, f, indent=4)

print("\n" + "="*60)
print("Analysis results saved to 'analysis_results.json'")
print("="*60)

# ---------- Discussion on Cultural Transformations ----------
print("\n" + "="*60)
print("DISCUSSION: CULTURAL TRANSFORMATIONS")
print("="*60)

discussion = """
The observed changes in happiness scores between 2017 and 2019 reflect subtle but significant
cultural transformations occurring globally. The mean annual percentage change of {:.3f}% suggests
a relatively stable global well-being landscape, yet beneath this stability lie shifting priorities
in how societies conceptualize and pursue happiness. The strong predictive power of social support
and healthy life expectancy, alongside GDP per capita, indicates a cultural evolution beyond purely
materialistic definitions of well-being toward more holistic, community-oriented values. This
transformation is particularly evident in the consistent correlation between GDP and happiness
across all three years (r ≈ {:.3f}), suggesting that while economic prosperity remains foundational,
its relationship with happiness is mediated increasingly by social and health factors. The variance
in freedom to make life choices ({:.4f}) across nations in 2019 highlights divergent cultural
trajectories, where some societies prioritize individual autonomy while others maintain collectivist
frameworks, both valid pathways to well-being within their cultural contexts.
""".format(mean_annual_change, (corr_2017 + corr_2018 + corr_2019)/3, freedom_variance)

print(discussion)

# Save discussion to file
with open('cultural_transformations_discussion.txt', 'w') as f:
    f.write("DISCUSSION: How Observed Changes in Happiness Scores Reflect Cultural Transformations\n")
    f.write("="*90 + "\n\n")
    f.write(discussion)

print("\nDiscussion saved to 'cultural_transformations_discussion.txt'")

print("\n" + "="*60)
print("ANALYSIS COMPLETE")
print("="*60)
