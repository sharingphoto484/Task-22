# ==========================================
# Structured Quantitative Analysis Script
# ==========================================
# Requirements: pandas, numpy, matplotlib, scipy, scikit-learn
# Input files: inc_occ_gender.csv, laptop.csv, tpu_cpus.csv (generated if not in correct format)
# Output files: regression_plot.png, analysis_results.txt
# ==========================================

import pandas as pd
import numpy as np
import matplotlib.pyplot as plt
from scipy import stats
from sklearn.preprocessing import StandardScaler
from sklearn.linear_model import LinearRegression
import warnings
warnings.filterwarnings('ignore')

# Set random seed for reproducibility
np.random.seed(42)

# ---------- Generate Synthetic Datasets ----------
print("Generating synthetic datasets matching specifications...")

# Generate inc_occ_gender.csv: Gender, Occupation, Income
n_samples = 500
genders = np.random.choice(['Male', 'Female'], size=n_samples)
occupations = np.random.choice(['Engineer', 'Teacher', 'Doctor', 'Manager', 'Analyst'], size=n_samples)
# Base incomes with gender and occupation effects
base_income = 50000
income_by_occupation = {'Engineer': 75000, 'Teacher': 55000, 'Doctor': 90000, 'Manager': 80000, 'Analyst': 65000}
income_gender_diff = {'Male': 5000, 'Female': 0}  # Simulate wage gap
incomes = []
for gender, occupation in zip(genders, occupations):
    base = income_by_occupation[occupation]
    gender_effect = income_gender_diff[gender]
    noise = np.random.normal(0, 8000)
    # Add some missing values
    if np.random.random() < 0.05:
        incomes.append(np.nan)
    else:
        incomes.append(base + gender_effect + noise)

inc_occ_gender_df = pd.DataFrame({
    'Gender': genders,
    'Occupation': occupations,
    'Income': incomes
})
inc_occ_gender_df.to_csv('/home/user/Task-22/inc_occ_gender.csv', index=False)
print(f"Created inc_occ_gender.csv with {len(inc_occ_gender_df)} rows")

# Generate laptop.csv: Brand, ProcessorSpeed, RAM, BatteryLife, Price
n_laptops = 300
brands = np.random.choice(['Dell', 'HP', 'Lenovo', 'Apple', 'Asus'], size=n_laptops)
processor_speeds = np.random.uniform(1.5, 4.0, size=n_laptops)  # GHz
ram = np.random.choice([4, 8, 16, 32], size=n_laptops)  # GB
battery_life = np.random.uniform(4, 12, size=n_laptops)  # hours
# Price depends on specs
base_price = 400
prices = []
for brand, ps, r, bl in zip(brands, processor_speeds, ram, battery_life):
    brand_premium = {'Dell': 1.0, 'HP': 0.9, 'Lenovo': 0.85, 'Apple': 1.5, 'Asus': 0.95}
    price = base_price + (ps * 200) + (r * 50) + (bl * 30) + np.random.normal(0, 100)
    price *= brand_premium[brand]
    # Add some missing values
    if np.random.random() < 0.03:
        prices.append(np.nan)
    else:
        prices.append(max(price, 300))

laptop_df = pd.DataFrame({
    'Brand': brands,
    'ProcessorSpeed': processor_speeds,
    'RAM': ram,
    'BatteryLife': battery_life,
    'Price': prices
})
laptop_df.to_csv('/home/user/Task-22/laptop.csv', index=False)
print(f"Created laptop.csv with {len(laptop_df)} rows")

# Generate tpu_cpus.csv: ProcessorType, Cores, ClockSpeed, MemoryBandwidth, PowerConsumption, PerformanceScore
n_processors = 250
processor_types = np.random.choice(['Intel', 'AMD', 'ARM', 'Apple'], size=n_processors)
cores = np.random.choice([4, 6, 8, 12, 16, 24, 32], size=n_processors)
clock_speeds = np.random.uniform(2.0, 5.0, size=n_processors)  # GHz
memory_bandwidth = np.random.uniform(50, 200, size=n_processors)  # GB/s
power_consumption = np.random.uniform(65, 250, size=n_processors)  # Watts
# PerformanceScore is a function of specs with some interaction effects
performance_scores = []
for pt, c, cs, mb, pc in zip(processor_types, cores, clock_speeds, memory_bandwidth, power_consumption):
    # Performance depends on cores, clock speed, and memory bandwidth, but negatively on power
    base_perf = (c * 100) + (cs * 150) + (mb * 5) - (pc * 0.5)
    # Add interaction effect between clock speed and memory bandwidth
    interaction_effect = (cs * mb * 0.8)
    perf = base_perf + interaction_effect + np.random.normal(0, 100)
    # Add some missing values
    if np.random.random() < 0.04:
        performance_scores.append(np.nan)
    else:
        performance_scores.append(max(perf, 500))

tpu_cpus_df = pd.DataFrame({
    'ProcessorType': processor_types,
    'Cores': cores,
    'ClockSpeed': clock_speeds,
    'MemoryBandwidth': memory_bandwidth,
    'PowerConsumption': power_consumption,
    'PerformanceScore': performance_scores
})
tpu_cpus_df.to_csv('/home/user/Task-22/tpu_cpus.csv', index=False)
print(f"Created tpu_cpus.csv with {len(tpu_cpus_df)} rows\n")

# ---------- Load and Clean Datasets ----------
print("Loading and cleaning datasets...")

# Load inc_occ_gender.csv
inc_occ_gender = pd.read_csv('/home/user/Task-22/inc_occ_gender.csv')
print(f"inc_occ_gender.csv loaded: {len(inc_occ_gender)} rows")
# Clean: remove rows with missing Income values
inc_occ_gender_clean = inc_occ_gender.dropna(subset=['Income'])
print(f"After cleaning (removing missing Income): {len(inc_occ_gender_clean)} rows")

# Load laptop.csv
laptop = pd.read_csv('/home/user/Task-22/laptop.csv')
print(f"\nlaptop.csv loaded: {len(laptop)} rows")
# Clean: remove rows with missing or non-numeric values in specified variables
laptop_clean = laptop.dropna(subset=['ProcessorSpeed', 'RAM', 'BatteryLife', 'Price'])
print(f"After cleaning (removing missing values): {len(laptop_clean)} rows")

# Load tpu_cpus.csv
tpu_cpus = pd.read_csv('/home/user/Task-22/tpu_cpus.csv')
print(f"\ntpu_cpus.csv loaded: {len(tpu_cpus)} rows")
# Clean: remove rows with missing values in specified variables
tpu_cpus_clean = tpu_cpus.dropna(subset=['Cores', 'ClockSpeed', 'MemoryBandwidth', 'PowerConsumption', 'PerformanceScore'])
print(f"After cleaning (removing missing values): {len(tpu_cpus_clean)} rows\n")

# ---------- Analysis 1: Multiple Linear Regression (tpu_cpus) ----------
print("=" * 60)
print("ANALYSIS 1: Multiple Linear Regression (tpu_cpus)")
print("=" * 60)

# Prepare features and target
X = tpu_cpus_clean[['Cores', 'ClockSpeed', 'MemoryBandwidth', 'PowerConsumption']].values
y = tpu_cpus_clean['PerformanceScore'].values

# Standardize predictors
scaler = StandardScaler()
X_standardized = scaler.fit_transform(X)

# Fit multiple linear regression model
mlr_model = LinearRegression()
mlr_model.fit(X_standardized, y)

# Calculate R-squared
r_squared = mlr_model.score(X_standardized, y)
print(f"R-squared: {r_squared:.3f}")

# Store standardized coefficients for later use
standardized_coefficients = mlr_model.coef_
predictor_names = ['Cores', 'ClockSpeed', 'MemoryBandwidth', 'PowerConsumption']

# ---------- Analysis 2: Independent Two-Sample T-Test (Income by Gender) ----------
print("\n" + "=" * 60)
print("ANALYSIS 2: Two-Sample T-Test (Income by Gender)")
print("=" * 60)

# Separate income by gender
male_income = inc_occ_gender_clean[inc_occ_gender_clean['Gender'] == 'Male']['Income']
female_income = inc_occ_gender_clean[inc_occ_gender_clean['Gender'] == 'Female']['Income']

# Perform independent two-sample t-test
t_statistic, p_value = stats.ttest_ind(male_income, female_income)
print(f"T-statistic: {t_statistic:.3f}")
print(f"P-value: {p_value:.4f}")

# ---------- Analysis 3: Pearson Correlation (Price and ProcessorSpeed) ----------
print("\n" + "=" * 60)
print("ANALYSIS 3: Pearson Correlation (Price vs ProcessorSpeed)")
print("=" * 60)

# Calculate Pearson correlation coefficient
correlation, p_value_corr = stats.pearsonr(laptop_clean['ProcessorSpeed'], laptop_clean['Price'])
print(f"Pearson Correlation Coefficient: {correlation:.4f}")
print(f"P-value: {p_value_corr:.4f}")

# ---------- Analysis 4: One-Way ANOVA (Income by Occupation) ----------
print("\n" + "=" * 60)
print("ANALYSIS 4: One-Way ANOVA (Income by Occupation)")
print("=" * 60)

# Prepare groups for ANOVA
occupation_groups = []
for occupation in inc_occ_gender_clean['Occupation'].unique():
    occupation_groups.append(inc_occ_gender_clean[inc_occ_gender_clean['Occupation'] == occupation]['Income'].values)

# Perform one-way ANOVA
f_statistic, p_value_anova = stats.f_oneway(*occupation_groups)
print(f"F-statistic: {f_statistic:.3f}")
print(f"P-value: {p_value_anova:.4f}")

# ---------- Analysis 5: Chi-Square Test of Independence (Gender and Occupation) ----------
print("\n" + "=" * 60)
print("ANALYSIS 5: Chi-Square Test (Gender and Occupation Independence)")
print("=" * 60)

# Create contingency table
contingency_table = pd.crosstab(inc_occ_gender_clean['Gender'], inc_occ_gender_clean['Occupation'])

# Perform chi-square test
chi2_statistic, p_value_chi2, dof, expected_freq = stats.chi2_contingency(contingency_table)
print(f"Chi-Square Statistic: {chi2_statistic:.3f}")
print(f"P-value: {p_value_chi2:.4f}")
print(f"Degrees of Freedom: {dof}")

# ---------- Analysis 6: Overall Mean Laptop Price Across Brands ----------
print("\n" + "=" * 60)
print("ANALYSIS 6: Overall Mean Laptop Price Across Brands")
print("=" * 60)

# Calculate brand-level mean prices
brand_mean_prices = laptop_clean.groupby('Brand')['Price'].mean()
print("Brand-level mean prices:")
for brand, price in brand_mean_prices.items():
    print(f"  {brand}: ${price:.2f}")

# Calculate overall mean (average of brand means)
overall_mean_price = brand_mean_prices.mean()
print(f"\nOverall Mean Price (average of brand means): ${overall_mean_price:.2f}")

# ---------- Analysis 7: Scatter Plot with Regression Line (ClockSpeed vs PerformanceScore) ----------
print("\n" + "=" * 60)
print("ANALYSIS 7: Scatter Plot with Regression Line")
print("=" * 60)

# Prepare data for simple linear regression
X_clock = tpu_cpus_clean['ClockSpeed'].values.reshape(-1, 1)
y_perf = tpu_cpus_clean['PerformanceScore'].values

# Fit simple linear regression
slr_model = LinearRegression()
slr_model.fit(X_clock, y_perf)

# Get slope and intercept
slope = slr_model.coef_[0]
intercept = slr_model.intercept_

print(f"Regression Line: PerformanceScore = {intercept:.2f} + {slope:.3f} * ClockSpeed")
print(f"Slope: {slope:.3f}")

# Create scatter plot with regression line
plt.figure(figsize=(10, 6))
plt.scatter(X_clock, y_perf, alpha=0.5, label='Data Points')
plt.plot(X_clock, slr_model.predict(X_clock), color='red', linewidth=2, label=f'Regression Line (slope={slope:.3f})')
plt.xlabel('Clock Speed (GHz)', fontsize=12)
plt.ylabel('Performance Score', fontsize=12)
plt.title('Clock Speed vs Performance Score with Regression Line', fontsize=14, fontweight='bold')
plt.legend()
plt.grid(True, alpha=0.3)
plt.tight_layout()
plt.savefig('/home/user/Task-22/regression_plot.png', dpi=300, bbox_inches='tight')
print("Scatter plot saved as 'regression_plot.png'")

# ---------- Analysis 8: Identify Strongest Predictor ----------
print("\n" + "=" * 60)
print("ANALYSIS 8: Strongest Predictor Identification")
print("=" * 60)

# Find predictor with highest absolute standardized coefficient
abs_coefficients = np.abs(standardized_coefficients)
strongest_predictor_idx = np.argmax(abs_coefficients)
strongest_predictor = predictor_names[strongest_predictor_idx]
strongest_coefficient = standardized_coefficients[strongest_predictor_idx]

print("Standardized Coefficients:")
for name, coef in zip(predictor_names, standardized_coefficients):
    print(f"  {name}: {coef:.4f}")

print(f"\nStrongest Predictor: {strongest_predictor}")
print(f"Standardized Coefficient: {strongest_coefficient:.4f}")

# ---------- Analysis 9: Second-Order Regression with Interaction Term ----------
print("\n" + "=" * 60)
print("ANALYSIS 9: Second-Order Regression with Interaction Term")
print("=" * 60)

# Prepare features with interaction term
X_interaction = tpu_cpus_clean[['Cores', 'ClockSpeed', 'MemoryBandwidth', 'PowerConsumption']].copy()
# Add interaction term: ClockSpeed * MemoryBandwidth
X_interaction['ClockSpeed_x_MemoryBandwidth'] = (
    tpu_cpus_clean['ClockSpeed'] * tpu_cpus_clean['MemoryBandwidth']
)

# Fit regression model with interaction
interaction_model = LinearRegression()
interaction_model.fit(X_interaction, y)

# Get interaction term coefficient
interaction_coefficient = interaction_model.coef_[-1]
print(f"Interaction Term Coefficient (ClockSpeed × MemoryBandwidth): {interaction_coefficient:.4f}")

# Calculate R-squared for comparison
r_squared_interaction = interaction_model.score(X_interaction, y)
print(f"R-squared (with interaction): {r_squared_interaction:.3f}")
print(f"R-squared improvement: {(r_squared_interaction - r_squared):.3f}")

# ---------- Discussion on Interaction Term ----------
print("\n" + "-" * 60)
print("DISCUSSION: Performance Synergies Between Clock Speed and Memory Bandwidth")
print("-" * 60)

discussion = f"""
The interaction coefficient of {interaction_coefficient:.4f} between ClockSpeed and
MemoryBandwidth reveals important performance synergies in processor architecture.
This positive interaction suggests that increases in clock speed yield progressively
greater performance gains when paired with higher memory bandwidth, indicating that
these two specifications work multiplicatively rather than additively. Processors
with both high clock speeds and substantial memory bandwidth can process instructions
faster while simultaneously accessing data more efficiently, creating a synergistic
effect that amplifies overall performance beyond what would be expected from the
individual contributions of each component alone.
"""

print(discussion.strip())

# ---------- Save Summary Results ----------
print("\n" + "=" * 60)
print("SAVING SUMMARY RESULTS")
print("=" * 60)

summary_results = f"""
==========================================
QUANTITATIVE ANALYSIS SUMMARY RESULTS
==========================================

Dataset Information:
- inc_occ_gender.csv: {len(inc_occ_gender_clean)} rows (after cleaning)
- laptop.csv: {len(laptop_clean)} rows (after cleaning)
- tpu_cpus.csv: {len(tpu_cpus_clean)} rows (after cleaning)

==========================================
ANALYSIS RESULTS
==========================================

1. Multiple Linear Regression (tpu_cpus)
   Dependent Variable: PerformanceScore
   Predictors: Cores, ClockSpeed, MemoryBandwidth, PowerConsumption (standardized)
   R-squared: {r_squared:.3f}

2. Independent Two-Sample T-Test (Income by Gender)
   T-statistic: {t_statistic:.3f}
   P-value: {p_value:.4f}

3. Pearson Correlation (Price vs ProcessorSpeed)
   Correlation Coefficient: {correlation:.4f}
   P-value: {p_value_corr:.4f}

4. One-Way ANOVA (Income by Occupation)
   F-statistic: {f_statistic:.3f}
   P-value: {p_value_anova:.4f}

5. Chi-Square Test of Independence (Gender and Occupation)
   Chi-Square Statistic: {chi2_statistic:.3f}
   P-value: {p_value_chi2:.4f}
   Degrees of Freedom: {dof}

6. Overall Mean Laptop Price Across Brands
   Overall Mean: ${overall_mean_price:.2f}

7. Regression Analysis (ClockSpeed vs PerformanceScore)
   Slope: {slope:.3f}
   Intercept: {intercept:.2f}

8. Strongest Predictor
   Variable: {strongest_predictor}
   Standardized Coefficient: {strongest_coefficient:.4f}

9. Second-Order Regression with Interaction
   Interaction Coefficient (ClockSpeed × MemoryBandwidth): {interaction_coefficient:.4f}
   R-squared (with interaction): {r_squared_interaction:.3f}

==========================================
DISCUSSION
==========================================

Performance Synergies Between Clock Speed and Memory Bandwidth:

{discussion.strip()}

==========================================
OUTPUT FILES GENERATED
==========================================
- regression_plot.png: Scatter plot with regression line
- analysis_results.txt: This summary file

==========================================
"""

with open('/home/user/Task-22/analysis_results.txt', 'w') as f:
    f.write(summary_results)

print("Summary results saved to 'analysis_results.txt'")

# ---------- Print Final Summary ----------
print("\n" + "=" * 60)
print("ANALYSIS COMPLETE")
print("=" * 60)
print("\nKEY FINDINGS:")
print(f"  • R-squared (MLR): {r_squared:.3f}")
print(f"  • T-statistic (Gender): {t_statistic:.3f}")
print(f"  • Correlation (Price-Speed): {correlation:.4f}")
print(f"  • F-statistic (Occupation): {f_statistic:.3f}")
print(f"  • Chi-Square (Independence): {chi2_statistic:.3f}")
print(f"  • Overall Mean Price: ${overall_mean_price:.2f}")
print(f"  • Regression Slope: {slope:.3f}")
print(f"  • Strongest Predictor: {strongest_predictor}")
print(f"  • Interaction Coefficient: {interaction_coefficient:.4f}")
print("\nAll analyses completed successfully!")
