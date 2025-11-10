# ==========================================
# Quantitative Analysis Script
# ==========================================
# Requirements: pandas, numpy, matplotlib, scipy, scikit-learn
# Input files: inc_occ_gender.csv, laptop.csv, tpu_cpus.csv (in same directory)
# Output files: scatter_regression.png, analysis_summary.json
# ==========================================

import pandas as pd
import numpy as np
import matplotlib.pyplot as plt
from scipy import stats
from sklearn.preprocessing import StandardScaler
from sklearn.linear_model import LinearRegression
import json
import re
import warnings
warnings.filterwarnings('ignore')

print("=" * 60)
print("QUANTITATIVE ANALYSIS - STRUCTURED STATISTICAL TESTING")
print("=" * 60)

# ---------- Load CSVs Robustly ----------
print("\n[1] Loading datasets...")

try:
    df_inc = pd.read_csv('inc_occ_gender.csv')
    df_laptop = pd.read_csv('laptop.csv')
    df_cpu = pd.read_csv('tpu_cpus.csv')
    print("✓ All datasets loaded successfully")
except Exception as e:
    print(f"✗ Error loading datasets: {e}")
    exit(1)

# ---------- Data Cleaning Functions ----------
print("\n[2] Cleaning datasets...")

def clean_inc_occ_gender(df):
    """Clean income/occupation/gender dataset"""
    # Keep only required columns
    cols_needed = ['Occupation', 'M_weekly', 'F_weekly', 'M_workers', 'F_workers']
    df = df[cols_needed].copy()

    # Replace 'Na' string with NaN
    df = df.replace('Na', np.nan)

    # Convert numeric columns
    for col in ['M_weekly', 'F_weekly', 'M_workers', 'F_workers']:
        df[col] = pd.to_numeric(df[col], errors='coerce')

    # Identify major categories (all uppercase occupation names)
    # and assign category to each occupation
    current_category = None
    categories = []

    for idx, row in df.iterrows():
        occupation = str(row['Occupation'])
        # Check if this is a major category (all uppercase)
        if occupation.isupper() and occupation != 'ALL OCCUPATIONS':
            current_category = occupation
            categories.append(None)  # Don't include category rows themselves
        else:
            categories.append(current_category)

    df['Category'] = categories

    # Remove rows with any missing values in required columns
    # Also remove category header rows (where Category is None)
    df_clean = df.dropna().copy()
    df_clean = df_clean[df_clean['Category'].notna()].copy()

    print(f"  - inc_occ_gender: {len(df)} → {len(df_clean)} rows (removed {len(df) - len(df_clean)} rows)")
    return df_clean

def clean_laptop(df):
    """Clean laptop dataset"""
    # Keep only required columns
    cols_needed = ['Model', 'Price', 'Ram', 'Core', 'Generation', 'Rating']
    df = df[cols_needed].copy()

    # Clean Price: remove ₹, commas, convert to numeric
    df['Price'] = df['Price'].astype(str).str.replace('₹', '').str.replace(',', '')
    df['Price'] = pd.to_numeric(df['Price'], errors='coerce')

    # Clean Ram: extract numeric value (e.g., "8 GB DDR4 RAM" → 8)
    df['Ram'] = df['Ram'].astype(str).str.extract(r'(\d+)')[0]
    df['Ram'] = pd.to_numeric(df['Ram'], errors='coerce')

    # Rating is already numeric or NaN
    df['Rating'] = pd.to_numeric(df['Rating'], errors='coerce')

    # Remove rows with missing values in required columns
    df_clean = df.dropna(subset=['Price', 'Ram', 'Rating']).copy()

    # Extract brand from Model (first word)
    df_clean['Brand'] = df_clean['Model'].str.split().str[0]

    print(f"  - laptop: {len(df)} → {len(df_clean)} rows (removed {len(df) - len(df_clean)} rows)")
    return df_clean

def clean_tpu_cpus(df):
    """Clean CPU/TPU dataset"""
    # Keep only required columns
    cols_needed = ['Name', 'Cores', 'Clock', 'L3_Cache', 'TDP', 'Process', 'Release']
    df = df[cols_needed].copy()

    # Clean Cores: extract first number
    df['Cores'] = df['Cores'].astype(str).str.extract(r'(\d+)')[0]
    df['Cores'] = pd.to_numeric(df['Cores'], errors='coerce')

    # Clean Clock: extract first numeric value (GHz)
    # Format: "2.1 to 2.4 GHz" or "1000 MHz"
    def extract_clock(clock_str):
        try:
            clock_str = str(clock_str)
            # Look for GHz values first
            if 'GHz' in clock_str:
                match = re.search(r'(\d+\.?\d*)\s*(?:to\s*(\d+\.?\d*)\s*)?GHz', clock_str)
                if match:
                    # Take first value
                    return float(match.group(1))
            # Look for MHz values
            elif 'MHz' in clock_str:
                match = re.search(r'(\d+)', clock_str)
                if match:
                    # Convert MHz to GHz
                    return float(match.group(1)) / 1000
            return np.nan
        except:
            return np.nan

    df['Clock'] = df['Clock'].apply(extract_clock)

    # Clean L3_Cache: extract numeric value from "6MB" format
    df['L3_Cache'] = df['L3_Cache'].astype(str).str.extract(r'(\d+)')[0]
    df['L3_Cache'] = pd.to_numeric(df['L3_Cache'], errors='coerce')

    # Clean TDP: extract numeric value from "65 W" format
    df['TDP'] = df['TDP'].astype(str).str.extract(r'(\d+)')[0]
    df['TDP'] = pd.to_numeric(df['TDP'], errors='coerce')

    # Remove rows with missing values in required columns
    df_clean = df.dropna(subset=['Cores', 'Clock', 'L3_Cache', 'TDP']).copy()

    print(f"  - tpu_cpus: {len(df)} → {len(df_clean)} rows (removed {len(df) - len(df_clean)} rows)")
    return df_clean

# Clean all datasets
df_inc_clean = clean_inc_occ_gender(df_inc)
df_laptop_clean = clean_laptop(df_laptop)
df_cpu_clean = clean_tpu_cpus(df_cpu)

print("✓ All datasets cleaned successfully\n")

# ---------- Initialize Results Dictionary ----------
results = {}

# ==========================================
# ANALYSIS 1: Multiple Linear Regression (tpu_cpus.csv)
# ==========================================
print("[3] Multiple Linear Regression - L3_Cache ~ Cores + Clock + TDP")
print("-" * 60)

# Prepare data
X = df_cpu_clean[['Cores', 'Clock', 'TDP']].values
y = df_cpu_clean['L3_Cache'].values

# Standardize predictors
scaler = StandardScaler()
X_scaled = scaler.fit_transform(X)

# Fit linear regression
lr_model = LinearRegression()
lr_model.fit(X_scaled, y)

# Calculate R-squared
r_squared = lr_model.score(X_scaled, y)

# Get standardized coefficients
coefficients = {
    'Cores': lr_model.coef_[0],
    'Clock': lr_model.coef_[1],
    'TDP': lr_model.coef_[2]
}

# Find highest standardized coefficient
highest_coef_var = max(coefficients, key=lambda k: abs(coefficients[k]))

print(f"  R-squared: {r_squared:.3f}")
print(f"  Standardized Coefficients:")
for var, coef in coefficients.items():
    print(f"    {var}: {coef:.6f}")
print(f"  Highest (absolute) coefficient: {highest_coef_var}")

results['regression'] = {
    'r_squared': round(r_squared, 3),
    'coefficients': {k: round(v, 6) for k, v in coefficients.items()},
    'highest_coefficient_variable': highest_coef_var
}

# ==========================================
# ANALYSIS 2: Independent Two-Sample t-test (inc_occ_gender.csv)
# ==========================================
print("\n[4] Independent Two-Sample t-test - M_weekly vs F_weekly")
print("-" * 60)

# Perform t-test
t_statistic, p_value = stats.ttest_ind(df_inc_clean['M_weekly'],
                                        df_inc_clean['F_weekly'])

print(f"  t-statistic: {t_statistic:.3f}")
print(f"  p-value: {p_value:.6f}")

results['t_test'] = {
    't_statistic': round(t_statistic, 3),
    'p_value': round(p_value, 6)
}

# ==========================================
# ANALYSIS 3: Pearson Correlation (laptop.csv)
# ==========================================
print("\n[5] Pearson Correlation - Price vs Ram")
print("-" * 60)

correlation, p_value_corr = stats.pearsonr(df_laptop_clean['Price'],
                                             df_laptop_clean['Ram'])

print(f"  Correlation coefficient: {correlation:.4f}")
print(f"  p-value: {p_value_corr:.6f}")

results['correlation'] = {
    'pearson_r': round(correlation, 4),
    'p_value': round(p_value_corr, 6)
}

# ==========================================
# ANALYSIS 4: One-Way ANOVA (inc_occ_gender.csv)
# ==========================================
print("\n[6] One-Way ANOVA - M_weekly by Occupation Category")
print("-" * 60)

# Group data by major occupation category
groups = [group['M_weekly'].values for name, group in df_inc_clean.groupby('Category')
          if len(group) > 1]  # Only include categories with multiple occupations

# Perform ANOVA
f_statistic, p_value_anova = stats.f_oneway(*groups)

print(f"  F-statistic: {f_statistic:.3f}")
print(f"  p-value: {p_value_anova:.6f}")
print(f"  Number of occupation categories: {len(groups)}")

results['anova'] = {
    'f_statistic': round(f_statistic, 3),
    'p_value': round(p_value_anova, 6),
    'n_groups': len(groups)
}

# ==========================================
# ANALYSIS 5: Chi-Square Test (inc_occ_gender.csv)
# ==========================================
print("\n[7] Chi-Square Test - M_workers vs F_workers by Occupation Category")
print("-" * 60)

# Create contingency table by major occupation category
contingency_table = df_inc_clean.groupby('Category')[['M_workers', 'F_workers']].sum()

# Perform chi-square test
chi2_statistic, p_value_chi2, dof, expected = stats.chi2_contingency(contingency_table.T)

print(f"  Chi-square statistic: {chi2_statistic:.3f}")
print(f"  p-value: {p_value_chi2:.6f}")
print(f"  Degrees of freedom: {dof}")

results['chi_square'] = {
    'chi2_statistic': round(chi2_statistic, 3),
    'p_value': round(p_value_chi2, 6),
    'degrees_of_freedom': dof
}

# ==========================================
# ANALYSIS 6: Overall Mean Laptop Price (laptop.csv)
# ==========================================
print("\n[8] Overall Mean Laptop Price (average of brand-level means)")
print("-" * 60)

# Calculate mean price per brand
brand_means = df_laptop_clean.groupby('Brand')['Price'].mean()

# Calculate overall mean (average of brand means)
overall_mean = brand_means.mean()

print(f"  Number of brands: {len(brand_means)}")
print(f"  Overall mean price: ₹{overall_mean:.2f}")

results['mean_price'] = {
    'overall_mean': round(overall_mean, 2),
    'n_brands': len(brand_means)
}

# ==========================================
# ANALYSIS 7: Scatter Plot with Regression Line (tpu_cpus.csv)
# ==========================================
print("\n[9] Creating Scatter Plot - Clock vs L3_Cache with regression line")
print("-" * 60)

# Prepare data for simple linear regression (Clock vs L3_Cache)
X_scatter = df_cpu_clean['Clock'].values.reshape(-1, 1)
y_scatter = df_cpu_clean['L3_Cache'].values

# Fit simple linear regression
lr_scatter = LinearRegression()
lr_scatter.fit(X_scatter, y_scatter)

# Get slope
slope = lr_scatter.coef_[0]
intercept = lr_scatter.intercept_

print(f"  Regression equation: L3_Cache = {intercept:.3f} + {slope:.3f} × Clock")
print(f"  Slope: {slope:.3f}")

# Create scatter plot
plt.figure(figsize=(10, 6))
plt.scatter(df_cpu_clean['Clock'], df_cpu_clean['L3_Cache'],
            alpha=0.5, s=30, color='steelblue', label='Data points')

# Plot regression line
x_line = np.linspace(df_cpu_clean['Clock'].min(), df_cpu_clean['Clock'].max(), 100)
y_line = slope * x_line + intercept
plt.plot(x_line, y_line, color='red', linewidth=2, label=f'Regression line (slope={slope:.3f})')

plt.xlabel('Clock Speed (GHz)', fontsize=12)
plt.ylabel('L3 Cache (MB)', fontsize=12)
plt.title('Processor Clock Speed vs L3 Cache Size\nwith Fitted Regression Line', fontsize=14, fontweight='bold')
plt.legend(fontsize=10)
plt.grid(True, alpha=0.3)
plt.tight_layout()
plt.savefig('scatter_regression.png', dpi=300, bbox_inches='tight')
print("  ✓ Scatter plot saved as 'scatter_regression.png'")

results['regression_line'] = {
    'slope': round(slope, 3),
    'intercept': round(intercept, 3)
}

# ==========================================
# SUMMARY OUTPUT
# ==========================================
print("\n" + "=" * 60)
print("ANALYSIS SUMMARY - ALL RESULTS")
print("=" * 60)

print("\n1. MULTIPLE LINEAR REGRESSION (L3_Cache ~ Cores + Clock + TDP)")
print(f"   • R-squared: {results['regression']['r_squared']}")
print(f"   • Highest standardized coefficient: {results['regression']['highest_coefficient_variable']}")

print("\n2. INDEPENDENT TWO-SAMPLE T-TEST (M_weekly vs F_weekly)")
print(f"   • t-statistic: {results['t_test']['t_statistic']}")

print("\n3. PEARSON CORRELATION (Price vs Ram)")
print(f"   • Correlation coefficient: {results['correlation']['pearson_r']}")

print("\n4. ONE-WAY ANOVA (M_weekly by Occupation)")
print(f"   • F-statistic: {results['anova']['f_statistic']}")

print("\n5. CHI-SQUARE TEST (M_workers vs F_workers)")
print(f"   • Chi-square statistic: {results['chi_square']['chi2_statistic']}")

print("\n6. OVERALL MEAN LAPTOP PRICE")
print(f"   • Mean price (average of brand means): ₹{results['mean_price']['overall_mean']}")

print("\n7. SCATTER PLOT REGRESSION LINE")
print(f"   • Slope of regression line: {results['regression_line']['slope']}")

print("\n8. HIGHEST STANDARDIZED COEFFICIENT")
print(f"   • Variable name: {results['regression']['highest_coefficient_variable']}")

# ==========================================
# DISCUSSION
# ==========================================
print("\n" + "=" * 60)
print("DISCUSSION - PROCESSOR SPECIFICATIONS & PERFORMANCE")
print("=" * 60)

discussion = """
The regression analysis reveals how processor architecture influences cache design:
Higher core counts and TDP values strongly predict larger L3 cache sizes, indicating
that performance-oriented processors require more cache for efficient multi-core operation.
"""

print(discussion.strip())

# Save results to JSON
with open('analysis_summary.json', 'w') as f:
    json.dump(results, f, indent=2)

print("\n✓ Results saved to 'analysis_summary.json'")
print("\n" + "=" * 60)
print("ANALYSIS COMPLETE")
print("=" * 60)
