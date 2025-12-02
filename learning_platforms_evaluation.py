# ==========================================
# Integrated Learning Platforms Evaluation Script
# ==========================================
# Requirements: pandas, numpy, matplotlib, scipy, scikit-learn, statsmodels
# Input files: skillshare.csv, edx.csv, Coursera.csv (in same directory)
# Output files: Various PNG plots with analysis results
# ==========================================

import pandas as pd
import numpy as np
import matplotlib.pyplot as plt
import seaborn as sns
from scipy import stats
from scipy.stats import spearmanr, chi2_contingency
from sklearn.linear_model import LogisticRegression
import statsmodels.api as sm
import statsmodels.formula.api as smf
from statsmodels.regression.quantile_regression import QuantReg
import re
import warnings
warnings.filterwarnings('ignore')

# Set style for plots
sns.set_style("whitegrid")
plt.rcParams['figure.figsize'] = (10, 6)

# ---------- Load CSVs Robustly ----------
print("Loading datasets...")
skillshare = pd.read_csv('skillshare.csv')
edx = pd.read_csv('edx.csv')
coursera = pd.read_csv('Coursera.csv')

print(f"Initial data shapes:")
print(f"Skillshare: {skillshare.shape}")
print(f"edX: {edx.shape}")
print(f"Coursera: {coursera.shape}")

# ---------- Extract Numeric Duration Values ----------
def extract_numeric_duration(duration_str):
    """Extract leading numeric value from duration string"""
    if pd.isna(duration_str):
        return np.nan

    duration_str = str(duration_str).strip()

    # Handle formats like "1h 19m", "3h 21m", "9h 56m"
    # Extract hours and minutes, convert to total hours
    match = re.search(r'(\d+)h\s*(\d+)m', duration_str)
    if match:
        hours = float(match.group(1))
        minutes = float(match.group(2))
        return hours + minutes / 60.0

    # Handle formats like "1h", "2h"
    match = re.search(r'(\d+)h', duration_str)
    if match:
        return float(match.group(1))

    # Handle formats like "30m"
    match = re.search(r'(\d+)m', duration_str)
    if match:
        return float(match.group(1)) / 60.0

    # Handle "Less Than X Hours" format
    match = re.search(r'Less Than (\d+)\s*Hours?', duration_str, re.IGNORECASE)
    if match:
        return float(match.group(1)) / 2.0  # Take half as estimate

    # Handle "1 - 4 Weeks" format (convert to hours assuming 10 hours/week)
    match = re.search(r'(\d+)\s*-\s*(\d+)\s*Weeks?', duration_str, re.IGNORECASE)
    if match:
        low = float(match.group(1))
        high = float(match.group(2))
        avg_weeks = (low + high) / 2.0
        return avg_weeks * 10  # Approximate hours per week

    # Handle "1 - 4 Years" format (convert to hours assuming 40 hours/month * 12 months)
    match = re.search(r'(\d+)\s*-\s*(\d+)\s*Years?', duration_str, re.IGNORECASE)
    if match:
        low = float(match.group(1))
        high = float(match.group(2))
        avg_years = (low + high) / 2.0
        return avg_years * 12 * 40  # Approximate hours per year

    # Handle "3 - 6 Months" or "4 meses" format (take average in months, convert to hours assuming 40 hours/month)
    match = re.search(r'(\d+)\s*-\s*(\d+)\s*(Months?|meses)', duration_str, re.IGNORECASE)
    if match:
        low = float(match.group(1))
        high = float(match.group(2))
        avg_months = (low + high) / 2.0
        return avg_months * 40  # Approximate hours per month

    # Handle single month format "24 months" or "24 meses"
    match = re.search(r'(\d+)\s*(Months?|meses)', duration_str, re.IGNORECASE)
    if match:
        months = float(match.group(1))
        return months * 40

    # Try to extract any leading numeric value
    match = re.search(r'^(\d+\.?\d*)', duration_str)
    if match:
        return float(match.group(1))

    return np.nan

# ---------- Clean Skillshare Dataset ----------
print("\n" + "="*50)
print("Processing Skillshare Dataset")
print("="*50)

# Extract numeric duration
skillshare['duration_numeric'] = skillshare['duration'].apply(extract_numeric_duration)

# Extract numeric students (remove commas and " students" text)
def extract_numeric_students(students_str):
    if pd.isna(students_str):
        return np.nan
    students_str = str(students_str).replace(',', '').replace(' students', '').strip()
    try:
        return float(students_str)
    except:
        return np.nan

skillshare['students_numeric'] = skillshare['students'].apply(extract_numeric_students)

# Remove rows with missing values in required variables
skillshare_clean = skillshare.dropna(subset=['duration_numeric', 'students_numeric'])

print(f"Skillshare after cleaning: {skillshare_clean.shape}")

# ---------- Clean edX Dataset ----------
print("\n" + "="*50)
print("Processing edX Dataset")
print("="*50)

# Create binary indicator for videotranscript
edx['transcript_binary'] = edx['videotranscript'].apply(
    lambda x: 1 if pd.notna(x) and str(x).strip() != '' and str(x).lower() != 'none' else 0
)

# Remove rows with missing values in required variables
edx_clean = edx.dropna(subset=['level', 'language', 'subject'])

print(f"edX after cleaning: {edx_clean.shape}")

# ---------- Clean Coursera Dataset ----------
print("\n" + "="*50)
print("Processing Coursera Dataset")
print("="*50)

# Extract numeric duration
coursera['duration_numeric'] = coursera['duration'].apply(extract_numeric_duration)

# Extract numeric reviewcount (remove 'k' and convert)
def extract_numeric_reviewcount(review_str):
    if pd.isna(review_str):
        return np.nan
    review_str = str(review_str).strip()

    # Handle "16.4k" format
    if 'k' in review_str.lower():
        match = re.search(r'(\d+\.?\d*)', review_str)
        if match:
            return float(match.group(1)) * 1000

    # Try to extract numeric value
    match = re.search(r'(\d+\.?\d*)', review_str)
    if match:
        return float(match.group(1))

    return np.nan

coursera['reviewcount_numeric'] = coursera['reviewcount'].apply(extract_numeric_reviewcount)

# Clean rating (should already be numeric)
coursera['rating_numeric'] = pd.to_numeric(coursera['rating'], errors='coerce')

# Create binary credit eligibility (true/false to Yes/No)
coursera['credit_eligible'] = coursera['crediteligibility'].apply(
    lambda x: 'Yes' if str(x).lower() == 'true' else 'No'
)

# Remove rows with missing values in required variables
coursera_clean = coursera.dropna(subset=[
    'rating_numeric', 'reviewcount_numeric', 'level',
    'certificatetype', 'duration_numeric'
])

print(f"Coursera after cleaning: {coursera_clean.shape}")

# ---------- Analysis 1: Quantile Regression (Coursera) ----------
print("\n" + "="*50)
print("Analysis 1: Quantile Regression - Coursera Rating")
print("="*50)

# Prepare data for quantile regression
qr_data = coursera_clean[['rating_numeric', 'reviewcount_numeric']].copy()
qr_data = qr_data[qr_data['reviewcount_numeric'] > 0]  # Ensure positive for log
qr_data['log_reviewcount'] = np.log(qr_data['reviewcount_numeric'])
qr_data = qr_data.dropna()

# Fit quantile regression at 90th percentile
X = sm.add_constant(qr_data['log_reviewcount'])
y = qr_data['rating_numeric']
qr_model = QuantReg(y, X).fit(q=0.90)
qr_coefficient = qr_model.params['log_reviewcount']

print(f"Quantile Regression Coefficient (90th percentile): {qr_coefficient:.4f}")

# Generate scatter plot
plt.figure(figsize=(10, 6))
plt.scatter(qr_data['log_reviewcount'], qr_data['rating_numeric'],
            alpha=0.5, s=20, color='steelblue')
plt.xlabel('Logarithm of Review Count')
plt.ylabel('Rating')
plt.title('Coursera Rating vs Log Review Count')
plt.grid(True, alpha=0.3)
plt.tight_layout()
plt.savefig('coursera_rating_scatter.png', dpi=300, bbox_inches='tight')
plt.close()
print("Saved: coursera_rating_scatter.png")

# ---------- Analysis 2: Chi-Square Test (Coursera) ----------
print("\n" + "="*50)
print("Analysis 2: Chi-Square Test - Level vs Certificate Type")
print("="*50)

# Create contingency table
contingency_table = pd.crosstab(
    coursera_clean['level'],
    coursera_clean['certificatetype']
)

# Perform chi-square test
chi2_stat, p_value, dof, expected = chi2_contingency(contingency_table)

print(f"Chi-Square Test Statistic: {chi2_stat:.3f}")
print(f"P-value: {p_value:.6f}")

# Generate heatmap
plt.figure(figsize=(12, 8))
sns.heatmap(contingency_table, annot=True, fmt='d', cmap='YlOrRd', cbar_kws={'label': 'Count'})
plt.xlabel('Certificate Type')
plt.ylabel('Level')
plt.title('Coursera Level vs Certificate Type Distribution')
plt.tight_layout()
plt.savefig('coursera_level_certificate_heatmap.png', dpi=300, bbox_inches='tight')
plt.close()
print("Saved: coursera_level_certificate_heatmap.png")

# ---------- Analysis 3: Spearman Correlation (edX) ----------
print("\n" + "="*50)
print("Analysis 3: Spearman Correlation - edX Linguistic Diversity")
print("="*50)

# Count distinct languages and subjects per level
diversity_df = edx_clean.groupby('level').agg({
    'language': lambda x: x.nunique(),
    'subject': lambda x: x.nunique()
}).reset_index()
diversity_df.columns = ['level', 'distinct_languages', 'distinct_subjects']

# Calculate Spearman correlation
spearman_edx, p_edx = spearmanr(
    diversity_df['distinct_languages'],
    diversity_df['distinct_subjects']
)

print(f"Spearman Correlation (Language vs Subject Diversity): {spearman_edx:.3f}")
print(f"P-value: {p_edx:.6f}")

# Generate bar chart
plt.figure(figsize=(10, 6))
x_pos = np.arange(len(diversity_df))
plt.bar(x_pos, diversity_df['distinct_languages'], color='teal', alpha=0.7)
plt.xlabel('Level')
plt.ylabel('Number of Distinct Languages')
plt.title('edX Instructional Diversity - Language Count by Level')
plt.xticks(x_pos, diversity_df['level'], rotation=45, ha='right')
plt.tight_layout()
plt.savefig('edx_language_diversity_bar.png', dpi=300, bbox_inches='tight')
plt.close()
print("Saved: edx_language_diversity_bar.png")

# ---------- Analysis 4: Spearman Correlation (Skillshare) ----------
print("\n" + "="*50)
print("Analysis 4: Spearman Correlation - Skillshare Duration vs Students")
print("="*50)

# Calculate Spearman correlation
spearman_skillshare, p_skillshare = spearmanr(
    skillshare_clean['duration_numeric'],
    skillshare_clean['students_numeric']
)

print(f"Spearman Correlation (Duration vs Students): {spearman_skillshare:.3f}")
print(f"P-value: {p_skillshare:.6f}")

# Generate scatter plot
plt.figure(figsize=(10, 6))
plt.scatter(skillshare_clean['duration_numeric'],
            skillshare_clean['students_numeric'],
            alpha=0.5, s=20, color='coral')
plt.xlabel('Duration (hours)')
plt.ylabel('Students')
plt.title('Skillshare Duration vs Student Enrollment')
plt.grid(True, alpha=0.3)
plt.tight_layout()
plt.savefig('skillshare_duration_students_scatter.png', dpi=300, bbox_inches='tight')
plt.close()
print("Saved: skillshare_duration_students_scatter.png")

# ---------- Analysis 5: Two-Sample T-Test (Coursera) ----------
print("\n" + "="*50)
print("Analysis 5: Two-Sample T-Test - Credit Eligibility")
print("="*50)

# Split groups
credit_yes = coursera_clean[coursera_clean['credit_eligible'] == 'Yes']['rating_numeric']
credit_no = coursera_clean[coursera_clean['credit_eligible'] == 'No']['rating_numeric']

# Perform t-test
t_stat, t_pvalue = stats.ttest_ind(credit_yes, credit_no)

print(f"T-Statistic: {t_stat:.3f}")
print(f"P-value: {t_pvalue:.6f}")

# Generate boxplot
plt.figure(figsize=(10, 6))
data_for_box = [credit_no, credit_yes]
plt.boxplot(data_for_box, labels=['No', 'Yes'], patch_artist=True,
            boxprops=dict(facecolor='lightblue', alpha=0.7))
plt.xlabel('Credit Eligibility Status')
plt.ylabel('Rating')
plt.title('Coursera Rating by Credit Eligibility')
plt.grid(True, alpha=0.3, axis='y')
plt.tight_layout()
plt.savefig('coursera_credit_eligibility_boxplot.png', dpi=300, bbox_inches='tight')
plt.close()
print("Saved: coursera_credit_eligibility_boxplot.png")

# ---------- Analysis 6: Logistic Regression (edX) ----------
print("\n" + "="*50)
print("Analysis 6: Logistic Regression - Transcript Availability")
print("="*50)

# Encode level as numeric (ordered categorical)
level_mapping = {
    'Introductory': 0,
    'Intermediate': 1,
    'Advanced': 2
}

edx_logit = edx_clean.copy()
edx_logit['level_numeric'] = edx_logit['level'].map(level_mapping)
edx_logit = edx_logit.dropna(subset=['level_numeric', 'transcript_binary'])

# Fit logistic regression
X_logit = edx_logit[['level_numeric']]
y_logit = edx_logit['transcript_binary']
logit_model = LogisticRegression(solver='lbfgs')
logit_model.fit(X_logit, y_logit)

logit_coefficient = logit_model.coef_[0][0]

print(f"Logistic Regression Coefficient (Level): {logit_coefficient:.4f}")

# Generate line plot for predicted probabilities
level_values = np.array([0, 1, 2]).reshape(-1, 1)
predicted_probs = logit_model.predict_proba(level_values)[:, 1]

plt.figure(figsize=(10, 6))
plt.plot(['Introductory', 'Intermediate', 'Advanced'],
         predicted_probs,
         marker='o', linewidth=2, markersize=10, color='darkgreen')
plt.xlabel('edX Level')
plt.ylabel('Predicted Probability of Transcript Availability')
plt.title('Predicted Transcript Availability by Course Level')
plt.grid(True, alpha=0.3)
plt.ylim([0, 1])
plt.tight_layout()
plt.savefig('edx_transcript_availability_line.png', dpi=300, bbox_inches='tight')
plt.close()
print("Saved: edx_transcript_availability_line.png")

# ---------- Analysis 7: Coefficient of Variation (Skillshare) ----------
print("\n" + "="*50)
print("Analysis 7: Coefficient of Variation - Student Engagement")
print("="*50)

# Calculate coefficient of variation
students_mean = skillshare_clean['students_numeric'].mean()
students_std = skillshare_clean['students_numeric'].std()
cv_students = students_std / students_mean

print(f"Coefficient of Variation (Students): {cv_students:.4f}")

# Generate density plot
plt.figure(figsize=(10, 6))
skillshare_clean['students_numeric'].plot(kind='density', color='purple', linewidth=2)
plt.xlabel('Students')
plt.ylabel('Density')
plt.title('Skillshare Student Distribution')
plt.grid(True, alpha=0.3)
plt.tight_layout()
plt.savefig('skillshare_student_density.png', dpi=300, bbox_inches='tight')
plt.close()
print("Saved: skillshare_student_density.png")

# ---------- Analysis 8: Platform Duration Comparison ----------
print("\n" + "="*50)
print("Analysis 8: Platform Duration Comparison")
print("="*50)

# Log transform durations (filter out zeros and negatives first)
skillshare_duration_positive = skillshare_clean[skillshare_clean['duration_numeric'] > 0]['duration_numeric']
coursera_duration_positive = coursera_clean[coursera_clean['duration_numeric'] > 0]['duration_numeric']

skillshare_log_duration = np.log(skillshare_duration_positive)
coursera_log_duration = np.log(coursera_duration_positive)

# Calculate mean difference
mean_diff = skillshare_log_duration.mean() - coursera_log_duration.mean()

print(f"Mean Log Duration Difference (Skillshare - Coursera): {mean_diff:.3f}")

# Generate spiral plot
plt.figure(figsize=(10, 10))
ax = plt.subplot(111, projection='polar')

# Combine data for spiral plot
all_durations = []
all_platforms = []

for idx, val in enumerate(skillshare_log_duration):
    all_durations.append(val)
    all_platforms.append('Skillshare')

for idx, val in enumerate(coursera_log_duration):
    all_durations.append(val)
    all_platforms.append('Coursera')

# Create angular positions
n_points = len(all_durations)
theta = np.linspace(0, 4 * np.pi, n_points)
r = all_durations

# Color by platform
colors = ['steelblue' if p == 'Skillshare' else 'coral' for p in all_platforms]

ax.scatter(theta, r, c=colors, alpha=0.3, s=10)
ax.set_ylabel('Log-Transformed Duration', labelpad=30)
ax.set_title('Platform Instructional Duration (Spiral)', pad=20)
plt.tight_layout()
plt.savefig('platform_duration_spiral.png', dpi=300, bbox_inches='tight')
plt.close()
print("Saved: platform_duration_spiral.png")

# ---------- Analysis 9: Maximum Absolute Standardized Coefficient ----------
print("\n" + "="*50)
print("Analysis 9: Maximum Absolute Standardized Coefficient")
print("="*50)

# For quantile regression: standardize by multiplying coefficient by (SD of X / SD of Y)
X_qr = qr_data['log_reviewcount']
y_qr = qr_data['rating_numeric']
std_x_qr = X_qr.std()
std_y_qr = y_qr.std()
standardized_qr = abs(qr_coefficient) * (std_x_qr / std_y_qr)

# For logistic regression: standardize by multiplying coefficient by SD of X
# (In logistic regression, Y is binary so we standardize differently)
std_x_logit = edx_logit['level_numeric'].std()
standardized_logit = abs(logit_coefficient) * std_x_logit

print(f"Quantile Regression Coefficient (raw): {qr_coefficient:.8f}")
print(f"Standardized Quantile Regression Coefficient: {standardized_qr:.4f}")
print(f"Logistic Regression Coefficient (raw): {logit_coefficient:.8f}")
print(f"Standardized Logistic Regression Coefficient: {standardized_logit:.4f}")

# Find maximum absolute value
if standardized_qr > standardized_logit:
    max_coef_value = standardized_qr
    max_source = "Quantile Regression"
else:
    max_coef_value = standardized_logit
    max_source = "Logistic Regression"

print(f"\nMaximum Absolute Standardized Coefficient: {max_coef_value:.4f}")
print(f"Source: {max_source}")

# ---------- Summary of Key Results ----------
print("\n" + "="*60)
print("KEY RESULTS SUMMARY")
print("="*60)
print(f"1. Quantile Regression Coefficient (log reviewcount): {qr_coefficient:.4f}")
print(f"2. Chi-Square Test Statistic (level vs certificate): {chi2_stat:.3f}")
print(f"3. Spearman Correlation (edX language/subject diversity): {spearman_edx:.3f}")
print(f"4. Spearman Correlation (Skillshare duration/students): {spearman_skillshare:.3f}")
print(f"5. T-Statistic (credit eligibility): {t_stat:.3f}")
print(f"6. Logistic Regression Coefficient (transcript/level): {logit_coefficient:.4f}")
print(f"7. Coefficient of Variation (Skillshare students): {cv_students:.4f}")
print(f"8. Mean Log Duration Difference (platforms): {mean_diff:.3f}")
print(f"9. Maximum Absolute Standardized Coefficient: {max_coef_value:.4f}")
print("="*60)

print("\nAnalysis complete. All plots saved.")
