#!/usr/bin/env python3
"""
Hospital Readmission Analysis
Comprehensive statistical and predictive modeling analysis
"""

import pandas as pd
import numpy as np
import matplotlib.pyplot as plt
import seaborn as sns
from sklearn.preprocessing import StandardScaler
from sklearn.linear_model import LogisticRegression
from sklearn.metrics import roc_curve, auc, f1_score
from scipy.stats import chi2_contingency
import statsmodels.api as sm
from statsmodels.regression.quantile_regression import QuantReg
import warnings
warnings.filterwarnings('ignore')

# Set style for visualizations
sns.set_style('whitegrid')
plt.rcParams['figure.figsize'] = (12, 6)

print("="*80)
print("HOSPITAL READMISSION ANALYSIS")
print("="*80)

# ============================================================================
# 1. DATA LOADING AND VALIDATION
# ============================================================================
print("\n[1] Loading and validating datasets...")

# Load datasets
train_df = pd.read_csv('train_df.csv')
test_df = pd.read_csv('test_df.csv')
sample_submission = pd.read_csv('sample_submission.csv')

print(f"Initial train_df shape: {train_df.shape}")
print(f"Initial test_df shape: {test_df.shape}")

# Data validation: Keep only rows with valid numeric values
# for num_procedures, days_in_hospital, and comorbidity_score
required_cols = ['num_procedures', 'days_in_hospital', 'comorbidity_score']

# Before filtering
print(f"\nBefore filtering - train_df shape: {train_df.shape}")

# Filter train_df
for col in required_cols:
    train_df = train_df[pd.to_numeric(train_df[col], errors='coerce').notna()]

# Filter test_df
for col in required_cols:
    test_df = test_df[pd.to_numeric(test_df[col], errors='coerce').notna()]

# Ensure numeric types
for col in required_cols:
    train_df[col] = pd.to_numeric(train_df[col])
    test_df[col] = pd.to_numeric(test_df[col])

# Also ensure age is numeric
train_df['age'] = pd.to_numeric(train_df['age'], errors='coerce')
train_df = train_df[train_df['age'].notna()]

test_df['age'] = pd.to_numeric(test_df['age'], errors='coerce')
test_df = test_df[test_df['age'].notna()]

print(f"After filtering - train_df shape: {train_df.shape}")
print(f"After filtering - test_df shape: {test_df.shape}")

# ============================================================================
# 2. LOGISTIC REGRESSION WITH STANDARDIZED PREDICTORS
# ============================================================================
print("\n[2] Building logistic regression model with standardized predictors...")

# Define predictors for the model
predictors = ['age', 'num_procedures', 'days_in_hospital', 'comorbidity_score']
target = 'readmitted'

# Prepare data
X_train = train_df[predictors].copy()
y_train = train_df[target].copy()

# Standardize using z-score transformation
scaler = StandardScaler()
X_train_scaled = scaler.fit_transform(X_train)
X_train_scaled_df = pd.DataFrame(X_train_scaled, columns=predictors, index=X_train.index)

# Fit logistic regression
logreg = LogisticRegression(random_state=42, max_iter=1000, solver='lbfgs')
logreg.fit(X_train_scaled, y_train)

print(f"Model trained successfully")
print(f"Intercept: {logreg.intercept_[0]:.4f}")
print(f"Coefficients:")
for name, coef in zip(predictors, logreg.coef_[0]):
    print(f"  {name}: {coef:.4f}")

# ============================================================================
# 3. ROC ANALYSIS AND OPTIMAL THRESHOLD (YOUDEN J STATISTIC)
# ============================================================================
print("\n[3] Performing ROC analysis and finding optimal threshold...")

# Get predicted probabilities
y_pred_proba = logreg.predict_proba(X_train_scaled)[:, 1]

# Calculate ROC curve
fpr, tpr, thresholds = roc_curve(y_train, y_pred_proba)

# Calculate AUC
roc_auc = auc(fpr, tpr)
print(f"AUC: {roc_auc:.3f}")

# Calculate Youden J statistic (TPR - FPR)
youden_j = tpr - fpr

# Find optimal threshold (maximum Youden J)
optimal_idx = np.argmax(youden_j)
optimal_threshold = thresholds[optimal_idx]
print(f"Optimal threshold (Youden J): {optimal_threshold:.3f}")

# Make predictions using optimal threshold
y_pred_optimal = (y_pred_proba >= optimal_threshold).astype(int)

# Calculate F1 score at optimal threshold
f1_optimal = f1_score(y_train, y_pred_optimal)
print(f"F1 score at optimal threshold: {f1_optimal:.3f}")

# ============================================================================
# 4. CHI-SQUARE TEST OF INDEPENDENCE (GENDER vs READMITTED)
# ============================================================================
print("\n[4] Conducting chi-square test for gender and readmitted...")

# Create contingency table
contingency_table = pd.crosstab(train_df['gender'], train_df['readmitted'])
print("\nContingency table:")
print(contingency_table)

# Perform chi-square test
chi2, p_value, dof, expected = chi2_contingency(contingency_table)
print(f"\nChi-square statistic: {chi2:.3f}")
print(f"P-value: {p_value:.4f}")
print(f"Degrees of freedom: {dof}")

# ============================================================================
# 5. CAUSAL COEFFICIENT ESTIMATION (MULTIPLE LOGISTIC REGRESSION)
# ============================================================================
print("\n[5] Estimating causal coefficient of days_in_hospital...")

# Fit multiple logistic regression using statsmodels for detailed output
# Predictors: days_in_hospital, comorbidity_score, num_procedures
causal_predictors = ['days_in_hospital', 'comorbidity_score', 'num_procedures']
X_causal = train_df[causal_predictors].copy()
X_causal = sm.add_constant(X_causal)  # Add intercept
y_causal = train_df[target].copy()

# Fit logistic regression
logit_model = sm.Logit(y_causal, X_causal)
logit_result = logit_model.fit(disp=False)

print("\nLogistic Regression Summary:")
print(logit_result.summary())

# Extract coefficient for days_in_hospital
causal_coef = logit_result.params['days_in_hospital']
print(f"\nCausal coefficient of days_in_hospital: {causal_coef:.4f}")

# ============================================================================
# 6. QUANTILE REGRESSION (τ = 0.90)
# ============================================================================
print("\n[6] Performing quantile regression at τ=0.90...")

# Dependent variable: days_in_hospital
# Predictors: num_procedures, comorbidity_score
quant_predictors = ['num_procedures', 'comorbidity_score']
X_quant = train_df[quant_predictors].copy()
X_quant = sm.add_constant(X_quant)  # Add intercept
y_quant = train_df['days_in_hospital'].copy()

# Fit quantile regression at tau = 0.90
quantreg_model = QuantReg(y_quant, X_quant)
quantreg_result = quantreg_model.fit(q=0.90)

print("\nQuantile Regression Summary (τ=0.90):")
print(quantreg_result.summary())

# Extract coefficient for comorbidity_score
quant_coef = quantreg_result.params['comorbidity_score']
print(f"\nCoefficient of comorbidity_score at τ=0.90: {quant_coef:.4f}")

# ============================================================================
# 7. BOXPLOT VISUALIZATION
# ============================================================================
print("\n[7] Creating boxplot visualization...")

# Create boxplot
plt.figure(figsize=(14, 6))
diagnosis_order = train_df.groupby('primary_diagnosis')['days_in_hospital'].median().sort_values().index
sns.boxplot(data=train_df, x='primary_diagnosis', y='days_in_hospital', order=diagnosis_order)
plt.xlabel('Primary Diagnosis', fontsize=12)
plt.ylabel('Days in Hospital', fontsize=12)
plt.title('Days in Hospital by Primary Diagnosis', fontsize=14, fontweight='bold')
plt.xticks(rotation=45, ha='right')
plt.tight_layout()
plt.savefig('boxplot_days_by_diagnosis.png', dpi=300, bbox_inches='tight')
print("Boxplot saved as 'boxplot_days_by_diagnosis.png'")

# Calculate median differences
median_by_diagnosis = train_df.groupby('primary_diagnosis')['days_in_hospital'].median()
print("\nMedian days_in_hospital by diagnosis:")
print(median_by_diagnosis.sort_values())

highest_median = median_by_diagnosis.max()
lowest_median = median_by_diagnosis.min()
median_difference = highest_median - lowest_median

print(f"\nHighest median: {highest_median:.3f} ({median_by_diagnosis.idxmax()})")
print(f"Lowest median: {lowest_median:.3f} ({median_by_diagnosis.idxmin()})")
print(f"Difference: {median_difference:.3f}")

# ============================================================================
# 8. IDENTIFY HIGHEST STANDARDIZED COEFFICIENT
# ============================================================================
print("\n[8] Identifying predictor with highest standardized coefficient...")

# Among: days_in_hospital, comorbidity_score, num_procedures
relevant_predictors = ['days_in_hospital', 'comorbidity_score', 'num_procedures']

# Get coefficients from the standardized logistic regression
coef_dict = {}
for i, name in enumerate(predictors):
    if name in relevant_predictors:
        coef_dict[name] = abs(logreg.coef_[0][i])

print("\nStandardized coefficients (absolute values):")
for name, coef in coef_dict.items():
    print(f"  {name}: {coef:.4f}")

# Find maximum
max_predictor = max(coef_dict, key=coef_dict.get)
print(f"\nPredictor with highest standardized coefficient: {max_predictor}")

# ============================================================================
# 9. SUMMARY OF RESULTS
# ============================================================================
print("\n" + "="*80)
print("SUMMARY OF RESULTS")
print("="*80)

print("\n--- Logistic Regression Model (ROC Analysis) ---")
print(f"AUC: {roc_auc:.3f}")
print(f"F1 Score at optimal threshold: {f1_optimal:.3f}")
print(f"Optimal threshold (Youden J): {optimal_threshold:.3f}")

print("\n--- Chi-Square Test (Gender vs Readmitted) ---")
print(f"Chi-square statistic: {chi2:.3f}")

print("\n--- Causal Coefficient Estimation ---")
print(f"Causal coefficient of days_in_hospital: {causal_coef:.4f}")

print("\n--- Quantile Regression (τ=0.90) ---")
print(f"Coefficient of comorbidity_score at τ=0.90: {quant_coef:.4f}")

print("\n--- Boxplot Analysis ---")
print(f"Difference between highest and lowest median diagnosis groups: {median_difference:.3f}")

print("\n--- Highest Standardized Coefficient ---")
print(f"Variable name: {max_predictor}")

print("\n" + "="*80)
print("ANALYSIS COMPLETE")
print("="*80)
