#!/usr/bin/env python3

# ==========================================
# Hospital Readmission Statistical Analysis
# ==========================================
# Requirements: pandas, numpy, matplotlib, seaborn, scikit-learn, scipy, statsmodels, pymc, arviz
# Input files: train_df.csv, test_df.csv, sample_submission.csv (in same directory)
# Output files: boxplot_days_by_diagnosis.png
#
# Purpose: Comprehensive statistical and predictive modeling analysis of hospital
#          readmission outcomes including logistic regression, ROC analysis,
#          chi-square tests, causal inference, Bayesian MCMC, and quantile regression
#
# Key Analyses:
#   1. Logistic regression with z-score standardized predictors
#   2. ROC curve analysis with Youden J statistic optimization
#   3. Chi-square test of independence (gender vs readmitted)
#   4. Causal coefficient estimation via multiple logistic regression
#   5. Bayesian inference with MCMC sampling for posterior distributions
#   6. Quantile regression at 90th percentile (τ=0.90)
#   7. Boxplot visualization of hospitalization duration by diagnosis
#   8. Standardized coefficient comparison across predictors
# ==========================================

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

# ---------- Configure Visualization Settings ----------
sns.set_style('whitegrid')
plt.rcParams['figure.figsize'] = (12, 6)

print("="*80)
print("HOSPITAL READMISSION ANALYSIS")
print("="*80)

# ============================================================================
# 1. DATA LOADING AND VALIDATION
# ============================================================================
print("\n[1] Loading and validating datasets...")

# ---------- Load CSV Files ----------
train_df = pd.read_csv('train_df.csv')
test_df = pd.read_csv('test_df.csv')
sample_submission = pd.read_csv('sample_submission.csv')

print(f"Initial train_df shape: {train_df.shape}")
print(f"Initial test_df shape: {test_df.shape}")

# ---------- Data Quality Validation ----------
# Keep only rows with valid numeric values for required columns
# This ensures analytical consistency across all operations
required_cols = ['num_procedures', 'days_in_hospital', 'comorbidity_score']

print(f"\nBefore filtering - train_df shape: {train_df.shape}")

# ---------- Filter Training Data ----------
for col in required_cols:
    train_df = train_df[pd.to_numeric(train_df[col], errors='coerce').notna()]

# ---------- Filter Test Data ----------
for col in required_cols:
    test_df = test_df[pd.to_numeric(test_df[col], errors='coerce').notna()]

# ---------- Convert to Numeric Types ----------
for col in required_cols:
    train_df[col] = pd.to_numeric(train_df[col])
    test_df[col] = pd.to_numeric(test_df[col])

# ---------- Validate Age Column ----------
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

# ---------- Define Model Variables ----------
predictors = ['age', 'num_procedures', 'days_in_hospital', 'comorbidity_score']
target = 'readmitted'

# ---------- Prepare Training Data ----------
X_train = train_df[predictors].copy()
y_train = train_df[target].copy()

# ---------- Standardize Predictors (Z-Score Transformation) ----------
# Z-score normalization ensures coefficient comparability across features
# Formula: z = (x - μ) / σ
scaler = StandardScaler()
X_train_scaled = scaler.fit_transform(X_train)
X_train_scaled_df = pd.DataFrame(X_train_scaled, columns=predictors, index=X_train.index)

# ---------- Fit Logistic Regression Model ----------
logreg = LogisticRegression(random_state=42, max_iter=1000, solver='lbfgs')
logreg.fit(X_train_scaled, y_train)

# ---------- Display Model Parameters ----------
print(f"Model trained successfully")
print(f"Intercept: {logreg.intercept_[0]:.4f}")
print(f"Coefficients:")
for name, coef in zip(predictors, logreg.coef_[0]):
    print(f"  {name}: {coef:.4f}")

# ============================================================================
# 3. ROC ANALYSIS AND OPTIMAL THRESHOLD (YOUDEN J STATISTIC)
# ============================================================================
print("\n[3] Performing ROC analysis and finding optimal threshold...")

# ---------- Generate Predicted Probabilities ----------
y_pred_proba = logreg.predict_proba(X_train_scaled)[:, 1]

# ---------- Calculate ROC Curve ----------
fpr, tpr, thresholds = roc_curve(y_train, y_pred_proba)

# ---------- Calculate Area Under Curve (AUC) ----------
roc_auc = auc(fpr, tpr)
print(f"AUC: {roc_auc:.3f}")

# ---------- Compute Youden J Statistic ----------
# Youden J = Sensitivity + Specificity - 1 = TPR - FPR
# Maximizing Youden J finds the optimal classification threshold
youden_j = tpr - fpr

# ---------- Find Optimal Threshold ----------
optimal_idx = np.argmax(youden_j)
optimal_threshold = thresholds[optimal_idx]
print(f"Optimal threshold (Youden J): {optimal_threshold:.3f}")

# ---------- Make Predictions at Optimal Threshold ----------
y_pred_optimal = (y_pred_proba >= optimal_threshold).astype(int)

# ---------- Calculate F1 Score ----------
f1_optimal = f1_score(y_train, y_pred_optimal)
print(f"F1 score at optimal threshold: {f1_optimal:.3f}")

# ============================================================================
# 4. CHI-SQUARE TEST OF INDEPENDENCE (GENDER vs READMITTED)
# ============================================================================
print("\n[4] Conducting chi-square test for gender and readmitted...")

# ---------- Create Contingency Table ----------
contingency_table = pd.crosstab(train_df['gender'], train_df['readmitted'])
print("\nContingency table:")
print(contingency_table)

# ---------- Perform Chi-Square Test ----------
# Tests null hypothesis: gender and readmission are independent
chi2, p_value, dof, expected = chi2_contingency(contingency_table)
print(f"\nChi-square statistic: {chi2:.3f}")
print(f"P-value: {p_value:.4f}")
print(f"Degrees of freedom: {dof}")

# ============================================================================
# 5. CAUSAL COEFFICIENT ESTIMATION (MULTIPLE LOGISTIC REGRESSION)
# ============================================================================
print("\n[5] Estimating causal coefficient of days_in_hospital...")

# ---------- Define Causal Model Predictors ----------
# Estimate effect of days_in_hospital while controlling for confounders
causal_predictors = ['days_in_hospital', 'comorbidity_score', 'num_procedures']
X_causal = train_df[causal_predictors].copy()
X_causal = sm.add_constant(X_causal)  # Add intercept term
y_causal = train_df[target].copy()

# ---------- Fit Multiple Logistic Regression ----------
# Using statsmodels for detailed statistical output
logit_model = sm.Logit(y_causal, X_causal)
logit_result = logit_model.fit(disp=False)

# ---------- Display Regression Summary ----------
print("\nLogistic Regression Summary:")
print(logit_result.summary())

# ---------- Extract Causal Coefficient ----------
causal_coef = logit_result.params['days_in_hospital']
print(f"\nCausal coefficient of days_in_hospital: {causal_coef:.4f}")

# ============================================================================
# 6. BAYESIAN INFERENCE WITH MCMC SAMPLING
# ============================================================================
print("\n[6] Performing Bayesian inference with MCMC sampling...")

# ---------- Import PyMC and ArviZ Libraries ----------
import pymc as pm
import arviz as az

# ---------- Prepare Data for Bayesian Model ----------
# Use the same standardized predictors as the frequentist logistic regression
X_bayes = X_train_scaled_df.values
y_bayes = y_train.values

# ---------- Define Bayesian Logistic Regression Model ----------
# Using non-informative priors for all parameters
print("Building Bayesian logistic regression model with non-informative priors...")

with pm.Model() as bayesian_model:
    # Non-informative priors for coefficients (Normal with large variance)
    # Using Normal(0, 10) as non-informative priors
    beta_age = pm.Normal('beta_age', mu=0, sigma=10)
    beta_num_procedures = pm.Normal('beta_num_procedures', mu=0, sigma=10)
    beta_days_in_hospital = pm.Normal('beta_days_in_hospital', mu=0, sigma=10)
    beta_comorbidity_score = pm.Normal('beta_comorbidity_score', mu=0, sigma=10)

    # Intercept with non-informative prior
    alpha = pm.Normal('alpha', mu=0, sigma=10)

    # Linear combination
    logit_p = (alpha +
               beta_age * X_bayes[:, 0] +
               beta_num_procedures * X_bayes[:, 1] +
               beta_days_in_hospital * X_bayes[:, 2] +
               beta_comorbidity_score * X_bayes[:, 3])

    # Likelihood (Bernoulli logistic regression)
    y_obs = pm.Bernoulli('y_obs', logit_p=logit_p, observed=y_bayes)

    # ---------- Perform MCMC Sampling ----------
    # Draw 5,000 posterior samples using NUTS sampler
    print("Drawing 5,000 posterior samples using MCMC (this may take a few minutes)...")
    trace = pm.sample(draws=5000,
                      tune=1000,  # Warm-up samples
                      random_seed=42,
                      progressbar=True,
                      return_inferencedata=True)

# ---------- Extract Posterior Statistics ----------
print("\nBayesian MCMC Sampling Complete!")

# Get posterior summary statistics
posterior_summary = az.summary(trace, var_names=['beta_days_in_hospital'])
print("\nPosterior Summary for days_in_hospital coefficient:")
print(posterior_summary)

# ---------- Extract Posterior Mean and Credible Interval ----------
# Extract posterior samples for days_in_hospital
posterior_samples_dih = trace.posterior['beta_days_in_hospital'].values.flatten()

# Calculate posterior mean
posterior_mean_dih = np.mean(posterior_samples_dih)

# Calculate 95% credible interval
credible_interval = np.percentile(posterior_samples_dih, [2.5, 97.5])
ci_lower = credible_interval[0]
ci_upper = credible_interval[1]

print(f"\nPosterior mean of days_in_hospital coefficient: {posterior_mean_dih:.4f}")
print(f"95% Credible Interval: [{ci_lower:.4f}, {ci_upper:.4f}]")

# ---------- Display All Posterior Means ----------
print("\nPosterior means for all coefficients:")
for var in ['beta_age', 'beta_num_procedures', 'beta_days_in_hospital', 'beta_comorbidity_score']:
    post_samples = trace.posterior[var].values.flatten()
    post_mean = np.mean(post_samples)
    print(f"  {var}: {post_mean:.4f}")

# ============================================================================
# 7. QUANTILE REGRESSION (τ = 0.90)
# ============================================================================
print("\n[7] Performing quantile regression at τ=0.90...")

# ---------- Define Quantile Regression Model ----------
# Dependent variable: days_in_hospital
# Predictors: num_procedures, comorbidity_score
# Quantile τ=0.90 examines high-end hospitalization behavior
quant_predictors = ['num_procedures', 'comorbidity_score']
X_quant = train_df[quant_predictors].copy()
X_quant = sm.add_constant(X_quant)  # Add intercept
y_quant = train_df['days_in_hospital'].copy()

# ---------- Fit Quantile Regression at 90th Percentile ----------
quantreg_model = QuantReg(y_quant, X_quant)
quantreg_result = quantreg_model.fit(q=0.90)

# ---------- Display Quantile Regression Summary ----------
print("\nQuantile Regression Summary (τ=0.90):")
print(quantreg_result.summary())

# ---------- Extract Coefficient for Comorbidity Score ----------
quant_coef = quantreg_result.params['comorbidity_score']
print(f"\nCoefficient of comorbidity_score at τ=0.90: {quant_coef:.4f}")

# ============================================================================
# 8. BOXPLOT VISUALIZATION
# ============================================================================
print("\n[8] Creating boxplot visualization...")

# ---------- Create Boxplot of Days by Diagnosis ----------
plt.figure(figsize=(14, 6))
diagnosis_order = train_df.groupby('primary_diagnosis')['days_in_hospital'].median().sort_values().index
sns.boxplot(data=train_df, x='primary_diagnosis', y='days_in_hospital', order=diagnosis_order)
plt.xlabel('Primary Diagnosis', fontsize=12)
plt.ylabel('Days in Hospital', fontsize=12)
plt.title('Days in Hospital by Primary Diagnosis', fontsize=14, fontweight='bold')
plt.xticks(rotation=45, ha='right')
plt.tight_layout()

# ---------- Save Boxplot to File ----------
plt.savefig('boxplot_days_by_diagnosis.png', dpi=300, bbox_inches='tight')
print("Boxplot saved as 'boxplot_days_by_diagnosis.png'")

# ---------- Calculate Median Statistics by Diagnosis ----------
median_by_diagnosis = train_df.groupby('primary_diagnosis')['days_in_hospital'].median()
print("\nMedian days_in_hospital by diagnosis:")
print(median_by_diagnosis.sort_values())

# ---------- Compute Difference Between Highest and Lowest Medians ----------
highest_median = median_by_diagnosis.max()
lowest_median = median_by_diagnosis.min()
median_difference = highest_median - lowest_median

print(f"\nHighest median: {highest_median:.3f} ({median_by_diagnosis.idxmax()})")
print(f"Lowest median: {lowest_median:.3f} ({median_by_diagnosis.idxmin()})")
print(f"Difference: {median_difference:.3f}")

# ============================================================================
# 9. IDENTIFY HIGHEST STANDARDIZED COEFFICIENT
# ============================================================================
print("\n[9] Identifying predictor with highest standardized coefficient...")

# ---------- Extract Standardized Coefficients ----------
# Compare among: days_in_hospital, comorbidity_score, num_procedures
relevant_predictors = ['days_in_hospital', 'comorbidity_score', 'num_procedures']

coef_dict = {}
for i, name in enumerate(predictors):
    if name in relevant_predictors:
        coef_dict[name] = abs(logreg.coef_[0][i])

# ---------- Display Standardized Coefficients ----------
print("\nStandardized coefficients (absolute values):")
for name, coef in coef_dict.items():
    print(f"  {name}: {coef:.4f}")

# ---------- Find Maximum Coefficient ----------
max_predictor = max(coef_dict, key=coef_dict.get)
print(f"\nPredictor with highest standardized coefficient: {max_predictor}")

# ============================================================================
# 10. SUMMARY OF RESULTS
# ============================================================================
print("\n" + "="*80)
print("SUMMARY OF RESULTS")
print("="*80)

# ---------- Logistic Regression & ROC Analysis ----------
print("\n--- Logistic Regression Model (ROC Analysis) ---")
print(f"AUC: {roc_auc:.3f}")
print(f"F1 Score at optimal threshold: {f1_optimal:.3f}")
print(f"Optimal threshold (Youden J): {optimal_threshold:.3f}")

# ---------- Chi-Square Test Results ----------
print("\n--- Chi-Square Test (Gender vs Readmitted) ---")
print(f"Chi-square statistic: {chi2:.3f}")

# ---------- Causal Analysis Results ----------
print("\n--- Causal Coefficient Estimation ---")
print(f"Causal coefficient of days_in_hospital: {causal_coef:.4f}")

# ---------- Bayesian MCMC Results ----------
print("\n--- Bayesian Inference (MCMC) ---")
print(f"Posterior mean of days_in_hospital coefficient: {posterior_mean_dih:.4f}")
print(f"95% Credible Interval: [{ci_lower:.4f}, {ci_upper:.4f}]")

# ---------- Quantile Regression Results ----------
print("\n--- Quantile Regression (τ=0.90) ---")
print(f"Coefficient of comorbidity_score at τ=0.90: {quant_coef:.4f}")

# ---------- Boxplot Analysis Results ----------
print("\n--- Boxplot Analysis ---")
print(f"Difference between highest and lowest median diagnosis groups: {median_difference:.3f}")

# ---------- Coefficient Comparison Results ----------
print("\n--- Highest Standardized Coefficient ---")
print(f"Variable name: {max_predictor}")

# ---------- Analysis Complete ----------
print("\n" + "="*80)
print("ANALYSIS COMPLETE")
print("="*80)
