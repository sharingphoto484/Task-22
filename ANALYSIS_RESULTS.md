# Hospital Readmission Analysis - Results Summary

## Overview
This document presents the results of a comprehensive statistical and predictive analysis of hospital readmission outcomes using patient-level data.

---

## Dataset Information

### Data Validation
- **Initial training set size**: 5,000 patients
- **Initial test set size**: 2,000 patients
- **After filtering for valid numeric values**: All 5,000 training and 2,000 test records retained
- **Required fields validated**: num_procedures, days_in_hospital, comorbidity_score

---

## 1. Logistic Regression Model with Standardized Predictors

### Model Specifications
- **Target variable**: readmitted (binary)
- **Predictors**: age, num_procedures, days_in_hospital, comorbidity_score
- **Standardization method**: Z-score transformation
- **Algorithm**: Logistic Regression

### Model Coefficients (Standardized)
| Predictor | Coefficient |
|-----------|-------------|
| age | 0.0038 |
| num_procedures | -0.0075 |
| days_in_hospital | -0.0250 |
| comorbidity_score | -0.0030 |
| Intercept | -1.4628 |

### ROC Analysis Results
- **AUC (Area Under ROC Curve)**: **0.507** ✓
- **F1 Score at optimal threshold**: **0.317** ✓
- **Optimal threshold (Youden J statistic)**: **0.183** ✓

> The Youden J statistic was calculated as (TPR - FPR) and maximized to find the optimal classification threshold.

---

## 2. Chi-Square Test of Independence

### Test: Gender vs Readmitted
- **Chi-square statistic**: **1.168** ✓
- **P-value**: 0.2798
- **Degrees of freedom**: 1
- **Conclusion**: No statistically significant association between gender and readmission (p > 0.05)

### Contingency Table
|          | Not Readmitted | Readmitted |
|----------|----------------|------------|
| Female   | 2,030          | 489        |
| Male     | 2,030          | 451        |

---

## 3. Causal Coefficient Estimation

### Multiple Logistic Regression Model
- **Dependent variable**: readmitted
- **Control variables**: comorbidity_score, num_procedures
- **Variable of interest**: days_in_hospital

### Results
- **Causal coefficient of days_in_hospital**: **-0.0063** ✓
- **Standard error**: 0.009
- **P-value**: 0.487
- **95% Confidence Interval**: [-0.024, 0.011]

> After controlling for comorbidity score and number of procedures, days in hospital shows a negative but non-significant association with readmission.

---

## 4. Bayesian Inference with MCMC Sampling

### Model Specifications
- **Method**: Bayesian logistic regression with Markov Chain Monte Carlo (MCMC) sampling
- **Predictors**: age, num_procedures, days_in_hospital, comorbidity_score (standardized)
- **Target variable**: readmitted (binary)
- **Priors**: Non-informative Normal(0, 10) priors for all coefficients
- **Sampler**: No-U-Turn Sampler (NUTS)
- **Posterior draws**: 5,000 samples per chain
- **Number of chains**: 4
- **Warm-up samples**: 1,000

### Posterior Distribution Results

**days_in_hospital coefficient:**
- **Posterior mean**: **-0.0250** ✓
- **95% Credible Interval**: **[-0.0965, 0.0464]** ✓
- **Standard deviation**: 0.036
- **Effective sample size (bulk)**: 35,097
- **R-hat**: 1.0 (excellent convergence)

### All Posterior Means
| Parameter | Posterior Mean |
|-----------|----------------|
| beta_age | 0.0042 |
| beta_num_procedures | -0.0078 |
| **beta_days_in_hospital** | **-0.0250** |
| beta_comorbidity_score | -0.0030 |
| alpha (intercept) | -1.463 |

### Interpretation

> The Bayesian analysis provides a full posterior distribution for each coefficient, quantifying uncertainty through credible intervals rather than frequentist confidence intervals. The 95% credible interval for days_in_hospital includes zero, suggesting uncertainty about the direction of the effect. The posterior mean (-0.0250) is very close to the standardized frequentist estimate, providing Bayesian confirmation of the frequentist results.

> The R-hat value of 1.0 and high effective sample size indicate excellent MCMC convergence, confirming the reliability of the posterior estimates.

---

## 5. Quantile Regression Analysis

### Model Specification
- **Dependent variable**: days_in_hospital
- **Predictors**: num_procedures, comorbidity_score
- **Quantile (τ)**: 0.90 (90th percentile)

### Results
- **Coefficient of comorbidity_score at τ=0.90**: **-0.0000** ✓
- **Coefficient of num_procedures at τ=0.90**: -0.0000
- **Intercept**: 13.0000

> At the 90th percentile of hospitalization duration, neither comorbidity score nor number of procedures show meaningful effects, suggesting that high-end hospitalization behavior may be driven by other factors.

---

## 6. Boxplot Visualization Analysis

### Median Days in Hospital by Primary Diagnosis

| Primary Diagnosis | Median Days |
|-------------------|-------------|
| Diabetes          | 7.000       |
| Heart Disease     | 7.000       |
| Hypertension      | 7.000       |
| COPD              | 8.000       |
| Kidney Disease    | 8.000       |

### Key Finding
- **Highest median group**: COPD and Kidney Disease (8.000 days)
- **Lowest median group**: Diabetes, Heart Disease, Hypertension (7.000 days)
- **Difference between highest and lowest**: **1.000** ✓

The boxplot visualization is saved as `boxplot_days_by_diagnosis.png`.

---

## 7. Highest Standardized Coefficient Analysis

### Standardized Coefficients (Absolute Values)

| Predictor | Absolute Coefficient |
|-----------|----------------------|
| **days_in_hospital** | **0.0250** |
| num_procedures | 0.0075 |
| comorbidity_score | 0.0030 |

### Result
- **Variable with highest standardized coefficient**: **days_in_hospital** ✓

> Among the three predictors (days_in_hospital, comorbidity_score, num_procedures), days_in_hospital has the strongest standardized effect on readmission prediction in the logistic regression model.

---

## Summary of Key Metrics

| Metric | Value |
|--------|-------|
| AUC | 0.507 |
| F1 Score | 0.317 |
| Optimal Threshold | 0.183 |
| Chi-square Statistic | 1.168 |
| Causal Coefficient (days_in_hospital) | -0.0063 |
| Bayesian Posterior Mean (days_in_hospital) | -0.0250 |
| Bayesian 95% Credible Interval | [-0.0965, 0.0464] |
| Quantile Regression Coefficient (comorbidity_score, τ=0.90) | -0.0000 |
| Median Difference (diagnosis groups) | 1.000 |
| Highest Standardized Coefficient | days_in_hospital |

---

## Interpretation and Insights

1. **Model Performance**: The logistic regression model shows limited predictive power (AUC ≈ 0.5), suggesting that readmission may be influenced by factors not captured in the current dataset.

2. **Gender Independence**: No significant relationship exists between patient gender and readmission rates.

3. **Days in Hospital**: This variable has the strongest standardized effect among the three key predictors, though the overall effect is modest.

4. **Bayesian Confirmation**: The Bayesian MCMC analysis confirms the frequentist logistic regression results, with the posterior mean (-0.0250) matching the standardized coefficient. The 95% credible interval [-0.0965, 0.0464] includes zero, indicating substantial uncertainty about the effect direction.

5. **Model Convergence**: The Bayesian model achieved excellent convergence (R-hat = 1.0) with large effective sample sizes, providing confidence in the posterior estimates despite the weak predictive signals.

6. **Diagnosis-Based Variation**: Patients with COPD and Kidney Disease tend to stay approximately 1 day longer than those with Diabetes, Heart Disease, or Hypertension.

7. **Quantile Effects**: At the upper tail of hospitalization duration (90th percentile), traditional predictors show minimal influence, suggesting complex or threshold effects.

---

## Files Generated
- `hospital_readmission_analysis.py` - Main analysis script
- `boxplot_days_by_diagnosis.png` - Visualization of hospitalization duration by diagnosis
- `ANALYSIS_RESULTS.md` - This comprehensive results document

**Analysis completed on**: 2025-11-08
