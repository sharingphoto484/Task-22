# Integrated Quantitative Analysis Results

## Study Overview
This comprehensive analysis examines three datasets to quantify relationships between sales operations, customer purchasing behavior, and workforce performance through deterministic statistical modeling.

---

## Dataset 1: Sales Operations Analysis (sales_data_sample.csv)

### Data Cleaning
- Original shape: 2,823 rows × 25 columns
- After removing duplicates: 2,823 rows (no duplicates found)
- Missing numeric values imputed with column means

### Feature Engineering
- **Profit Column**: SALES − (PRICEEACH × QUANTITYORDERED × 0.7)

### Time-Series Decomposition Results
Analysis of aggregated daily sales using additive decomposition (30-day period):
- **Trend Magnitude**: 7427.380
- **Seasonal Component Strength**: 0.030

### Principal Component Analysis (PCA)
Features analyzed: SALES, QUANTITYORDERED, MSRP, Profit
- **PC1 Variance Explained**: 67.46%
- **PC2 Variance Explained**: 25.02%
- **Total Variance (PC1 + PC2)**: 92.48%

### Correlation Analysis
**Mean Absolute Correlation**: 0.529

Correlation matrix for sales features visualized in `sales_correlation_heatmap.png`

---

## Dataset 2: Customer Behavior Analysis (Shopping Trends And Customer Behaviour Dataset.csv)

### Data Cleaning
- Original shape: 3,900 rows × 18 columns
- After removing duplicates: 3,900 rows (no duplicates found)
- Missing numeric values imputed with column means

### Feature Engineering
- **Annual Spending**: Purchase Amount (USD) × 12
- **Purchase Frequency**: Mapped from categorical frequency values to numeric monthly rates

### K-Means Clustering
Standardized features: Age, Annual Spending, Purchase Frequency

**Silhouette Analysis Results**:
- **Optimal Number of Clusters**: 9.00
- **Silhouette Score**: 0.31

### Correlation Analysis
**Pearson Correlation (Age vs Annual Spending)**: -0.0104

This indicates a negligible negative correlation between age and annual spending.

---

## Dataset 3: Workforce Performance Analysis (HR_comma_sep.csv)

### Data Cleaning
- Original shape: 14,999 rows × 10 columns
- After removing duplicates: 11,991 rows (3,008 duplicates removed)
- Missing numeric values imputed with column means

### Structural Equation Model (SEM)
**Model Specification**: Latent construct "Job Satisfaction" (indicated by satisfaction_level and last_evaluation) predicts binary outcome "left" (turnover)

**Implementation**: PCA-based approach for latent variable construction

#### Measurement Model - Factor Loadings:
- **JobSat → satisfaction_level**: 0.7400
- **JobSat → last_evaluation**: 0.7400

#### Structural Model - Path Coefficient:
- **JobSat → Turnover**: -0.2277

The negative coefficient indicates that higher job satisfaction is associated with lower turnover probability.

### Chi-Square Test of Independence
**Variables**: salary × left

**Results**:
- **Chi-Square Statistic**: 175.2107
- **p-value**: 0.0000

The extremely low p-value (< 0.0001) indicates a highly significant relationship between salary level and employee turnover.

---

## Comparative Analysis

### Key Comparison
**Question**: Does the standardized path coefficient between Job Satisfaction and Turnover exceed the mean absolute correlation magnitude from the sales heatmap?

- **Mean Absolute Correlation (Sales)**: 0.529
- **|JobSat → Turnover Coefficient|**: 0.2277

**Result**: **NO** - The path coefficient (0.2277) does NOT exceed the mean correlation (0.529)

---

## Summary of Key Findings

### Sales Operations
1. Sales data exhibits strong trend component (magnitude: 7427.380)
2. Minimal seasonal variation (strength: 0.030)
3. First two principal components explain 92.48% of variance
4. Moderate to strong inter-feature correlations (mean: 0.529)

### Customer Behavior
1. Customer segmentation identified 9 optimal behavioral clusters
2. Moderate cluster quality (silhouette: 0.31)
3. Age and annual spending are essentially uncorrelated (-0.0104)

### Workforce Performance
1. Job satisfaction construct shows balanced indicator loadings (both 0.74)
2. Job satisfaction negatively predicts turnover (-0.2277)
3. Salary level significantly influences turnover (χ² = 175.21, p < 0.0001)
4. The job satisfaction-turnover relationship is weaker than typical sales feature correlations

---

## Files Generated
- `integrated_analysis.py` - Complete analysis script
- `sales_correlation_heatmap.png` - Correlation visualization
- `ANALYSIS_RESULTS.md` - This comprehensive results report

---

**Analysis Date**: 2025-11-06
**Analysis Environment**: Python with pandas, numpy, scikit-learn, statsmodels, scipy
