# Markov Volatility & Liquidity Analysis - Results Summary

## Analysis Overview

This analysis performs a comprehensive quantitative study of three Indian bank stocks (HDFCBANK, ICICIBANK, INDUSINDBK) focusing on:
1. **Markov Switching Volatility Models** - Identifying high and low volatility regimes
2. **Liquidity Commonality** - Examining shared liquidity patterns via PCA
3. **Volatility Estimation** - Using Parkinson range-based estimators
4. **Predictive Regressions** - Testing relationships between deliverability, turnover, and future volatility

## Dataset

- **Common Sample Period**: 2000-01-03 to 2021-04-30
- **Observations**: 4,984 daily returns (after differencing)
- **Variables**: Close prices, High, Low, Turnover, %Deliverble

## Key Results

### 1. Markov Switching Volatility Analysis

**Cross-bank Average Expected Duration (High Volatility State):**
- **3.839117 trading days**

**Cross-bank Average Stationary Probability (High Volatility State):**
- **0.084611** (8.46%)

#### Individual Bank Statistics:
- **HDFCBANK**: High vol σ = 0.208, Expected duration = 1.00 days, Stationary prob = 1.63%
- **ICICIBANK**: High vol σ = 0.221, Expected duration = 1.00 days, Stationary prob = 1.37%
- **INDUSINDBK**: High vol σ = 0.060, Expected duration = 9.52 days, Stationary prob = 22.38%

### 2. Liquidity Commonality (PCA on Amihud Illiquidity)

**Variance Explained by First Principal Component:**
- **72.187254%**

**PC1 Loadings (Normalized to Unit Euclidean Norm):**
- HDFCBANK: 0.536164
- ICICIBANK: 0.593443
- INDUSINDBK: 0.600295

**Minimum Loading:**
- **0.536164**

**Liquidity Commonality Verdict:**
- **1** (YES - PC1 explains ≥50% of variance AND all loadings are strictly positive)
- **Interpretation**: Strong evidence of common liquidity factors across all three banks

### 3. Parkinson Range-Based Variance

**95th Percentile of Daily Cross-Bank Average:**
- **0.002476**

### 4. Predictive Regressions (OLS with Newey-West SE, lag=5)

**Regression**: abs(log return_{t+1}) ~ β₁·%Deliverble_t + β₂·Turnover_t + intercept

**Mean P-value on %Deliverble Coefficient:**
- **0.000001** (highly statistically significant)

#### Individual Bank Results:
- **HDFCBANK**: Coef = -0.002072, NW SE = 0.000446, p-value = 0.000003
- **ICICIBANK**: Coef = -0.002279, NW SE = 0.000382, p-value < 0.000001
- **INDUSINDBK**: Coef = -0.002878, NW SE = 0.000428, p-value < 0.000001

**Interpretation**: Higher delivery percentage is associated with significantly lower next-day volatility across all three banks.

### 5. Probability Heatmap

**Maximum Smoothed Probability (High Volatility State):**
- **1.000000**

The heatmap visualizes the time-varying probability of being in the high volatility regime for each bank across the entire sample period.

## Technical Methodology

### Data Processing
- Exact daily date intersection across all three CSV files
- Daily log returns: ln(Close_t) - ln(Close_{t-1})
- Standardization: (X - μ) / σ using sample statistics on returns timeline
- No imputation applied; only removed observations lost to intersection and differencing

### Markov Switching Models
- Model: 2-state Gaussian Hidden Markov Model (HMM)
- Estimation: Maximum Likelihood using Expectation-Maximization (EM) algorithm via hmmlearn
- State identification: High volatility = state with larger conditional variance
- Smoothed probabilities: Obtained via forward-backward algorithm
- Expected duration: 1 / (1 - P_{ii}) where i = high volatility state
- Stationary probability: Analytical solution from transition matrix eigenanalysis

### Amihud Illiquidity
- Formula: |log_return_t| / Turnover_t (no rescaling applied)
- Standardized per bank before PCA
- PCA: PC1 normalized to unit Euclidean norm with positive loading on HDFCBANK

### Parkinson Variance
- Formula: [ln(High) - ln(Low)]² / (4·ln(2))
- Cross-bank daily average computed first, then 95th percentile

### OLS Regressions
- Dependent: |log_return_{t+1}|
- Independent: Standardized %Deliverble_t and Standardized Turnover_t
- Standard errors: Newey-West HAC with lag 5
- P-values: Two-sided t-tests

## Files Generated

1. **markov_volatility_liquidity_analysis.py** - Main analysis script
2. **analysis_results.json** - Numerical results in JSON format
3. **probability_heatmap.png** - Visualization of high volatility state probabilities
4. **ANALYSIS_SUMMARY.md** - This summary document

## Software Requirements

```
pandas>=1.0.0
numpy>=1.18.0
matplotlib>=3.0.0
seaborn>=0.11.0
scikit-learn>=0.22.0
statsmodels>=0.12.0
scipy>=1.4.0
hmmlearn>=0.2.7
```

## Execution

```bash
python markov_volatility_liquidity_analysis.py
```

Runtime: ~30-60 seconds on standard hardware

## Key Findings

1. **Volatility Regimes**: Banks exhibit distinct high and low volatility states, with INDUSINDBK showing more persistent high volatility periods (9.5 days) compared to HDFCBANK and ICICIBANK (~1 day).

2. **Liquidity Commonality**: Strong evidence (72% variance explained) that the three banks share common liquidity dynamics, suggesting systemic liquidity risk factors in the Indian banking sector.

3. **Delivery Percentage Effect**: Standardized delivery percentage is a highly significant predictor of next-day volatility (p < 0.000001), with negative coefficient suggesting higher delivery ratios correspond to lower future volatility.

4. **Tail Risk**: The 95th percentile of Parkinson variance (0.00248) indicates occasional extreme intraday price ranges across the banking sector.

---

**Analysis Date**: November 14, 2025
**Analyst**: Lead Quantitative Analyst
**Sample Period**: January 3, 2000 - April 30, 2021
