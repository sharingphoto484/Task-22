# GSADF Volatility & Dependence Analysis

## Overview
This analysis performs comprehensive quantitative financial analysis on three automotive stocks: TSLA (Tesla), TM (Toyota), and VWAGY (Volkswagen). The analysis includes bubble detection, volatility estimation, tail dependence measurement, liquidity assessment, and systemic risk transmission evaluation.

## Data
- **Tickers**: TSLA, TM, VWAGY
- **Common Sample Period**: 1,258 trading days after intersection
- **Sample Size After Differencing**: 1,257 observations
- **Data Columns**: Date, Open, High, Low, Close, Adj Close, Volume

## Methodology

### 1. GSADF Bubble Detection
- **Method**: Generalized Supremum Augmented Dickey-Fuller test (Phillips, Shi & Yu)
- **Initial Window**: 1% of sample (12 observations)
- **Critical Values**: Obtained from 2,000 Monte Carlo replications
- **Significance Level**: 5%
- **Datestamping**: BSADF algorithm with contiguous period merging

### 2. Yang-Zhang Volatility
- **Window**: 30 trading days (rolling)
- **Components**: Overnight, open-to-close, and Rogers-Satchell volatilities
- **Annualization**: Daily variance × 252, then square root
- **Aggregation**: Median across full sample per ticker, then equal-weighted average

### 3. Lower Tail Dependence
- **Window**: 252 trading days (rolling)
- **Threshold**: 5% quantile computed separately for each series
- **Measure**: Fraction of days both series below their thresholds
- **Aggregation**: Average over full sample across three pairs (TSLA-TM, TSLA-VWAGY, TM-VWAGY)

### 4. Amihud Illiquidity
- **Daily Measure**: |log return| / (Close × Volume)
- **Aggregation**: Averaged to calendar months
- **Standardization**: By in-sample mean and standard deviation per ticker
- **Reporting**: Average of last 12 months across tickers

### 5. Tail Dependence Heatmap
- **Period**: Last 24 month-ends
- **Window**: 252 trading days ending at each month-end
- **Display**: Heatmap showing dependence for each pair at each month-end

### 6. CoVaR Risk Transmission
- **Method**: Quantile regression at 5% level
- **Source**: TSLA
- **Targets**: TM and VWAGY
- **States**: TSLA at 5% quantile vs. median
- **Verdict**: 1 if average ΔCoVaR > 0 and both significant at 5%, else 0

## Results

### Key Findings

1. **Total Bubble Episodes**: **3**
   - TM: 1 episode (2020-03-16 to 2020-03-20)
   - VWAGY: 2 episodes
     - Episode 1: 2018-02-05 to 2018-02-09
     - Episode 2: 2020-03-16 to 2020-03-20
   - TSLA: 0 episodes detected

2. **Cross-Asset Average Median Yang-Zhang Volatility**: **31.62%**
   - TSLA: 49.40%
   - TM: 16.34%
   - VWAGY: 29.11%

3. **Average Lower Tail Dependence**: **0.0133**
   - TSLA-TM: 0.0098
   - TSLA-VWAGY: 0.0105
   - TM-VWAGY: 0.0196

4. **Twelve-Month Average Standardized Amihud**: **-0.848**
   - TSLA: -1.232
   - TM: -0.649
   - VWAGY: -0.664
   - *Negative values indicate below-average illiquidity (higher liquidity) in recent months*

5. **Maximum Heatmap Tail Dependence**: **0.0357**
   - Found in recent 24-month rolling windows

6. **CoVaR Risk Transmission Verdict**: **0**
   - No significant positive risk transmission from TSLA to TM/VWAGY at 5% level
   - ΔCoVaR values did not meet criteria for systemic risk transmission

## Interpretation

### Bubble Detection
- The GSADF test detected explosive behavior primarily during March 2020 (COVID-19 market crash period) for TM and VWAGY
- VWAGY also showed explosive behavior in early February 2018
- TSLA showed no statistically significant bubble episodes under the GSADF framework

### Volatility
- TSLA exhibits substantially higher volatility (49.4%) than traditional automakers
- TM shows the lowest volatility at 16.3%, consistent with an established manufacturer
- The cross-asset average of 31.6% reflects the diverse risk profiles

### Tail Dependence
- Very low average tail dependence (1.33%) suggests limited extreme co-movement
- TM-VWAGY pair shows highest tail dependence (1.96%), consistent with both being traditional automakers
- Electric vs. traditional auto manufacturers show lower extreme correlation

### Liquidity
- All three stocks show above-average liquidity in the most recent 12 months (negative standardized Amihud)
- TSLA shows the highest relative liquidity improvement
- Traditional automakers also demonstrate improved liquidity

### Systemic Risk
- No significant risk spillover from TSLA to traditional automakers at extreme quantiles
- The CoVaR analysis suggests these stocks operate with relatively independent tail risk

## Files Generated

- `volatility_dependence_analysis.py`: Complete analysis script
- `analysis_results.json`: Numerical results in JSON format
- `tail_dependence_heatmap.png`: Visual heatmap of rolling tail dependence
- `ANALYSIS_SUMMARY.md`: This documentation file

## Requirements

```
pandas>=2.0.0
numpy>=1.24.0
matplotlib>=3.7.0
seaborn>=0.12.0
scipy>=1.10.0
statsmodels>=0.14.0
```

## Execution

```bash
python volatility_dependence_analysis.py
```

The script will:
1. Load and intersect the three CSV files
2. Calculate log returns
3. Run GSADF tests with Monte Carlo simulations (~15-20 minutes)
4. Compute all volatility and dependence measures
5. Generate heatmap visualization
6. Save results to JSON

## Notes

- The analysis uses optimized GSADF implementation with windowing steps for computational efficiency
- All calculations strictly follow the common date intersection
- No imputation is performed; only rows lost to intersection and differencing are removed
- The script includes robust error handling for numerical edge cases
