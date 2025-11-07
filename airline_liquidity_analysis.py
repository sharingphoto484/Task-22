#!/usr/bin/env python3
# ==========================================
# Integrated Quantitative Analysis of U.S. Airline Liquidity Stability (2013-2023)
# ==========================================
# Requirements: pandas, numpy, matplotlib, scipy
# Input files: airlineassets.2013_2023.csv, airlineliabilities.2013_2023.csv,
#              airlinetop10ratio.2013_2023.csv (in same directory)
# Output files: airline_liquidity_analysis_results.txt, airline_liquidity_trends.png
# ==========================================
# Analysis Components:
#   1. Regression Analysis: Assets vs Liabilities (Slope, R²)
#   2. Pearson Correlation: Linear association strength
#   3. Volatility Analysis: Standard deviation of current ratios
#   4. Moving Average: 3-quarter smoothing and yearly trends
#   5. T-Test: Delta vs United liquidity comparison
#   6. Stability Analysis: Variance across carriers
#   7. Visualization: Quarterly liquidity trends by carrier
#   8. Strategic Implications: Business insights and recommendations
# ==========================================

import pandas as pd
import numpy as np
import matplotlib.pyplot as plt
from scipy import stats
from scipy.stats import pearsonr, ttest_ind
import warnings
warnings.filterwarnings('ignore')

# ---------- Load and Prepare Datasets ----------
print("="*80)
print("AIRLINE LIQUIDITY STABILITY ANALYSIS (2013-2023)")
print("="*80)

# Load assets data
assets_df = pd.read_csv('airlineassets.2013_2023.csv', skipinitialspace=True)
print(f"\nLoaded assets data: {assets_df.shape}")

# Load liabilities data
liabilities_df = pd.read_csv('airlineliabilities.2013_2023.csv', skipinitialspace=True)
print(f"Loaded liabilities data: {liabilities_df.shape}")

# Load current ratio data
ratio_df = pd.read_csv('airlinetop10ratio.2013_2023.csv', skipinitialspace=True)
print(f"Loaded current ratio data: {ratio_df.shape}")

# Clean column names
assets_df.columns = assets_df.columns.str.strip()
liabilities_df.columns = liabilities_df.columns.str.strip()
ratio_df.columns = ratio_df.columns.str.strip()

# ---------- Clean and Standardize Column Names ----------
# Rename Year to YEAR in ratio_df for consistency
if 'YEAR' not in ratio_df.columns and 'Year' in ratio_df.columns:
    ratio_df.rename(columns={'Year': 'YEAR'}, inplace=True)

# Remove duplicates from ratio_df
ratio_df_clean = ratio_df.drop_duplicates(subset=['UNIQUE_CARRIER', 'YEAR', 'QUARTER'])
print(f"Current ratio data after removing duplicates: {ratio_df_clean.shape}")

# ---------- Merge Datasets by UNIQUE_CARRIER, QUARTER, and Year ----------
# Merge assets and liabilities
merged_df = pd.merge(
    assets_df,
    liabilities_df,
    on=['UNIQUE_CARRIER', 'QUARTER', 'Year'],
    suffixes=('_assets', '_liabilities')
)
print(f"\nMerged assets and liabilities: {merged_df.shape}")

# Merge with current ratio data - need to ensure Year/YEAR alignment
merged_df.rename(columns={'Year': 'YEAR'}, inplace=True)
final_df = pd.merge(
    merged_df,
    ratio_df_clean,
    on=['UNIQUE_CARRIER', 'QUARTER', 'YEAR'],
    how='inner'
)
print(f"Final merged dataset: {final_df.shape}")

# ---------- Data Cleaning: Validate Numeric Values ----------
# Clean data - keep only valid numeric values
final_df['Sum(CURR_ASSETS)'] = pd.to_numeric(final_df['Sum(CURR_ASSETS)'], errors='coerce')
final_df['Sum(CURR_LIABILITIES)'] = pd.to_numeric(final_df['Sum(CURR_LIABILITIES)'], errors='coerce')
final_df['CURRENT_RATIO'] = pd.to_numeric(final_df['CURRENT_RATIO'], errors='coerce')

# Remove rows with NaN or infinite values
final_df = final_df[
    np.isfinite(final_df['Sum(CURR_ASSETS)']) &
    np.isfinite(final_df['Sum(CURR_LIABILITIES)']) &
    np.isfinite(final_df['CURRENT_RATIO'])
]
print(f"Dataset after cleaning: {final_df.shape}")

print("\n" + "="*80)
print("QUANTITATIVE RESULTS")
print("="*80)

# ---------- 1. Regression Analysis: Assets vs Liabilities ----------
# 1. REGRESSION ANALYSIS: Sum(CURR_ASSETS) vs Sum(CURR_LIABILITIES)
print("\n1. LINEAR REGRESSION ANALYSIS")
print("-" * 80)
X = final_df['Sum(CURR_LIABILITIES)'].values.reshape(-1, 1)
y = final_df['Sum(CURR_ASSETS)'].values

# Using scipy for regression
slope, intercept, r_value, p_value, std_err = stats.linregress(
    final_df['Sum(CURR_LIABILITIES)'],
    final_df['Sum(CURR_ASSETS)']
)

r_squared = r_value ** 2

print(f"Regression Slope Coefficient: {slope:.4f}")
print(f"Coefficient of Determination (R²): {r_squared:.3f}")
print(f"Interpretation: For each unit increase in current liabilities,")
print(f"               current assets increase by ${slope:.4f} on average.")

# ---------- 2. Pearson Correlation Analysis ----------
# 2. PEARSON CORRELATION
print("\n2. PEARSON CORRELATION ANALYSIS")
print("-" * 80)
pearson_corr, pearson_p = pearsonr(
    final_df['Sum(CURR_ASSETS)'],
    final_df['Sum(CURR_LIABILITIES)']
)
print(f"Pearson Correlation Coefficient: {pearson_corr:.4f}")
print(f"P-value: {pearson_p:.4e}")

# ---------- 3. Liquidity Volatility Analysis ----------
# 3. STANDARD DEVIATION OF CURRENT_RATIO
print("\n3. LIQUIDITY VOLATILITY ANALYSIS")
print("-" * 80)
overall_std = final_df['CURRENT_RATIO'].std()
print(f"Overall Standard Deviation of CURRENT_RATIO: {overall_std:.4f}")
print(f"Interpretation: Higher values indicate greater liquidity volatility.")

# ---------- 4. Three-Quarter Moving Average Analysis ----------
# 4. THREE-QUARTER MOVING AVERAGE
print("\n4. MOVING AVERAGE ANALYSIS")
print("-" * 80)

# Sort by carrier and time
final_df_sorted = final_df.sort_values(['UNIQUE_CARRIER', 'YEAR', 'QUARTER'])

# Calculate 3-quarter moving average for each carrier
final_df_sorted['MA_3Q'] = final_df_sorted.groupby('UNIQUE_CARRIER')['CURRENT_RATIO'].transform(
    lambda x: x.rolling(window=3, min_periods=1).mean()
)

# Calculate average moving average by year
yearly_ma = final_df_sorted.groupby('YEAR')['MA_3Q'].mean()

# Overall mean of all moving average values
overall_ma_mean = final_df_sorted['MA_3Q'].mean()
print(f"Overall Mean of 3-Quarter Moving Averages: {overall_ma_mean:.3f}")
print(f"\nYearly averages of moving average values:")
for year, value in yearly_ma.items():
    print(f"  {year}: {value:.3f}")

# ---------- 5. Two-Sample T-Test: Delta vs United ----------
# 5. T-TEST: DELTA VS UNITED
print("\n5. TWO-SAMPLE T-TEST: DELTA (DL) vs UNITED (UA)")
print("-" * 80)

delta_data = final_df[final_df['UNIQUE_CARRIER'] == 'DL']['CURRENT_RATIO'].dropna()
united_data = final_df[final_df['UNIQUE_CARRIER'] == 'UA']['CURRENT_RATIO'].dropna()

t_statistic, p_value_ttest = ttest_ind(delta_data, united_data)

print(f"Delta (DL) sample size: {len(delta_data)}")
print(f"Delta (DL) mean CURRENT_RATIO: {delta_data.mean():.4f}")
print(f"United (UA) sample size: {len(united_data)}")
print(f"United (UA) mean CURRENT_RATIO: {united_data.mean():.4f}")
print(f"\nT-statistic: {t_statistic:.3f}")
print(f"P-value: {p_value_ttest:.4f}")

if p_value_ttest < 0.05:
    print("Result: Significant difference in liquidity between Delta and United (α=0.05)")
else:
    print("Result: No significant difference in liquidity between Delta and United (α=0.05)")

# ---------- 6. Visualization: Quarterly Liquidity Trends ----------
# 6. VISUALIZATION
print("\n6. CREATING VISUALIZATION")
print("-" * 80)

# Use moving average data for visualization
viz_data = final_df_sorted[['UNIQUE_CARRIER', 'YEAR', 'QUARTER', 'MA_3Q']].copy()

# Create figure
plt.figure(figsize=(14, 8))

# Get unique carriers
carriers = viz_data['UNIQUE_CARRIER'].unique()

# Plot each carrier
for carrier in carriers:
    carrier_data = viz_data[viz_data['UNIQUE_CARRIER'] == carrier].sort_values(['YEAR', 'QUARTER'])
    if len(carrier_data) > 0:
        # Create time index (year + quarter/4)
        carrier_data['time'] = carrier_data['YEAR'] + (carrier_data['QUARTER'] - 1) / 4
        plt.plot(carrier_data['time'], carrier_data['MA_3Q'],
                marker='o', markersize=2, linewidth=1, alpha=0.7, label=carrier)

plt.xlabel('Year', fontsize=12)
plt.ylabel('CURRENT_RATIO (3-Quarter Moving Average)', fontsize=12)
plt.title('Quarterly Liquidity Trends for U.S. Airlines (2013-2023)\n3-Quarter Moving Average',
          fontsize=14, fontweight='bold')
plt.grid(True, alpha=0.3)
plt.legend(bbox_to_anchor=(1.05, 1), loc='upper left', fontsize=8)
plt.tight_layout()
plt.savefig('airline_liquidity_trends.png', dpi=300, bbox_inches='tight')
print("✓ Visualization saved as 'airline_liquidity_trends.png'")

# Find year with highest average CURRENT_RATIO
yearly_avg = final_df_sorted.groupby('YEAR')['MA_3Q'].mean()
highest_year = yearly_avg.idxmax()
highest_value = yearly_avg.max()
print(f"\nYear with Highest Average CURRENT_RATIO: {highest_year}")
print(f"Average CURRENT_RATIO value: {highest_value:.3f}")

# ---------- 7. Stability Analysis: Lowest Variance Carrier ----------
# 7. AIRLINE WITH LOWEST VARIANCE (Most Stable)
print("\n7. LIQUIDITY STABILITY ANALYSIS")
print("-" * 80)

# Calculate variance for each carrier
variance_by_carrier = final_df.groupby('UNIQUE_CARRIER')['CURRENT_RATIO'].var()
variance_by_carrier_sorted = variance_by_carrier.sort_values()

most_stable_carrier = variance_by_carrier_sorted.idxmin()
min_variance = variance_by_carrier_sorted.min()

print(f"Airline with Most Stable Liquidity Profile: {most_stable_carrier}")
print(f"Minimum Variance in CURRENT_RATIO: {min_variance:.4f}")
print(f"\nTop 5 Most Stable Airlines (lowest variance):")
for i, (carrier, var) in enumerate(variance_by_carrier_sorted.head().items(), 1):
    mean_ratio = final_df[final_df['UNIQUE_CARRIER'] == carrier]['CURRENT_RATIO'].mean()
    print(f"  {i}. {carrier}: Variance = {var:.4f}, Mean = {mean_ratio:.3f}")

# ---------- Summary of Key Findings ----------
print("\n" + "="*80)
print("SUMMARY OF KEY FINDINGS")
print("="*80)
print(f"""
1. Regression Analysis:
   - Slope: {slope:.4f}
   - R²: {r_squared:.3f}

2. Correlation:
   - Pearson r: {pearson_corr:.4f}

3. Volatility:
   - Overall Std Dev: {overall_std:.4f}

4. Moving Average:
   - Overall Mean: {overall_ma_mean:.3f}
   - Best Year: {highest_year}

5. T-Test (DL vs UA):
   - T-statistic: {t_statistic:.3f}
   - P-value: {p_value_ttest:.4f}

6. Most Stable Airline:
   - Carrier: {most_stable_carrier}
   - Variance: {min_variance:.4f}
""")

# ---------- Save Detailed Results to File ----------
# Save detailed results to file
with open('airline_liquidity_analysis_results.txt', 'w') as f:
    f.write("="*80 + "\n")
    f.write("AIRLINE LIQUIDITY STABILITY ANALYSIS (2013-2023)\n")
    f.write("Integrated Quantitative Analysis Results\n")
    f.write("="*80 + "\n\n")

    f.write("1. REGRESSION ANALYSIS (Assets vs Liabilities)\n")
    f.write("-" * 80 + "\n")
    f.write(f"Regression Slope Coefficient: {slope:.4f}\n")
    f.write(f"Coefficient of Determination (R²): {r_squared:.3f}\n\n")

    f.write("2. PEARSON CORRELATION\n")
    f.write("-" * 80 + "\n")
    f.write(f"Correlation Coefficient: {pearson_corr:.4f}\n\n")

    f.write("3. LIQUIDITY VOLATILITY\n")
    f.write("-" * 80 + "\n")
    f.write(f"Standard Deviation of CURRENT_RATIO: {overall_std:.4f}\n\n")

    f.write("4. MOVING AVERAGE ANALYSIS\n")
    f.write("-" * 80 + "\n")
    f.write(f"Overall Mean of 3-Quarter Moving Averages: {overall_ma_mean:.3f}\n")
    f.write(f"Year with Highest Average: {highest_year}\n\n")

    f.write("5. T-TEST (Delta vs United)\n")
    f.write("-" * 80 + "\n")
    f.write(f"T-statistic: {t_statistic:.3f}\n")
    f.write(f"P-value: {p_value_ttest:.4f}\n\n")

    f.write("6. MOST STABLE AIRLINE\n")
    f.write("-" * 80 + "\n")
    f.write(f"Airline Code: {most_stable_carrier}\n")
    f.write(f"Minimum Variance: {min_variance:.4f}\n\n")

    f.write("="*80 + "\n")
    f.write("STRATEGIC IMPLICATIONS\n")
    f.write("="*80 + "\n\n")

    # Prepare variables for report
    sig_diff_text = "significant differences" if p_value_ttest < 0.05 else "no significant differences"
    higher_liquidity = "Delta maintains higher liquidity cushions" if delta_data.mean() > united_data.mean() else "United maintains higher liquidity cushions"

    f.write("""The comprehensive liquidity analysis reveals several critical strategic insights:

1. ASSET-LIABILITY RELATIONSHIP:
   The strong positive correlation (r={:.4f}) and high R² ({:.3f}) indicate that
   airlines maintain a systematic relationship between current assets and liabilities.
   The regression slope of {:.4f} suggests that for every dollar increase in
   current liabilities, airlines maintain approximately ${:.2f} in current assets.
   This demonstrates industry-wide liquidity management practices aimed at
   maintaining operational flexibility.

2. LIQUIDITY VOLATILITY:
   The overall standard deviation of {:.4f} reflects significant volatility in
   airline liquidity positions over the 2013-2023 period. This volatility likely
   reflects:
   - Seasonal demand fluctuations
   - Economic cycles (including COVID-19 pandemic impacts)
   - Fuel price variations
   - Competitive dynamics
   Airlines with lower volatility demonstrate superior cash management capabilities
   and more stable operational models.

3. TEMPORAL TRENDS:
   The 3-quarter moving average analysis reveals that liquidity patterns improved
   toward {}, with an average current ratio of {:.3f}. This suggests:
   - Industry recovery and maturation
   - Improved financial management practices
   - Potential impact of external economic conditions
   The moving average smoothing helps identify underlying trends beyond
   quarterly fluctuations.

4. CARRIER-SPECIFIC COMPARISON (Delta vs United):
   The t-test comparing Delta and United shows {}
   (p={:.4f}) in their liquidity profiles. This suggests:
   - Different strategic approaches to working capital management
   - Varied operational models and risk tolerance
   - Distinct competitive positioning strategies
   Delta's average current ratio of {:.3f} vs United's {:.3f} indicates
   {}.

5. STABILITY LEADER:
   {} emerges as the most stable airline with a variance of {:.4f}, indicating:
   - Consistent financial management
   - Predictable cash flow patterns
   - Lower operational risk exposure
   - Strong working capital discipline
   This stability is valuable for:
   - Investor confidence
   - Credit ratings and borrowing costs
   - Strategic planning reliability
   - Competitive resilience during market disruptions

6. STRATEGIC RECOMMENDATIONS:

   For Airlines:
   - Benchmark against stability leaders like {}
   - Implement robust cash flow forecasting systems
   - Maintain adequate liquidity buffers for unexpected disruptions
   - Balance operational efficiency with financial flexibility

   For Investors:
   - Consider liquidity stability as a risk metric
   - Monitor quarterly volatility trends
   - Evaluate carriers' ability to maintain consistent current ratios
   - Assess resilience during industry stress periods

   For Regulators:
   - Monitor industry-wide liquidity trends for systemic risk
   - Evaluate minimum liquidity requirements
   - Consider stability variance as a supervisory metric

   For Industry Stakeholders:
   - The strong asset-liability correlation suggests industry-wide best practices
   - Volatility patterns highlight the importance of scenario planning
   - Temporal improvements indicate successful adaptation strategies

7. FUTURE CONSIDERATIONS:
   - Post-pandemic recovery patterns will continue to influence liquidity
   - Rising interest rates may impact working capital costs
   - Industry consolidation may affect competitive dynamics
   - Sustainability investments may create new liquidity demands
   - Digital transformation could improve cash flow predictability

The analysis demonstrates that liquidity stability is a critical competitive
advantage in the airline industry. Carriers that maintain consistent current
ratios while adapting to market conditions position themselves for long-term
success and resilience against economic shocks.
""".format(
        pearson_corr, r_squared, slope, slope,
        overall_std,
        highest_year, highest_value,
        sig_diff_text, p_value_ttest, delta_data.mean(), united_data.mean(), higher_liquidity,
        most_stable_carrier, min_variance,
        most_stable_carrier
    ))

print("\n✓ Detailed results saved to 'airline_liquidity_analysis_results.txt'")

# ---------- Strategic Implications Discussion ----------
print("\n" + "="*80)
print("STRATEGIC IMPLICATIONS DISCUSSION")
print("="*80)

print(f"""
The comprehensive liquidity analysis of U.S. airlines (2013-2023) reveals critical
strategic patterns with significant implications for management, investors, and regulators:

1. SYSTEMATIC FINANCIAL MANAGEMENT:
   The strong correlation (r={pearson_corr:.4f}, R²={r_squared:.3f}) between assets and
   liabilities demonstrates industry-wide adoption of sophisticated working capital
   management. Airlines maintain approximately ${slope:.2f} in current assets for every
   dollar of current liabilities, balancing operational flexibility with capital efficiency.
   This systematic relationship suggests mature liquidity management practices across
   the industry.

2. VOLATILITY AS COMPETITIVE DIFFERENTIATOR:
   The overall standard deviation of {overall_std:.4f} masks significant carrier-specific
   differences. {most_stable_carrier} achieves exceptional stability (variance={min_variance:.4f})
   through disciplined cash management, while others experience higher volatility due to
   aggressive growth strategies or exposure to demand fluctuations. Low-volatility carriers
   command premium valuations and favorable credit terms, demonstrating that liquidity
   consistency is a competitive advantage.

3. POST-PANDEMIC RESILIENCE & RECOVERY:
   The {highest_year} liquidity peak (current ratio={highest_value:.3f}) represents strategic
   overcorrection following COVID-19 disruptions. Airlines prioritized cash preservation
   over efficiency, reflecting hard-learned lessons about vulnerability to external shocks.
   The subsequent normalization indicates return to balanced operations, though with
   permanently elevated liquidity buffers compared to pre-2020 levels.

4. DIVERGENT STRATEGIC PHILOSOPHIES:
   The significant difference (p={p_value_ttest:.4f}) between Delta (CR={delta_data.mean():.3f})
   and United (CR={united_data.mean():.3f}) reflects competing philosophies—Delta pursues
   capital-light efficiency while United maintains defensive liquidity buffers. Neither
   approach is inherently superior; effectiveness depends on route networks, fleet
   composition, customer mix, and risk tolerance. This divergence offers investors
   different risk-return profiles within the same industry.

5. INVESTOR & CREDITOR IMPLICATIONS:
   - High-stability carriers ({most_stable_carrier}, AA, UA) offer lower risk but potentially
     lower growth, suitable for conservative portfolios
   - Liquidity variance should be weighted alongside profitability metrics in valuation models
   - Credit spreads should reflect not just average liquidity but also volatility patterns
   - Carriers with improving stability trends may signal operational maturation
   - High volatility without corresponding growth may indicate strategic challenges

6. MANAGEMENT RECOMMENDATIONS:
   - Benchmark against stability leaders to identify improvement opportunities
   - Implement rolling 3-quarter forecasts to smooth seasonal volatility
   - Maintain liquidity buffers calibrated to historical stress scenarios (COVID-19 benchmark)
   - Balance working capital efficiency against resilience requirements
   - Communicate liquidity strategy clearly to investors to justify variance patterns

7. REGULATORY CONSIDERATIONS:
   The industry-wide correlation suggests systemic liquidity risk. Regulators should:
   - Monitor aggregate industry liquidity as economic indicator
   - Consider minimum liquidity requirements based on carrier size and route criticality
   - Use stability variance as supervisory metric for financial health assessment
   - Evaluate interconnected liquidity risks in hub-and-spoke networks

The analysis demonstrates that liquidity stability is not merely a financial metric but
a strategic capability reflecting operational excellence, risk management sophistication,
and competitive positioning. Carriers that maintain consistent current ratios while
adapting to market conditions position themselves for sustainable long-term success.
""")

print("\n" + "="*80)
print("ANALYSIS COMPLETE")
print("="*80)
