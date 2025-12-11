# ==========================================
# Integrated Customer & Revenue Analysis Script
# ==========================================
# Requirements: pandas, numpy, matplotlib, scipy, openpyxl
# Input files: Customer.xlsx, Track.xlsx, Invoice.xlsx (in same directory)
# Output files: cv_by_country.png, track_duration_boxplot.png,
#               weighted_avg_scatter.png, revenue_growth_line.png,
#               zscore_histogram.png
# ==========================================

import pandas as pd
import numpy as np
import matplotlib.pyplot as plt
from scipy import stats

# ---------- Load Excel Files ----------
print("Loading data files...")
customer_df = pd.read_excel('Customer.xlsx')
track_df = pd.read_excel('Track.xlsx')
invoice_df = pd.read_excel('Invoice.xlsx')

print(f"Loaded {len(customer_df)} customers, {len(track_df)} tracks, {len(invoice_df)} invoices")

# ==========================================
# ANALYSIS 1: Coefficient of Variation by Country
# ==========================================
# Strategic Insight: For countries with high coefficient of variation (>40%),
# implement tiered pricing and personalized engagement strategies to capture both
# high-value and price-sensitive customer segments effectively.

# ---------- Merge Invoice with Customer Data ----------
print("\n=== ANALYSIS 1: Coefficient of Variation by Country ===")
invoice_customer = invoice_df.merge(customer_df[['CustomerId', 'Country']], on='CustomerId', how='left')

# ---------- Group by Country and Calculate Statistics ----------
country_stats = invoice_customer.groupby('Country')['Total'].agg(['mean', 'std', 'count']).reset_index()
country_stats.columns = ['Country', 'Mean', 'StdDev', 'Count']

# ---------- Filter Countries with >= 5 Invoices ----------
country_stats_filtered = country_stats[country_stats['Count'] >= 5].copy()

# ---------- Calculate Coefficient of Variation ----------
country_stats_filtered['CV'] = (country_stats_filtered['StdDev'] / country_stats_filtered['Mean']) * 100

# ---------- Identify Maximum CV Country ----------
max_cv_country = country_stats_filtered.loc[country_stats_filtered['CV'].idxmax()]
print(f"\nCountry with Maximum CV: {max_cv_country['Country']}")
print(f"Coefficient of Variation: {max_cv_country['CV']:.2f}%")

# ---------- Create Horizontal Bar Chart ----------
cv_sorted = country_stats_filtered.sort_values('CV', ascending=True)
plt.figure(figsize=(10, 8))
plt.barh(cv_sorted['Country'], cv_sorted['CV'], color='steelblue')
plt.xlabel('Coefficient of Variation (%)')
plt.ylabel('Country')
plt.title('Coefficient of Variation in Customer Spending by Country\n(Countries with ≥5 Invoices)')
plt.grid(axis='x', alpha=0.3)
plt.tight_layout()
plt.savefig('cv_by_country.png', dpi=300, bbox_inches='tight')
plt.close()
print("Chart saved: cv_by_country.png")

# ==========================================
# ANALYSIS 2: Interquartile Range for Track Duration
# ==========================================

# ---------- Extract Track Duration in Milliseconds ----------
print("\n=== ANALYSIS 2: Interquartile Range for Track Duration ===")
track_durations = track_df['Milliseconds'].dropna()

# ---------- Calculate Quartiles using Linear Interpolation ----------
q1 = track_durations.quantile(0.25)
q3 = track_durations.quantile(0.75)

# ---------- Calculate IQR ----------
iqr = q3 - q1
print(f"\nFirst Quartile (Q1): {q1:.0f} ms")
print(f"Third Quartile (Q3): {q3:.0f} ms")
print(f"Interquartile Range (IQR): {iqr:.0f} ms")

# ---------- Create Box Plot ----------
plt.figure(figsize=(8, 6))
plt.boxplot(track_durations, vert=True, patch_artist=True,
            boxprops=dict(facecolor='lightblue'),
            medianprops=dict(color='red', linewidth=2),
            flierprops=dict(marker='o', markerfacecolor='gray', markersize=3, alpha=0.3))
plt.ylabel('Track Duration (milliseconds)')
plt.title('Distribution of Track Durations')
plt.grid(axis='y', alpha=0.3)
plt.tight_layout()
plt.savefig('track_duration_boxplot.png', dpi=300, bbox_inches='tight')
plt.close()
print("Chart saved: track_duration_boxplot.png")

# ==========================================
# ANALYSIS 3: Weighted Average Invoice Value
# ==========================================

# ---------- Aggregate by Customer ----------
print("\n=== ANALYSIS 3: Weighted Average Invoice Value ===")
customer_metrics = invoice_df.groupby('CustomerId')['Total'].agg(
    transaction_frequency='count',
    avg_invoice_value='mean'
).reset_index()

# ---------- Calculate Weighted Contribution ----------
customer_metrics['weighted_contribution'] = (
    customer_metrics['avg_invoice_value'] * customer_metrics['transaction_frequency']
)

# ---------- Calculate Overall Weighted Average ----------
total_weighted_sum = customer_metrics['weighted_contribution'].sum()
total_frequency = customer_metrics['transaction_frequency'].sum()
weighted_avg_invoice = total_weighted_sum / total_frequency

print(f"\nWeighted Average Invoice Value: ${weighted_avg_invoice:.2f}")

# ---------- Create Scatter Plot ----------
plt.figure(figsize=(10, 6))
scatter = plt.scatter(
    customer_metrics['transaction_frequency'],
    customer_metrics['avg_invoice_value'],
    s=customer_metrics['weighted_contribution'] * 2,
    alpha=0.5,
    c=customer_metrics['weighted_contribution'],
    cmap='viridis'
)
plt.xlabel('Transaction Frequency (Number of Invoices)')
plt.ylabel('Average Invoice Value ($)')
plt.title('Customer Transaction Patterns with Weighted Contributions')
plt.colorbar(scatter, label='Weighted Contribution')
plt.grid(alpha=0.3)
plt.tight_layout()
plt.savefig('weighted_avg_scatter.png', dpi=300, bbox_inches='tight')
plt.close()
print("Chart saved: weighted_avg_scatter.png")

# ==========================================
# ANALYSIS 4: Year-over-Year Revenue Growth (2012-2013)
# ==========================================

# ---------- Extract Year from Invoice Date ----------
print("\n=== ANALYSIS 4: Year-over-Year Revenue Growth ===")
invoice_df['Year'] = pd.to_datetime(invoice_df['InvoiceDate']).dt.year

# ---------- Calculate Annual Revenue ----------
annual_revenue = invoice_df.groupby('Year')['Total'].sum().reset_index()
annual_revenue.columns = ['Year', 'Revenue']

# ---------- Calculate Growth Rate 2012-2013 ----------
revenue_2012 = annual_revenue[annual_revenue['Year'] == 2012]['Revenue'].values[0]
revenue_2013 = annual_revenue[annual_revenue['Year'] == 2013]['Revenue'].values[0]
growth_rate = ((revenue_2013 - revenue_2012) / revenue_2012) * 100

print(f"\n2012 Revenue: ${revenue_2012:.2f}")
print(f"2013 Revenue: ${revenue_2013:.2f}")
print(f"Year-over-Year Growth Rate (2012-2013): {growth_rate:.2f}%")

# ---------- Create Line Chart ----------
plt.figure(figsize=(10, 6))
plt.plot(annual_revenue['Year'], annual_revenue['Revenue'], marker='o', linewidth=2, markersize=8)

# Highlight 2012-2013 segment
years_2012_2013 = annual_revenue[annual_revenue['Year'].isin([2012, 2013])]
plt.plot(years_2012_2013['Year'], years_2012_2013['Revenue'],
         marker='o', linewidth=3, markersize=10, color='red', label='2012-2013 Growth')

plt.xlabel('Year')
plt.ylabel('Total Revenue ($)')
plt.title('Annual Revenue Trend (2009-2013)')
plt.grid(alpha=0.3)
plt.legend()
plt.tight_layout()
plt.savefig('revenue_growth_line.png', dpi=300, bbox_inches='tight')
plt.close()
print("Chart saved: revenue_growth_line.png")

# ==========================================
# ANALYSIS 5: Z-Score Standardization for File Size
# ==========================================

# ---------- Extract File Size in Bytes ----------
print("\n=== ANALYSIS 5: Z-Score Analysis for File Size ===")
file_sizes = track_df['Bytes'].dropna()

# ---------- Calculate Population Statistics ----------
mean_bytes = file_sizes.mean()
std_bytes = file_sizes.std(ddof=0)  # Population standard deviation

# ---------- Calculate Z-Scores ----------
track_df['ZScore_Bytes'] = (track_df['Bytes'] - mean_bytes) / std_bytes

# ---------- Count Extreme Outliers ----------
extreme_outliers = track_df[track_df['ZScore_Bytes'].abs() > 3]
outlier_count = len(extreme_outliers)

print(f"\nMean File Size: {mean_bytes:.2f} bytes")
print(f"Std Dev File Size: {std_bytes:.2f} bytes")
print(f"Number of Extreme Outliers (|Z| > 3): {outlier_count}")

# ---------- Create Histogram ----------
plt.figure(figsize=(10, 6))
plt.hist(track_df['ZScore_Bytes'].dropna(), bins=50, color='skyblue', edgecolor='black', alpha=0.7)
plt.axvline(-3, color='red', linestyle='--', linewidth=2, label='Z = -3 threshold')
plt.axvline(3, color='red', linestyle='--', linewidth=2, label='Z = +3 threshold')
plt.xlabel('Z-Score')
plt.ylabel('Frequency (Number of Tracks)')
plt.title('Distribution of File Size Z-Scores with Outlier Thresholds')
plt.xlim(-4, 4)
plt.legend()
plt.grid(alpha=0.3)
plt.tight_layout()
plt.savefig('zscore_histogram.png', dpi=300, bbox_inches='tight')
plt.close()
print("Chart saved: zscore_histogram.png")

# ==========================================
# ANALYSIS 6: Month with Minimum Revenue
# ==========================================

# ---------- Extract Month from Invoice Date ----------
print("\n=== ANALYSIS 6: Month with Minimum Revenue ===")
invoice_df['Month'] = pd.to_datetime(invoice_df['InvoiceDate']).dt.month

# ---------- Aggregate Revenue by Month ----------
monthly_revenue = invoice_df.groupby('Month')['Total'].sum().reset_index()
monthly_revenue.columns = ['Month', 'Revenue']

# ---------- Identify Minimum Revenue Month ----------
min_revenue_month = monthly_revenue.loc[monthly_revenue['Revenue'].idxmin(), 'Month']
min_revenue_value = monthly_revenue.loc[monthly_revenue['Revenue'].idxmin(), 'Revenue']

print(f"\nMonth with Minimum Revenue: {min_revenue_month}")
print(f"Revenue for Month {min_revenue_month}: ${min_revenue_value:.2f}")

# ==========================================
# ANALYSIS 7: Support Rep Serving Most Countries
# ==========================================

# ---------- Count Unique Countries per Support Rep ----------
print("\n=== ANALYSIS 7: Support Rep Serving Most Countries ===")
rep_country_counts = customer_df.groupby('SupportRepId')['Country'].nunique().reset_index()
rep_country_counts.columns = ['SupportRepId', 'UniqueCountries']

# ---------- Identify Support Rep with Maximum Countries ----------
max_rep = rep_country_counts.loc[rep_country_counts['UniqueCountries'].idxmax()]

print(f"\nSupport Rep ID Serving Most Countries: {int(max_rep['SupportRepId'])}")
print(f"Number of Unique Countries Served: {int(max_rep['UniqueCountries'])}")

# ==========================================
# ANALYSIS 8: Percentage of Premium-Priced Tracks
# ==========================================

# ---------- Count Total and Premium Tracks ----------
print("\n=== ANALYSIS 8: Premium Pricing Analysis ===")
total_tracks = len(track_df)
premium_tracks = len(track_df[track_df['UnitPrice'] == 1.99])

# ---------- Calculate Percentage ----------
premium_percentage = (premium_tracks / total_tracks) * 100

print(f"\nTotal Tracks: {total_tracks}")
print(f"Premium-Priced Tracks ($1.99): {premium_tracks}")
print(f"Percentage of Premium Tracks: {premium_percentage:.2f}%")

# ==========================================
# SUMMARY OF KEY OUTPUTS
# ==========================================
print("\n" + "="*60)
print("SUMMARY OF KEY OUTPUTS")
print("="*60)
print(f"1. Maximum CV Country: {max_cv_country['Country']} - CV: {max_cv_country['CV']:.2f}%")
print(f"2. Track Duration IQR: {iqr:.0f} ms")
print(f"3. Weighted Average Invoice Value: ${weighted_avg_invoice:.2f}")
print(f"4. YoY Growth Rate (2012-2013): {growth_rate:.2f}%")
print(f"5. Extreme Outliers in File Size: {outlier_count}")
print(f"6. Month with Minimum Revenue: {min_revenue_month}")
print(f"7. Support Rep Serving Most Countries: {int(max_rep['SupportRepId'])}")
print(f"8. Premium-Priced Tracks Percentage: {premium_percentage:.2f}%")
print("="*60)
print("\nAll visualizations saved successfully!")
