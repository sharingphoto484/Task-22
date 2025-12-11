# ==========================================
# Delivery Logistics Analysis Script
# ==========================================
# Requirements: pandas, numpy, matplotlib, scipy, scikit-learn
# Input files: updated.xlsx, encoded_cleaned_test.xlsx, cleaned_test.xlsx (in same directory)
# Output files: cv_traffic_density.png, delivery_time_boxplot.png,
#               weighted_avg_scatter.png, rating_vs_time_scatter.png,
#               zscore_histogram.png
# ==========================================

import pandas as pd
import numpy as np
import matplotlib.pyplot as plt
from scipy import stats
import warnings
warnings.filterwarnings('ignore')

# ---------- Load Data ----------
print("Loading data...")
df = pd.read_excel('updated.xlsx')
print(f"Loaded {len(df)} delivery records from updated.xlsx")
print()

# ==========================================
# Analysis 1: Coefficient of Variation by Road Traffic Density
# ==========================================
print("=" * 60)
print("ANALYSIS 1: Coefficient of Variation by Road Traffic Density")
print("=" * 60)

# ---------- Group by Road Traffic Density ----------
cv_analysis = df.groupby('Road_traffic_density')['Time_taken(min)'].agg([
    ('count', 'count'),
    ('mean', 'mean'),
    ('std', 'std')
]).reset_index()

# ---------- Filter Groups with >= 100 Deliveries ----------
cv_analysis = cv_analysis[cv_analysis['count'] >= 100].copy()

# ---------- Calculate Coefficient of Variation ----------
cv_analysis['cv_percentage'] = (cv_analysis['std'] / cv_analysis['mean']) * 100

# ---------- Find Traffic Density with Maximum CV ----------
max_cv_row = cv_analysis.loc[cv_analysis['cv_percentage'].idxmax()]
max_cv_density = max_cv_row['Road_traffic_density']
max_cv_value = max_cv_row['cv_percentage']

print(f"Traffic Density Level with Maximum CV: {int(max_cv_density)}")
print(f"Maximum Coefficient of Variation: {max_cv_value:.2f}%")
print()

# ---------- Generate Horizontal Bar Chart ----------
cv_sorted = cv_analysis.sort_values('cv_percentage', ascending=True)
plt.figure(figsize=(10, 6))
plt.barh(cv_sorted['Road_traffic_density'].astype(str), cv_sorted['cv_percentage'], color='steelblue')
plt.xlabel('Coefficient of Variation (%)', fontsize=12)
plt.ylabel('Road Traffic Density Level', fontsize=12)
plt.title('Coefficient of Variation by Road Traffic Density\n(Groups with ≥100 Deliveries)', fontsize=14, fontweight='bold')
plt.grid(axis='x', alpha=0.3)
plt.tight_layout()
plt.savefig('cv_traffic_density.png', dpi=300, bbox_inches='tight')
plt.close()
print("Chart saved: cv_traffic_density.png")
print()

# ==========================================
# Analysis 2: Interquartile Range of Delivery Times
# ==========================================
print("=" * 60)
print("ANALYSIS 2: Interquartile Range of Delivery Times")
print("=" * 60)

# ---------- Calculate Q1 and Q3 using Linear Interpolation ----------
delivery_times = df['Time_taken(min)'].values
q1 = np.percentile(delivery_times, 25, method='linear')
q3 = np.percentile(delivery_times, 75, method='linear')

# ---------- Calculate IQR ----------
iqr = q3 - q1

print(f"First Quartile (Q1): {q1:.2f} minutes")
print(f"Third Quartile (Q3): {q3:.2f} minutes")
print(f"Interquartile Range (IQR): {iqr:.2f} minutes")
print()

# ---------- Generate Box Plot ----------
plt.figure(figsize=(8, 10))
plt.boxplot(delivery_times, vert=True, patch_artist=True,
            boxprops=dict(facecolor='lightblue', color='steelblue', linewidth=2),
            whiskerprops=dict(color='steelblue', linewidth=1.5),
            capprops=dict(color='steelblue', linewidth=1.5),
            medianprops=dict(color='red', linewidth=2),
            flierprops=dict(marker='o', markerfacecolor='orange', markersize=5, alpha=0.5))
plt.ylabel('Delivery Time (minutes)', fontsize=12)
plt.title('Distribution of Delivery Times\nBox Plot Analysis', fontsize=14, fontweight='bold')
plt.grid(axis='y', alpha=0.3)
plt.tight_layout()
plt.savefig('delivery_time_boxplot.png', dpi=300, bbox_inches='tight')
plt.close()
print("Chart saved: delivery_time_boxplot.png")
print()

# ==========================================
# Analysis 3: Weighted Average Delivery Time by Person
# ==========================================
print("=" * 60)
print("ANALYSIS 3: Weighted Average Delivery Time by Person")
print("=" * 60)

# ---------- Aggregate by Delivery Person ----------
person_stats = df.groupby('Delivery_person_ID')['Time_taken(min)'].agg([
    ('frequency', 'count'),
    ('avg_time', 'mean')
]).reset_index()

# ---------- Calculate Weighted Contributions ----------
person_stats['weighted_contribution'] = person_stats['avg_time'] * person_stats['frequency']

# ---------- Calculate Overall Weighted Average ----------
total_weighted_sum = person_stats['weighted_contribution'].sum()
total_frequency = person_stats['frequency'].sum()
weighted_avg = total_weighted_sum / total_frequency

print(f"Total Delivery Persons: {len(person_stats)}")
print(f"Total Deliveries: {total_frequency}")
print(f"Weighted Average Delivery Time: {weighted_avg:.2f} minutes")
print()

# ---------- Generate Scatter Plot ----------
plt.figure(figsize=(12, 8))
scatter = plt.scatter(person_stats['frequency'], person_stats['avg_time'],
                     s=person_stats['weighted_contribution']/10,
                     c=person_stats['weighted_contribution'],
                     cmap='viridis', alpha=0.6, edgecolors='black', linewidth=0.5)
plt.xlabel('Delivery Frequency (Number of Deliveries)', fontsize=12)
plt.ylabel('Average Delivery Time (minutes)', fontsize=12)
plt.title('Delivery Person Performance Analysis\n(Point size proportional to weighted contribution)',
          fontsize=14, fontweight='bold')
plt.colorbar(scatter, label='Weighted Contribution')
plt.grid(alpha=0.3)
plt.tight_layout()
plt.savefig('weighted_avg_scatter.png', dpi=300, bbox_inches='tight')
plt.close()
print("Chart saved: weighted_avg_scatter.png")
print()

# ==========================================
# Analysis 4: Pearson Correlation - Ratings vs Delivery Time
# ==========================================
print("=" * 60)
print("ANALYSIS 4: Pearson Correlation - Ratings vs Delivery Time")
print("=" * 60)

# ---------- Remove Missing Ratings ----------
df_ratings = df[['Delivery_person_Ratings', 'Time_taken(min)']].dropna()

# ---------- Calculate Pearson Correlation ----------
pearson_corr, p_value = stats.pearsonr(df_ratings['Delivery_person_Ratings'],
                                        df_ratings['Time_taken(min)'])

print(f"Sample Size (after removing missing ratings): {len(df_ratings)}")
print(f"Pearson Correlation Coefficient: {pearson_corr:.4f}")
print(f"P-value: {p_value:.6f}")
print()

# ---------- Generate Scatter Plot with Regression Line ----------
plt.figure(figsize=(12, 8))
plt.scatter(df_ratings['Delivery_person_Ratings'], df_ratings['Time_taken(min)'],
           alpha=0.5, s=30, c='steelblue', edgecolors='black', linewidth=0.3)

# Fit regression line
z = np.polyfit(df_ratings['Delivery_person_Ratings'], df_ratings['Time_taken(min)'], 1)
p = np.poly1d(z)
x_line = np.linspace(df_ratings['Delivery_person_Ratings'].min(),
                     df_ratings['Delivery_person_Ratings'].max(), 100)
plt.plot(x_line, p(x_line), "r-", linewidth=2, label=f'Trend Line (r={pearson_corr:.4f})')

plt.xlabel('Delivery Person Ratings', fontsize=12)
plt.ylabel('Delivery Time (minutes)', fontsize=12)
plt.title('Relationship Between Delivery Person Ratings and Delivery Time',
          fontsize=14, fontweight='bold')
plt.legend(fontsize=10)
plt.grid(alpha=0.3)
plt.tight_layout()
plt.savefig('rating_vs_time_scatter.png', dpi=300, bbox_inches='tight')
plt.close()
print("Chart saved: rating_vs_time_scatter.png")
print()

# ==========================================
# Analysis 5: Z-Score Standardization and Outlier Detection
# ==========================================
print("=" * 60)
print("ANALYSIS 5: Z-Score Standardization and Outlier Detection")
print("=" * 60)

# ---------- Calculate Population Mean and Standard Deviation ----------
population_mean = df['Time_taken(min)'].mean()
population_std = df['Time_taken(min)'].std(ddof=0)  # Population std

# ---------- Calculate Z-Scores ----------
df['z_score'] = (df['Time_taken(min)'] - population_mean) / population_std

# ---------- Count Extreme Outliers ----------
extreme_outliers = df[np.abs(df['z_score']) > 2.5]
outlier_count = len(extreme_outliers)

print(f"Population Mean: {population_mean:.2f} minutes")
print(f"Population Standard Deviation: {population_std:.2f} minutes")
print(f"Count of Extreme Outliers (|z-score| > 2.5): {outlier_count}")
print()

# ---------- Generate Histogram with Reference Lines ----------
plt.figure(figsize=(12, 8))
plt.hist(df['z_score'], bins=50, color='steelblue', edgecolor='black', alpha=0.7)
plt.axvline(x=-2.5, color='red', linestyle='--', linewidth=2, label='z = -2.5 (Outlier Threshold)')
plt.axvline(x=2.5, color='red', linestyle='--', linewidth=2, label='z = +2.5 (Outlier Threshold)')
plt.axvline(x=0, color='green', linestyle='-', linewidth=1.5, alpha=0.7, label='Mean (z = 0)')
plt.xlabel('Z-Score', fontsize=12)
plt.ylabel('Frequency (Count of Deliveries)', fontsize=12)
plt.title(f'Z-Score Distribution of Delivery Times\n({outlier_count} extreme outliers detected)',
          fontsize=14, fontweight='bold')
plt.xlim(-4, 4)
plt.legend(fontsize=10)
plt.grid(alpha=0.3)
plt.tight_layout()
plt.savefig('zscore_histogram.png', dpi=300, bbox_inches='tight')
plt.close()
print("Chart saved: zscore_histogram.png")
print()

# ==========================================
# Analysis 6: Weather Condition with Minimum Average Delivery Time
# ==========================================
print("=" * 60)
print("ANALYSIS 6: Weather Condition with Minimum Average Delivery Time")
print("=" * 60)

# ---------- Group by Weather Conditions ----------
weather_avg = df.groupby('Weatherconditions')['Time_taken(min)'].mean().reset_index()
weather_avg.columns = ['weather_code', 'avg_time']

# ---------- Find Weather with Minimum Average Time ----------
min_weather_row = weather_avg.loc[weather_avg['avg_time'].idxmin()]
min_weather_code = int(min_weather_row['weather_code'])
min_weather_time = min_weather_row['avg_time']

print(f"Weather Condition with Minimum Average Delivery Time: {min_weather_code}: {min_weather_time:.2f}")
print()

# ==========================================
# Analysis 7: Rating Dispersion Ratio
# ==========================================
print("=" * 60)
print("ANALYSIS 7: Rating Dispersion Ratio (Max/Min Average Rating)")
print("=" * 60)

# ---------- Calculate Average Rating per Person ----------
person_ratings = df.groupby('Delivery_person_ID')['Delivery_person_Ratings'].mean().dropna()

# ---------- Find Max and Min Average Ratings ----------
max_avg_rating = person_ratings.max()
min_avg_rating = person_ratings.min()

# ---------- Calculate Ratio ----------
rating_ratio = max_avg_rating / min_avg_rating

print(f"Maximum Average Rating: {max_avg_rating:.4f}")
print(f"Minimum Average Rating: {min_avg_rating:.4f}")
print(f"Rating Dispersion Ratio (Max/Min): {rating_ratio:.4f}")
print()

# ==========================================
# Analysis 8: Std Dev for Traffic Density with Highest Average Time
# ==========================================
print("=" * 60)
print("ANALYSIS 8: Standard Deviation for Highest Avg Time Traffic Density")
print("=" * 60)

# ---------- Calculate Average Delivery Time by Traffic Density ----------
traffic_avg = df.groupby('Road_traffic_density')['Time_taken(min)'].mean().reset_index()
traffic_avg.columns = ['traffic_density', 'avg_time']

# ---------- Find Traffic Density with Maximum Average Time ----------
max_avg_traffic = traffic_avg.loc[traffic_avg['avg_time'].idxmax(), 'traffic_density']

# ---------- Calculate Std Dev for That Traffic Density ----------
traffic_std = df[df['Road_traffic_density'] == max_avg_traffic]['Time_taken(min)'].std(ddof=1)
traffic_mean = df[df['Road_traffic_density'] == max_avg_traffic]['Time_taken(min)'].mean()

print(f"Traffic Density Level with Highest Average Time: {int(max_avg_traffic)}")
print(f"Average Delivery Time for this level: {traffic_mean:.2f} minutes")
print(f"Standard Deviation for this level: {traffic_std:.2f} minutes")
print()

# ==========================================
# Summary of Key Outputs
# ==========================================
print("=" * 60)
print("SUMMARY OF KEY OUTPUTS")
print("=" * 60)
print(f"1. Maximum Coefficient of Variation: {max_cv_value:.2f}%")
print(f"2. Interquartile Range: {iqr:.2f} minutes")
print(f"3. Weighted Average Delivery Time: {weighted_avg:.2f} minutes")
print(f"4. Pearson Correlation Coefficient: {pearson_corr:.4f}")
print(f"5. Extreme Outlier Count: {outlier_count}")
print(f"6. Weather with Min Avg Time: {min_weather_code}: {min_weather_time:.2f}")
print(f"7. Rating Dispersion Ratio: {rating_ratio:.4f}")
print(f"8. Std Dev for Highest Avg Traffic: {traffic_std:.2f} minutes")
print("=" * 60)
print("\nAnalysis complete! All charts have been saved.")
