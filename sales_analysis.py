# ==========================================
# Integrated Sales Behavior Analysis Script
# ==========================================
# Requirements: pandas, numpy, matplotlib, scipy, scikit-learn
# Input files: sales_hourly.csv, sales_daily.csv, sales_monthly.csv (in same directory)
# Output files: Multiple visualization PNG files
# ==========================================

import pandas as pd
import numpy as np
import matplotlib.pyplot as plt
from scipy.signal import hilbert
from scipy.interpolate import splrep, splev, BSpline, UnivariateSpline
from scipy.stats import entropy
from scipy.special import rel_entr
from sklearn.metrics import mutual_info_score
import warnings
warnings.filterwarnings('ignore')

# ---------- Load CSVs Robustly ----------
def load_and_prepare_data():
    """Load all three datasets and prepare them for analysis"""

    # Load hourly data
    hourly = pd.read_csv('sales_hourly.csv')
    # Sum all sales columns (excluding date and metadata)
    sales_cols = ['M01AB', 'M01AE', 'N02BA', 'N02BE', 'N05B', 'N05C', 'R03', 'R06']
    hourly['Sales'] = hourly[sales_cols].apply(pd.to_numeric, errors='coerce').sum(axis=1)
    hourly['Date'] = pd.to_datetime(hourly['datum'], errors='coerce')
    hourly = hourly[['Date', 'Hour', 'Sales']].dropna()

    # Load daily data
    daily = pd.read_csv('sales_daily.csv')
    daily['Sales'] = daily[sales_cols].apply(pd.to_numeric, errors='coerce').sum(axis=1)
    daily['Date'] = pd.to_datetime(daily['datum'], errors='coerce')
    daily = daily[['Date', 'Sales']].dropna()

    # Load monthly data
    monthly = pd.read_csv('sales_monthly.csv')
    sales_cols_monthly = ['M01AB', 'M01AE', 'N02BA', 'N02BE', 'N05B', 'N05C', 'R03', 'R06']
    monthly['Sales'] = monthly[sales_cols_monthly].apply(pd.to_numeric, errors='coerce').sum(axis=1)
    monthly['Date'] = pd.to_datetime(monthly['datum'], errors='coerce')
    monthly = monthly[['Date', 'Sales']].dropna()

    return hourly, daily, monthly

# ---------- Stage 1: Hilbert Transform on Hourly Sales ----------
def hilbert_transform_analysis(hourly):
    """Apply Hilbert transform to hourly sales sequence"""

    # Create continuous sequence ordered by Date and hour
    hourly_sorted = hourly.sort_values(['Date', 'Hour']).reset_index(drop=True)
    sales_sequence = hourly_sorted['Sales'].values

    # Apply Hilbert transform
    analytic_signal = hilbert(sales_sequence)
    analytic_amplitude = np.abs(analytic_signal)

    # Calculate average analytic-signal amplitude
    avg_amplitude = np.mean(analytic_amplitude)

    # Generate line chart
    plt.figure(figsize=(12, 6))
    plt.plot(range(len(sales_sequence)), sales_sequence, linewidth=0.8)
    plt.xlabel('Hour Index')
    plt.ylabel('Sales Magnitude')
    plt.title('Hourly Sales Time Series')
    plt.tight_layout()
    plt.savefig('hourly_sales_chart.png', dpi=300)
    plt.close()

    return avg_amplitude, analytic_amplitude

# ---------- Stage 2: Nonlinear Association (Daily-Hourly) ----------
def nonlinear_association_analysis(hourly, daily):
    """Compute ratio of max hourly to daily sales and fit cubic spline"""

    # Find common dates
    hourly['DateOnly'] = hourly['Date'].dt.date
    daily['DateOnly'] = daily['Date'].dt.date
    common_dates = set(hourly['DateOnly']) & set(daily['DateOnly'])

    ratios = []
    dates_list = []

    for date in sorted(common_dates):
        # Get max hourly sales for this date
        max_hourly = hourly[hourly['DateOnly'] == date]['Sales'].max()
        # Get daily sales for this date
        daily_sales = daily[daily['DateOnly'] == date]['Sales'].values

        if len(daily_sales) > 0 and daily_sales[0] != 0 and not np.isnan(max_hourly):
            ratio = max_hourly / daily_sales[0]
            ratios.append(ratio)
            dates_list.append(date)

    ratios = np.array(ratios)
    x_indices = np.arange(len(ratios))

    # Fit cubic smoothing spline
    spline = UnivariateSpline(x_indices, ratios, k=3, s=len(ratios)*0.1)

    # Calculate curvature (integral of squared second derivatives)
    # Second derivative of spline
    x_fine = np.linspace(0, len(ratios)-1, 1000)
    second_deriv = spline.derivative(n=2)(x_fine)
    curvature = np.trapz(second_deriv**2, x_fine)

    # Generate scatter plot
    plt.figure(figsize=(12, 6))
    plt.scatter(x_indices, ratios, alpha=0.6, s=20)
    plt.plot(x_fine, spline(x_fine), 'r-', linewidth=2, label='Cubic Spline')
    plt.xlabel('Date Index')
    plt.ylabel('Ratio Magnitude')
    plt.title('Max Hourly to Daily Sales Ratio')
    plt.legend()
    plt.tight_layout()
    plt.savefig('ratio_scatter_plot.png', dpi=300)
    plt.close()

    return curvature

# ---------- Stage 3: Multiscale Intensity Signature (Daily) ----------
def multiscale_intensity_analysis(daily):
    """Calculate entropy for 5-day blocks of daily sales"""

    daily_sorted = daily.sort_values('Date').reset_index(drop=True)
    sales_values = daily_sorted['Sales'].values

    block_size = 5
    entropies = []

    # Create blocks of 5 consecutive days
    for i in range(len(sales_values) - block_size + 1):
        block = sales_values[i:i+block_size]

        # Normalize to create probability distribution
        if block.sum() > 0:
            prob_dist = block / block.sum()
            # Calculate entropy
            block_entropy = entropy(prob_dist)
            entropies.append(block_entropy)

    mean_entropy = np.mean(entropies)

    # Generate boxplot
    plt.figure(figsize=(8, 6))
    plt.boxplot([entropies], labels=['5-Day Blocks'])
    plt.xlabel('Block Category')
    plt.ylabel('Entropy')
    plt.title('Blockwise Entropy Distribution')
    plt.tight_layout()
    plt.savefig('entropy_boxplot.png', dpi=300)
    plt.close()

    return mean_entropy

# ---------- Stage 4: Structural Smoothness Index (Monthly) ----------
def structural_smoothness_analysis(monthly):
    """Fit B-spline degree 4 to monthly sales"""

    monthly_sorted = monthly.sort_values('Date').reset_index(drop=True)
    sales_values = monthly_sorted['Sales'].values
    x_indices = np.arange(len(sales_values))

    # Fit B-spline of degree 4
    k = min(4, len(sales_values) - 1)  # Ensure k is valid
    spline = UnivariateSpline(x_indices, sales_values, k=k, s=len(sales_values)*10)
    fitted_values = spline(x_indices)

    # Calculate integrated squared deviation
    squared_deviation = np.sum((fitted_values - sales_values)**2)

    # Generate line chart
    plt.figure(figsize=(12, 6))
    plt.plot(x_indices, sales_values, 'o-', label='Observed', linewidth=2, markersize=6)
    plt.plot(x_indices, fitted_values, 's-', label='B-Spline Fitted', linewidth=2, markersize=4)
    plt.xlabel('Month Index')
    plt.ylabel('Sales Value')
    plt.title('Monthly Sales: Observed vs Fitted')
    plt.legend()
    plt.tight_layout()
    plt.savefig('monthly_sales_chart.png', dpi=300)
    plt.close()

    return squared_deviation

# ---------- Stage 5: Cross-Frequency Coupling (Daily-Monthly) ----------
def cross_frequency_coupling_analysis(daily, monthly):
    """Calculate mutual information between daily and monthly sales"""

    # Add month column to daily
    daily['Month'] = daily['Date'].dt.to_period('M')
    monthly['Month'] = monthly['Date'].dt.to_period('M')

    # Merge daily with monthly
    merged = daily.merge(monthly, on='Month', suffixes=('_daily', '_monthly'))

    daily_sales = merged['Sales_daily'].values
    monthly_sales = merged['Sales_monthly'].values

    # Bin both into 10 equal-frequency intervals
    daily_bins = pd.qcut(daily_sales, q=10, labels=False, duplicates='drop')
    monthly_bins = pd.qcut(monthly_sales, q=10, labels=False, duplicates='drop')

    # Calculate mutual information
    mi_estimate = mutual_info_score(daily_bins, monthly_bins)

    # Generate heatmap
    contingency = pd.crosstab(daily_bins, monthly_bins)
    plt.figure(figsize=(10, 8))
    plt.imshow(contingency.values, cmap='YlOrRd', aspect='auto')
    plt.colorbar(label='Frequency')
    plt.xlabel('Monthly Bin')
    plt.ylabel('Daily Bin')
    plt.title('Daily-to-Monthly Sales Association Heatmap')
    plt.tight_layout()
    plt.savefig('coupling_heatmap.png', dpi=300)
    plt.close()

    return mi_estimate

# ---------- Stage 6: Structural Divergence Index (Hourly) ----------
def structural_divergence_analysis(hourly):
    """Fit principal curve to hour-level means"""

    # Group by hour and calculate mean sales
    hour_means = hourly.groupby('Hour')['Sales'].mean().sort_index()
    hours = hour_means.index.values
    means = hour_means.values

    # Fit principal curve (using polynomial approximation)
    # Since principal curve is complex, we'll use polynomial fitting as approximation
    coeffs = np.polyfit(hours, means, deg=5)
    fitted_curve = np.polyval(coeffs, hours)

    # Sum of squared distances
    distance_sum = np.sum((means - fitted_curve)**2)

    # Generate scatter plot
    plt.figure(figsize=(12, 6))
    plt.scatter(hours, means, alpha=0.7, s=50, label='Hour-level Means')
    plt.plot(hours, fitted_curve, 'r-', linewidth=2, label='Fitted Curve')
    plt.xlabel('Hour')
    plt.ylabel('Mean Sales')
    plt.title('Hour-Level Mean Sales with Fitted Curve')
    plt.legend()
    plt.tight_layout()
    plt.savefig('hour_means_scatter.png', dpi=300)
    plt.close()

    return distance_sum

# ---------- Stage 7: Cross-Scale Regularity Measure ----------
def cross_scale_regularity_analysis(daily, monthly):
    """Calculate std of daily-to-monthly sales ratio"""

    # Add month column
    daily['Month'] = daily['Date'].dt.to_period('M')
    monthly['Month'] = monthly['Date'].dt.to_period('M')

    # Merge
    merged = daily.merge(monthly, on='Month', suffixes=('_daily', '_monthly'))

    # Calculate ratio
    ratios = merged['Sales_daily'] / merged['Sales_monthly']
    ratios = ratios.replace([np.inf, -np.inf], np.nan).dropna()

    # Standard deviation
    std_deviation = np.std(ratios)

    return std_deviation

# ---------- Stage 8: Stability Index (Daily) ----------
def stability_index_analysis(daily):
    """Calculate mean absolute difference from 5-day centered median"""

    daily_sorted = daily.sort_values('Date').reset_index(drop=True)
    sales_values = daily_sorted['Sales'].values

    # Calculate centered 5-day moving median
    centered_median = pd.Series(sales_values).rolling(window=5, center=True).median()

    # Remove NaN values (edges where full window doesn't exist)
    valid_indices = ~centered_median.isna()

    # Mean absolute difference
    mad = np.mean(np.abs(sales_values[valid_indices] - centered_median[valid_indices]))

    return mad

# ---------- Main Analysis Pipeline ----------
def main():
    print("=" * 60)
    print("SALES BEHAVIOR ANALYSIS - KEY OUTPUTS")
    print("=" * 60)

    # Load data
    print("\nLoading datasets...")
    hourly, daily, monthly = load_and_prepare_data()

    # Stage 1: Hilbert Transform
    print("\n[1/8] Computing Hilbert transform...")
    avg_amplitude, analytic_amplitude = hilbert_transform_analysis(hourly)
    print(f"Average Analytic-Signal Amplitude: {avg_amplitude:.4f}")

    # Stage 2: Nonlinear Association
    print("\n[2/8] Computing nonlinear association...")
    curvature = nonlinear_association_analysis(hourly, daily)
    print(f"Spline Curvature: {curvature:.4f}")

    # Stage 3: Multiscale Intensity
    print("\n[3/8] Computing multiscale intensity...")
    mean_entropy = multiscale_intensity_analysis(daily)
    print(f"Mean Entropy: {mean_entropy:.3f}")

    # Stage 4: Structural Smoothness
    print("\n[4/8] Computing structural smoothness...")
    squared_deviation = structural_smoothness_analysis(monthly)
    print(f"Integrated Squared Deviation: {squared_deviation:.3f}")

    # Stage 5: Cross-Frequency Coupling
    print("\n[5/8] Computing cross-frequency coupling...")
    mi_estimate = cross_frequency_coupling_analysis(daily, monthly)
    print(f"Mutual Information Estimate: {mi_estimate:.4f}")

    # Stage 6: Structural Divergence
    print("\n[6/8] Computing structural divergence...")
    distance_sum = structural_divergence_analysis(hourly)
    print(f"Distance Sum: {distance_sum:.3f}")

    # Stage 7: Cross-Scale Regularity
    print("\n[7/8] Computing cross-scale regularity...")
    std_deviation = cross_scale_regularity_analysis(daily, monthly)
    print(f"Standard Deviation: {std_deviation:.3f}")

    # Stage 8: Stability Index
    print("\n[8/8] Computing stability index...")
    mad = stability_index_analysis(daily)
    print(f"Mean Absolute Difference: {mad:.3f}")

    # Answer open-ended question
    print("\n" + "=" * 60)
    print("VERDICT ON EXTREME ANALYTIC-SIGNAL AMPLITUDES")
    print("=" * 60)
    print("Extreme analytic-signal amplitudes emerge when rapid hourly sales fluctuations coincide with sustained high-frequency variability across consecutive time intervals.")

    print("\n" + "=" * 60)
    print("Analysis complete. All visualizations saved as PNG files.")
    print("=" * 60)

if __name__ == "__main__":
    main()
