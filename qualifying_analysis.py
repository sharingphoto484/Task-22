import pandas as pd
import numpy as np
import matplotlib.pyplot as plt
import seaborn as sns
from sklearn.cluster import KMeans
from sklearn.preprocessing import StandardScaler
from scipy import stats
import statsmodels.api as sm
from statsmodels.regression.quantile_regression import QuantReg
import re
import warnings
warnings.filterwarnings('ignore')

def parse_time_value(value):
    """
    Parse a time/gap value according to the specification:
    - M:SS.mmm, MM:SS.mmm, MM:SS.s → absolute lap time in seconds
    - Numeric (with/without +) → gap in seconds
    - "- -", "--", empty, or missing → None
    """
    if pd.isna(value) or value == "":
        return None, None

    value_str = str(value).strip()

    # Check for missing indicators
    if value_str in ["- -", "--", ""]:
        return None, None

    # Check for time format (M:SS.mmm or MM:SS.mmm or MM:SS.s)
    time_pattern = r'^(\d{1,2}):(\d{2})\.(\d+)$'
    match = re.match(time_pattern, value_str)
    if match:
        minutes = int(match.group(1))
        seconds = int(match.group(2))
        fraction = match.group(3)
        # Normalize fraction to milliseconds
        fraction_seconds = float(f"0.{fraction}")
        total_seconds = minutes * 60 + seconds + fraction_seconds
        return total_seconds, 'absolute'

    # Check for numeric gap (with or without + sign)
    try:
        # Remove leading + if present
        if value_str.startswith('+'):
            value_str = value_str[1:]
        gap = float(value_str)
        return gap, 'gap'
    except ValueError:
        return None, None

def load_and_process_session(filepath, session_label):
    """Load and process a single qualifying session."""
    df = pd.read_csv(filepath)
    df['session'] = session_label

    # Parse gap column
    parsed = df['gap'].apply(parse_time_value)
    df['parsed_value'] = parsed.apply(lambda x: x[0])
    df['value_type'] = parsed.apply(lambda x: x[1])

    return df

def compute_absolute_lap_times(df):
    """
    Within each session and index snapshot, compute absolute lap times.
    Take minimum absolute lap time as reference and convert gaps to absolute times.
    """
    results = []

    for session in df['session'].unique():
        session_df = df[df['session'] == session].copy()

        for idx in session_df['index'].unique():
            snapshot = session_df[session_df['index'] == idx].copy()

            # Find absolute times in this snapshot
            absolute_times = snapshot[snapshot['value_type'] == 'absolute']['parsed_value']

            if len(absolute_times) > 0:
                # Use minimum absolute time as reference
                reference_time = absolute_times.min()

                # Convert all values to absolute times
                snapshot['lap_time_sec'] = snapshot.apply(
                    lambda row: row['parsed_value'] if row['value_type'] == 'absolute'
                    else (reference_time + row['parsed_value'] if row['value_type'] == 'gap' else None),
                    axis=1
                )
            else:
                # No absolute times in this snapshot
                snapshot['lap_time_sec'] = None

            results.append(snapshot)

    return pd.concat(results, ignore_index=True)

def main():
    print("=" * 80)
    print("70th Anniversary Grand Prix - Qualifying Analysis")
    print("=" * 80)

    # Load all three sessions
    print("\n1. Loading data files...")
    q1 = load_and_process_session('70_Anniversary Q1_LapbyLap.csv', 'Q1')
    q2 = load_and_process_session('70_Anniversary Q2_LapbyLap.csv', 'Q2')
    q3 = load_and_process_session('70_Anniversary Q3_LapbyLap.csv', 'Q3')

    # Combine all sessions
    df_all = pd.concat([q1, q2, q3], ignore_index=True)
    print(f"   Total rows loaded: {len(df_all)}")

    # Compute absolute lap times
    print("\n2. Computing absolute lap times...")
    df_all = compute_absolute_lap_times(df_all)

    # Drop rows without valid lap_time_sec
    df_valid = df_all[df_all['lap_time_sec'].notna()].copy()
    print(f"   Valid laps: {len(df_valid)}")

    # Compute session-level metrics
    print("\n3. Computing session-level metrics...")
    session_results = []

    for session in ['Q1', 'Q2', 'Q3']:
        session_df = df_valid[df_valid['session'] == session].copy()

        if len(session_df) > 0:
            pole_time_sec = session_df['lap_time_sec'].min()
            session_df['pole_time_sec'] = pole_time_sec
            session_df['gap_to_pole_sec'] = session_df['lap_time_sec'] - pole_time_sec
            session_df['gap_to_pole_pct'] = (session_df['gap_to_pole_sec'] / pole_time_sec) * 100

            session_results.append(session_df)
            print(f"   {session}: Pole time = {pole_time_sec:.3f}s")

    df_combined = pd.concat(session_results, ignore_index=True)

    # Compute global z-score of gap_to_pole_pct
    print("\n4. Computing global z-scores...")
    mean_gap_pct = df_combined['gap_to_pole_pct'].mean()
    std_gap_pct = df_combined['gap_to_pole_pct'].std()
    df_combined['z_score'] = (df_combined['gap_to_pole_pct'] - mean_gap_pct) / std_gap_pct
    print(f"   Mean gap_to_pole_pct: {mean_gap_pct:.4f}%")
    print(f"   Std gap_to_pole_pct: {std_gap_pct:.4f}%")

    # Aggregate to driver level
    print("\n5. Aggregating to driver level...")
    driver_features = df_combined.groupby('driver').agg({
        'lap_time_sec': 'min',  # best_lap_time_sec
        'gap_to_pole_pct': 'mean',  # mean_gap_to_pole_pct
    }).reset_index()

    driver_features.columns = ['driver', 'best_lap_time_sec', 'mean_gap_to_pole_pct']

    # Compute share_within_0_5pct
    def compute_share_within_0_5pct(group):
        return (group['gap_to_pole_pct'] < 0.5).sum() / len(group)

    share_within = df_combined.groupby('driver').apply(compute_share_within_0_5pct).reset_index()
    share_within.columns = ['driver', 'share_within_0_5pct']

    driver_features = driver_features.merge(share_within, on='driver')
    print(f"   Number of drivers: {len(driver_features)}")

    # Standardize driver features
    print("\n6. Standardizing driver features and running k-means clustering...")
    scaler = StandardScaler()
    feature_cols = ['best_lap_time_sec', 'mean_gap_to_pole_pct', 'share_within_0_5pct']
    driver_features_std = scaler.fit_transform(driver_features[feature_cols])

    # Create DataFrame with standardized features
    driver_features_std_df = pd.DataFrame(
        driver_features_std,
        columns=[f'{col}_std' for col in feature_cols],
        index=driver_features.index
    )
    driver_features = pd.concat([driver_features, driver_features_std_df], axis=1)

    # Run k-means clustering
    kmeans = KMeans(n_clusters=3, random_state=42, n_init=10)
    driver_features['cluster'] = kmeans.fit_predict(driver_features_std)

    # Identify fastest cluster
    cluster_means = driver_features.groupby('cluster')['mean_gap_to_pole_pct'].mean()
    fastest_cluster = cluster_means.idxmin()
    print(f"   Fastest cluster: {fastest_cluster}")
    print(f"   Cluster means of mean_gap_to_pole_pct:")
    for cluster, mean_val in cluster_means.items():
        print(f"      Cluster {cluster}: {mean_val:.4f}%")

    # Add cluster info to main dataframe
    df_combined = df_combined.merge(driver_features[['driver', 'cluster']], on='driver')
    df_combined['is_fastest_cluster'] = (df_combined['cluster'] == fastest_cluster).astype(int)

    # Identify outliers
    print("\n7. Identifying outlier laps...")
    df_combined['is_outlier'] = (df_combined['z_score'] < -3).astype(int)
    n_outliers = df_combined['is_outlier'].sum()
    print(f"   Number of outlier laps (z_score < -3): {n_outliers}")

    # Fit quantile regression
    print("\n8. Fitting quantile regression (tau=0.9)...")
    X = driver_features_std_df
    X = sm.add_constant(X)

    # Merge standardized features with df_combined
    df_combined = df_combined.merge(
        driver_features[['driver'] + [f'{col}_std' for col in feature_cols]],
        on='driver'
    )

    y = df_combined['gap_to_pole_sec']
    X_reg = df_combined[[f'{col}_std' for col in feature_cols]].copy()
    X_reg = sm.add_constant(X_reg)

    qr_model = QuantReg(y, X_reg)
    qr_results = qr_model.fit(q=0.9)

    print("\n   Quantile Regression Results (tau=0.9):")
    print(qr_results.summary())

    # Get fitted values
    df_combined['qr_fitted'] = qr_results.fittedvalues

    # Perform Welch t-test
    print("\n9. Performing Welch t-test...")
    fastest_cluster_laps = df_combined[df_combined['is_fastest_cluster'] == 1]['gap_to_pole_pct']
    other_laps = df_combined[df_combined['is_fastest_cluster'] == 0]['gap_to_pole_pct']

    t_stat, p_value = stats.ttest_ind(fastest_cluster_laps, other_laps, equal_var=False)
    print(f"   T-statistic: {t_stat:.6f}")
    print(f"   P-value: {p_value:.6e}")

    # Create correlation heatmap
    print("\n10. Creating correlation heatmap...")

    # For the heatmap, we need correlations between standardized driver features and fitted values
    # We need to get one fitted value per driver (using their best lap or mean)
    driver_qr_fitted = df_combined.groupby('driver').agg({
        'best_lap_time_sec_std': 'first',
        'mean_gap_to_pole_pct_std': 'first',
        'share_within_0_5pct_std': 'first',
        'qr_fitted': 'mean'
    }).reset_index()

    # Compute correlation matrix
    corr_cols = ['best_lap_time_sec_std', 'mean_gap_to_pole_pct_std', 'share_within_0_5pct_std', 'qr_fitted']
    corr_matrix = driver_qr_fitted[corr_cols].corr()

    # Rename columns for better display
    corr_matrix.index = ['best_lap_time_sec_std', 'mean_gap_to_pole_pct_std', 'share_within_0_5pct_std', 'QR_fitted_tau_0.9']
    corr_matrix.columns = ['best_lap_time_sec_std', 'mean_gap_to_pole_pct_std', 'share_within_0_5pct_std', 'QR_fitted_tau_0.9']

    plt.figure(figsize=(10, 8))
    sns.heatmap(corr_matrix, annot=True, fmt='.3f', cmap='coolwarm', center=0,
                square=True, linewidths=1, cbar_kws={"shrink": 0.8})
    plt.title('Correlation Matrix: Standardized Driver Features & Quantile Regression Fitted Values\n(tau=0.9)',
              fontsize=12, fontweight='bold')
    plt.tight_layout()
    plt.savefig('correlation_heatmap.png', dpi=300, bbox_inches='tight')
    print("   Heatmap saved as 'correlation_heatmap.png'")

    # Compute required metrics
    print("\n" + "=" * 80)
    print("FINAL REPORT")
    print("=" * 80)

    # 1. Overall median of gap_to_pole_sec
    median_gap_to_pole_sec = df_combined['gap_to_pole_sec'].median()
    print(f"\n1. Overall median of gap_to_pole_sec: {median_gap_to_pole_sec:.6f}")

    # 2. Mean z_score of gap_to_pole_pct for fastest_cluster laps
    mean_z_score_fastest = df_combined[df_combined['is_fastest_cluster'] == 1]['z_score'].mean()
    print(f"\n2. Mean z_score of gap_to_pole_pct for fastest_cluster laps: {mean_z_score_fastest:.6f}")

    # 3. Coefficient on best_lap_time_sec from tau=0.9 quantile regression
    coef_best_lap_time = qr_results.params['best_lap_time_sec_std']
    print(f"\n3. Coefficient on best_lap_time_sec_std (tau=0.9 QR): {coef_best_lap_time:.6f}")

    # 4. P-value from Welch t-test
    print(f"\n4. P-value from Welch t-test: {p_value:.6e}")

    # 5. Binary: at least one outlier exists
    has_outlier = int(n_outliers > 0)
    print(f"\n5. At least one outlier lap exists (z_score < -3): {has_outlier}")

    # 6. Binary: fastest_cluster drivers account for >0.5 of outlier laps
    if n_outliers > 0:
        outlier_laps = df_combined[df_combined['is_outlier'] == 1]
        fastest_cluster_outliers = outlier_laps[outlier_laps['is_fastest_cluster'] == 1]
        share_fastest_outliers = len(fastest_cluster_outliers) / n_outliers
        fastest_cluster_majority_outliers = int(share_fastest_outliers > 0.5)
        print(f"\n6. Fastest_cluster drivers account for >0.5 of outlier laps: {fastest_cluster_majority_outliers}")
        print(f"   (Share: {share_fastest_outliers:.4f})")
    else:
        fastest_cluster_majority_outliers = 0
        print(f"\n6. Fastest_cluster drivers account for >0.5 of outlier laps: {fastest_cluster_majority_outliers}")
        print(f"   (No outliers exist)")

    print("\n" + "=" * 80)

    # Save detailed results
    print("\n11. Saving detailed results...")
    df_combined.to_csv('qualifying_combined_dataset.csv', index=False)
    driver_features.to_csv('driver_features.csv', index=False)

    print("\nAnalysis complete!")
    print(f"   - Combined dataset: qualifying_combined_dataset.csv ({len(df_combined)} laps)")
    print(f"   - Driver features: driver_features.csv ({len(driver_features)} drivers)")
    print(f"   - Correlation heatmap: correlation_heatmap.png")

    return {
        'median_gap_to_pole_sec': median_gap_to_pole_sec,
        'mean_z_score_fastest': mean_z_score_fastest,
        'coef_best_lap_time': coef_best_lap_time,
        'p_value_welch': p_value,
        'has_outlier': has_outlier,
        'fastest_cluster_majority_outliers': fastest_cluster_majority_outliers
    }

if __name__ == "__main__":
    results = main()
