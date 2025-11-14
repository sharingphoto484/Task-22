# ==========================================
# Markov Volatility & Liquidity Analysis
# ==========================================
# Requirements: pandas, numpy, matplotlib, seaborn, scipy, scikit-learn, statsmodels, hmmlearn
# Input files: HDFCBANK.csv, ICICIBANK.csv, INDUSINDBK.csv (in same directory)
# Output files: probability_heatmap.png
# ==========================================

import pandas as pd
import numpy as np
import matplotlib.pyplot as plt
import seaborn as sns
from sklearn.decomposition import PCA
from sklearn.preprocessing import StandardScaler
import statsmodels.api as sm
from statsmodels.stats.sandwich_covariance import cov_hac
from scipy import stats
import warnings
warnings.filterwarnings('ignore')

# Try to import hmmlearn for Markov models
try:
    from hmmlearn import hmm
    USE_HMMLEARN = True
except ImportError:
    USE_HMMLEARN = False
    print("hmmlearn not available, will use statsmodels")

# ---------- Load CSVs Robustly ----------
print("=" * 60)
print("Loading bank data files...")
print("=" * 60)

# File names
bank_files = {
    'HDFCBANK': 'HDFCBANK.csv',
    'ICICIBANK': 'ICICIBANK.csv',
    'INDUSINDBK': 'INDUSINDBK.csv'
}

# Load each CSV
bank_data = {}
for bank_name, file_name in bank_files.items():
    df = pd.read_csv(file_name)
    df['Date'] = pd.to_datetime(df['Date'])
    df = df.sort_values('Date').reset_index(drop=True)
    bank_data[bank_name] = df
    print(f"Loaded {bank_name}: {len(df)} rows")

# ---------- Form Exact Daily Date Intersection ----------
print("\n" + "=" * 60)
print("Finding exact daily date intersection...")
print("=" * 60)

# Get date sets
date_sets = [set(df['Date']) for df in bank_data.values()]
common_dates = sorted(list(set.intersection(*date_sets)))
print(f"Common dates found: {len(common_dates)}")
print(f"Date range: {common_dates[0]} to {common_dates[-1]}")

# Filter to common dates only
for bank_name in bank_data.keys():
    bank_data[bank_name] = bank_data[bank_name][bank_data[bank_name]['Date'].isin(common_dates)]
    bank_data[bank_name] = bank_data[bank_name].sort_values('Date').reset_index(drop=True)
    print(f"{bank_name}: {len(bank_data[bank_name])} rows after intersection")

# ---------- Compute Daily Log Returns ----------
print("\n" + "=" * 60)
print("Computing daily log returns...")
print("=" * 60)

# Compute log returns for each bank
returns_data = {}
for bank_name, df in bank_data.items():
    df['log_close'] = np.log(df['Close'])
    df['log_return'] = df['log_close'].diff()
    # Remove first row (NaN from differencing)
    df_clean = df.iloc[1:].copy()
    returns_data[bank_name] = df_clean
    print(f"{bank_name}: {len(df_clean)} observations after differencing")
    print(f"  Mean return: {df_clean['log_return'].mean():.6f}")
    print(f"  Std return: {df_clean['log_return'].std():.6f}")

# Store common dates after differencing
common_dates_returns = returns_data['HDFCBANK']['Date'].tolist()

# ---------- Fit 2-State Markov Switching Volatility Models Using HMM ----------
print("\n" + "=" * 60)
print("Fitting 2-state Markov switching volatility models...")
print("=" * 60)

markov_models = {}
model_results = {}

for bank_name, df in returns_data.items():
    print(f"\nFitting model for {bank_name}...")

    # Prepare data for Markov switching model
    y = df['log_return'].values.reshape(-1, 1)

    try:
        # Use Gaussian HMM with 2 components
        # Better initialization using variance-based splitting
        returns_flat = y.flatten()
        median_abs = np.median(np.abs(returns_flat))

        # Split into low and high volatility groups
        low_vol_mask = np.abs(returns_flat) <= median_abs
        high_vol_mask = np.abs(returns_flat) > median_abs

        # Initial means (close to zero for both states)
        init_means = np.array([[0.0], [0.0]])

        # Initial covariances based on empirical split
        var_low = np.var(returns_flat[low_vol_mask])
        var_high = np.var(returns_flat[high_vol_mask])
        init_covars = np.array([[[var_low]], [[var_high]]])

        # Initial transition matrix (moderate persistence)
        init_transmat = np.array([[0.95, 0.05],
                                  [0.10, 0.90]])

        # Initial state probabilities
        init_startprob = np.array([0.75, 0.25])

        model = hmm.GaussianHMM(
            n_components=2,
            covariance_type="full",
            n_iter=2000,
            tol=1e-6,
            random_state=42,
            verbose=False,
            params="stmc",
            init_params=""
        )

        # Set initial parameters
        model.startprob_ = init_startprob
        model.transmat_ = init_transmat
        model.means_ = init_means
        model.covars_ = init_covars

        # Fit the model
        model.fit(y)

        # Get parameters
        means = model.means_.flatten()
        covars = model.covars_.flatten()
        trans_matrix = model.transmat_

        # Identify high volatility state (larger variance)
        if covars[0] > covars[1]:
            high_vol_state = 0
            high_vol_sigma = np.sqrt(covars[0])
            low_vol_sigma = np.sqrt(covars[1])
        else:
            high_vol_state = 1
            high_vol_sigma = np.sqrt(covars[1])
            low_vol_sigma = np.sqrt(covars[0])

        # Get smoothed probabilities using posterior
        smoothed_probs = model.predict_proba(y)

        # Probability of staying in high volatility state
        p_high_high = trans_matrix[high_vol_state, high_vol_state]

        # Expected duration of high volatility state
        expected_duration = 1.0 / (1.0 - p_high_high)

        # Stationary probability of high volatility state
        # For 2-state model: solve pi * P = pi
        # Analytical solution: pi_0 = (1-p11)/(2-p00-p11)
        p00 = trans_matrix[0, 0]
        p11 = trans_matrix[1, 1]
        if high_vol_state == 0:
            stationary_prob = (1 - p11) / (2 - p00 - p11)
        else:
            stationary_prob = (1 - p00) / (2 - p00 - p11)

        # Store results
        model_results[bank_name] = {
            'model': model,
            'high_vol_state': high_vol_state,
            'high_vol_sigma': high_vol_sigma,
            'low_vol_sigma': low_vol_sigma,
            'trans_matrix': trans_matrix,
            'smoothed_probs': smoothed_probs,
            'expected_duration': expected_duration,
            'stationary_prob': stationary_prob,
            'p_high_high': p_high_high
        }

        print(f"  High volatility state: State {high_vol_state}")
        print(f"  High vol sigma: {high_vol_sigma:.6f}")
        print(f"  Low vol sigma: {low_vol_sigma:.6f}")
        print(f"  P(stay in high vol): {p_high_high:.4f}")
        print(f"  Expected duration (high vol): {expected_duration:.2f} days")
        print(f"  Stationary prob (high vol): {stationary_prob:.4f}")

    except Exception as e:
        print(f"  ERROR fitting {bank_name}: {e}")
        model_results[bank_name] = None

# ---------- Compute Cross-Bank Averages for Markov Statistics ----------
print("\n" + "=" * 60)
print("Computing cross-bank averages...")
print("=" * 60)

expected_durations = [model_results[bank]['expected_duration']
                     for bank in ['HDFCBANK', 'ICICIBANK', 'INDUSINDBK']
                     if model_results.get(bank) is not None]
stationary_probs = [model_results[bank]['stationary_prob']
                   for bank in ['HDFCBANK', 'ICICIBANK', 'INDUSINDBK']
                   if model_results.get(bank) is not None]

if len(expected_durations) > 0:
    avg_expected_duration = np.mean(expected_durations)
    avg_stationary_prob = np.mean(stationary_probs)
    print(f"Successful models: {len(expected_durations)}/3")
    print(f"Average expected duration (high vol): {avg_expected_duration:.6f} days")
    print(f"Average stationary probability (high vol): {avg_stationary_prob:.6f}")
else:
    avg_expected_duration = np.nan
    avg_stationary_prob = np.nan
    print("WARNING: No models converged successfully")

# ---------- Compute Amihud Illiquidity Measures ----------
print("\n" + "=" * 60)
print("Computing Amihud illiquidity measures...")
print("=" * 60)

amihud_data = {}
for bank_name, df in returns_data.items():
    # Amihud = |log_return| / Turnover (no rescaling)
    df['amihud'] = np.abs(df['log_return']) / df['Turnover']

    # Replace inf and very large values
    df['amihud'] = df['amihud'].replace([np.inf, -np.inf], np.nan)

    amihud_data[bank_name] = df
    valid_amihud = df['amihud'][~np.isnan(df['amihud'])]
    print(f"{bank_name} Amihud mean: {valid_amihud.mean():.10f}, valid obs: {len(valid_amihud)}")

# ---------- Standardize Amihud Series ----------
print("\n" + "=" * 60)
print("Standardizing Amihud series...")
print("=" * 60)

amihud_standardized = {}
for bank_name, df in amihud_data.items():
    amihud_values = df['amihud'].values
    mean_val = np.nanmean(amihud_values)
    std_val = np.nanstd(amihud_values, ddof=0)
    amihud_std = (amihud_values - mean_val) / std_val
    amihud_standardized[bank_name] = amihud_std
    df['amihud_std'] = amihud_std
    print(f"{bank_name} standardized Amihud: mean={np.nanmean(amihud_std):.6f}, std={np.nanstd(amihud_std, ddof=0):.6f}")

# ---------- Perform PCA on Standardized Amihud ----------
print("\n" + "=" * 60)
print("Performing PCA on standardized Amihud measures...")
print("=" * 60)

# Create matrix for PCA (observations x banks)
amihud_matrix = np.column_stack([
    amihud_standardized['HDFCBANK'],
    amihud_standardized['ICICIBANK'],
    amihud_standardized['INDUSINDBK']
])

# Remove rows with NaN
valid_rows = ~np.isnan(amihud_matrix).any(axis=1)
amihud_matrix_clean = amihud_matrix[valid_rows]

print(f"PCA input: {amihud_matrix_clean.shape[0]} observations x {amihud_matrix_clean.shape[1]} banks")

# Fit PCA
pca = PCA()
pca.fit(amihud_matrix_clean)

# Get first principal component
pc1_loadings = pca.components_[0]
variance_explained = pca.explained_variance_ratio_[0] * 100

print(f"Variance explained by PC1: {variance_explained:.4f}%")
print(f"PC1 loadings (raw): {pc1_loadings}")

# Normalize to unit Euclidean norm
pc1_norm = pc1_loadings / np.linalg.norm(pc1_loadings)

# Ensure loading on HDFCBANK is positive
if pc1_norm[0] < 0:
    pc1_norm = -pc1_norm

print(f"PC1 loadings (normalized): {pc1_norm}")
print(f"  HDFCBANK: {pc1_norm[0]:.6f}")
print(f"  ICICIBANK: {pc1_norm[1]:.6f}")
print(f"  INDUSINDBK: {pc1_norm[2]:.6f}")

min_loading = np.min(pc1_norm)
print(f"Minimum loading: {min_loading:.6f}")

# ---------- Compute Parkinson Range-Based Variance ----------
print("\n" + "=" * 60)
print("Computing Parkinson range-based variance...")
print("=" * 60)

parkinson_variances = {}
for bank_name, df in returns_data.items():
    # Parkinson variance = (ln(High) - ln(Low))^2 / (4 * ln(2))
    log_high = np.log(df['High'])
    log_low = np.log(df['Low'])
    park_var = ((log_high - log_low) ** 2) / (4 * np.log(2))
    parkinson_variances[bank_name] = park_var.values
    print(f"{bank_name} Parkinson variance mean: {park_var.mean():.8f}")

# Compute daily cross-bank average
park_var_matrix = np.column_stack([
    parkinson_variances['HDFCBANK'],
    parkinson_variances['ICICIBANK'],
    parkinson_variances['INDUSINDBK']
])
daily_avg_park_var = np.mean(park_var_matrix, axis=1)

# Get 95th percentile
percentile_95 = np.percentile(daily_avg_park_var, 95)
print(f"\n95th percentile of daily average Parkinson variance: {percentile_95:.8f}")

# ---------- OLS Regressions with Newey-West Standard Errors ----------
print("\n" + "=" * 60)
print("Running OLS regressions with Newey-West SE...")
print("=" * 60)

p_values_deliverble = []

for bank_name, df in returns_data.items():
    print(f"\nRegression for {bank_name}:")

    # Check how many valid %Deliverble values we have
    valid_deliverble_count = (~df['%Deliverble'].isna()).sum()
    print(f"  Valid %Deliverble observations: {valid_deliverble_count}/{len(df)}")

    if valid_deliverble_count < 10:
        print(f"  Skipping {bank_name} - insufficient data")
        continue

    # Create a mask for valid observations
    # We need: non-NaN %Deliverble, non-NaN Turnover, and non-NaN next-day return
    df_reg = df.copy()
    df_reg['abs_return_next'] = np.abs(df_reg['log_return'].shift(-1))

    # Create validity mask
    valid_mask = (~df_reg['%Deliverble'].isna() &
                  ~df_reg['Turnover'].isna() &
                  ~df_reg['abs_return_next'].isna())

    df_valid = df_reg[valid_mask].copy()

    print(f"  Valid observations for regression: {len(df_valid)}")

    if len(df_valid) < 10:
        print(f"  Skipping {bank_name} - insufficient valid observations")
        continue

    # Standardize %Deliverble on valid data only
    deliverble = df_valid['%Deliverble'].values
    deliverble_mean = np.mean(deliverble)
    deliverble_std = np.std(deliverble, ddof=0)
    deliverble_std_array = (deliverble - deliverble_mean) / deliverble_std

    # Standardize Turnover on valid data only
    turnover = df_valid['Turnover'].values
    turnover_mean = np.mean(turnover)
    turnover_std = np.std(turnover, ddof=0)
    turnover_std_array = (turnover - turnover_mean) / turnover_std

    # Dependent variable
    y = df_valid['abs_return_next'].values

    # Independent variables
    X_data = np.column_stack([
        deliverble_std_array,
        turnover_std_array
    ])
    X = sm.add_constant(X_data)

    # Fit OLS
    ols_model = sm.OLS(y, X)
    ols_result = ols_model.fit()

    # Compute Newey-West covariance with lag 5
    nw_cov = cov_hac(ols_result, nlags=5)
    nw_se = np.sqrt(np.diag(nw_cov))

    # Compute t-statistics
    t_stats = ols_result.params / nw_se

    # Compute two-sided p-values
    p_values = 2 * (1 - stats.t.cdf(np.abs(t_stats), df=len(y) - len(ols_result.params)))

    # P-value for %Deliverble (index 1, after constant)
    p_val_deliverble = p_values[1]
    p_values_deliverble.append(p_val_deliverble)

    print(f"  Coefficient on %Deliverble: {ols_result.params[1]:.6f}")
    print(f"  Newey-West SE: {nw_se[1]:.6f}")
    print(f"  P-value: {p_val_deliverble:.6f}")

# Compute mean p-value
if len(p_values_deliverble) > 0:
    mean_p_value = np.mean(p_values_deliverble)
    print(f"\nMean p-value on %Deliverble: {mean_p_value:.6f}")
else:
    mean_p_value = np.nan
    print(f"\nWARNING: No successful regressions")

# ---------- Create Probability Heatmap ----------
print("\n" + "=" * 60)
print("Creating probability heatmap...")
print("=" * 60)

# Prepare heatmap data
heatmap_data = []
bank_order = ['HDFCBANK', 'ICICIBANK', 'INDUSINDBK']
successful_banks = []

for bank_name in bank_order:
    if model_results.get(bank_name) is not None:
        smoothed_probs = model_results[bank_name]['smoothed_probs']
        high_vol_state = model_results[bank_name]['high_vol_state']

        # Extract probability of high volatility state
        prob_high_vol = smoothed_probs[:, high_vol_state]
        heatmap_data.append(prob_high_vol)
        successful_banks.append(bank_name)

if len(heatmap_data) > 0:
    heatmap_matrix = np.array(heatmap_data)
    max_probability = np.max(heatmap_matrix)

    print(f"Heatmap shape: {heatmap_matrix.shape}")
    print(f"Maximum probability in heatmap: {max_probability:.6f}")
else:
    heatmap_matrix = None
    max_probability = np.nan
    print("WARNING: No successful models for heatmap")

# Create heatmap
if heatmap_matrix is not None:
    plt.figure(figsize=(20, 6))
    sns.heatmap(
        heatmap_matrix,
        cmap='YlOrRd',
        xticklabels=False,
        yticklabels=successful_banks,
        cbar_kws={'label': 'P(High Volatility State)'},
        vmin=0,
        vmax=1
    )
    plt.xlabel('Date (chronological order)')
    plt.ylabel('Bank')
    plt.title('Smoothed Probability of High Volatility State')
    plt.tight_layout()
    plt.savefig('probability_heatmap.png', dpi=150, bbox_inches='tight')
    print("Heatmap saved as 'probability_heatmap.png'")
    plt.close()

    # Print detailed heatmap statistics
    print("\n" + "=" * 60)
    print("HEATMAP DETAILED STATISTICS")
    print("=" * 60)
    for i, bank in enumerate(successful_banks):
        probs = heatmap_matrix[i]
        print(f"\n{bank}:")
        print(f"  Mean probability: {np.mean(probs):.6f}")
        print(f"  Median probability: {np.median(probs):.6f}")
        print(f"  Min probability: {np.min(probs):.6f}")
        print(f"  Max probability: {np.max(probs):.6f}")
        print(f"  Std deviation: {np.std(probs):.6f}")
        print(f"  Days with P > 0.5: {np.sum(probs > 0.5)}")
        print(f"  Days with P > 0.8: {np.sum(probs > 0.8)}")
        print(f"  Days with P > 0.9: {np.sum(probs > 0.9)}")

    # Print ASCII representation (simplified view)
    print("\n" + "=" * 60)
    print("HEATMAP ASCII VISUALIZATION (First 100 days)")
    print("=" * 60)
    print("Legend: . = Low (0-0.3), o = Medium (0.3-0.7), O = High (0.7-0.9), # = Very High (0.9-1.0)")
    print()
    for i, bank in enumerate(successful_banks):
        probs = heatmap_matrix[i, :100]  # First 100 days
        ascii_row = ""
        for p in probs:
            if p < 0.3:
                ascii_row += "."
            elif p < 0.7:
                ascii_row += "o"
            elif p < 0.9:
                ascii_row += "O"
            else:
                ascii_row += "#"
        print(f"{bank:12s}: {ascii_row}")
else:
    print("Skipping heatmap creation (no data available)")

# ---------- Liquidity Commonality Verdict ----------
print("\n" + "=" * 60)
print("Determining liquidity commonality verdict...")
print("=" * 60)

# Condition 1: PC1 explains >= 50% of variance
condition1 = variance_explained >= 50.0

# Condition 2: All three loadings are strictly positive
condition2 = all(pc1_norm > 0)

# Verdict: 1 if both conditions met, 0 otherwise
liquidity_verdict = 1 if (condition1 and condition2) else 0

print(f"PC1 variance explained >= 50%: {condition1} ({variance_explained:.2f}%)")
print(f"All loadings strictly positive: {condition2}")
print(f"  HDFCBANK: {pc1_norm[0]:.6f} > 0: {pc1_norm[0] > 0}")
print(f"  ICICIBANK: {pc1_norm[1]:.6f} > 0: {pc1_norm[1] > 0}")
print(f"  INDUSINDBK: {pc1_norm[2]:.6f} > 0: {pc1_norm[2] > 0}")
print(f"Liquidity commonality verdict: {liquidity_verdict}")

# ---------- Final Report ----------
print("\n" + "=" * 60)
print("FINAL RESULTS")
print("=" * 60)

print(f"\n1. Cross-bank average expected duration (high vol state): {avg_expected_duration:.6f}")
print(f"2. Cross-bank average stationary probability (high vol state): {avg_stationary_prob:.6f}")
print(f"3. PCA variance explained by PC1: {variance_explained:.6f}%")
print(f"4. Minimum PC1 loading: {min_loading:.6f}")
print(f"5. 95th percentile of cross-bank avg Parkinson variance: {percentile_95:.8f}")
print(f"6. Mean p-value on %Deliverble: {mean_p_value:.10f}")
print(f"7. Maximum probability in heatmap: {max_probability:.6f}")
print(f"8. Liquidity commonality verdict: {liquidity_verdict}")

print("\n" + "=" * 60)
print("Analysis complete!")
print("=" * 60)
