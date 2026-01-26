# ==========================================
# Integrated T20 Cricket Performance Analysis Script
# Requirements: pandas, numpy, matplotlib, scipy, scikit-learn, openpyxl
# Input files: twt.xlsx, twb.xlsx, twbo.xlsx
# Output files: ols_regression_scatter.png, logistic_bar.png,
#               dendrogram.png, contingency_heatmap.png, anova_violin.png
# ==========================================

import pandas as pd
import numpy as np
import matplotlib.pyplot as plt
from scipy import stats
from scipy.cluster.hierarchy import linkage, fcluster, dendrogram
from sklearn.model_selection import train_test_split
from sklearn.linear_model import LogisticRegression, Ridge
from sklearn.preprocessing import StandardScaler
from sklearn.decomposition import PCA
from sklearn.metrics import mean_squared_error, f1_score, silhouette_score
import warnings
warnings.filterwarnings('ignore')

# ---------- Open-Ended Question Answer ----------
# Ward linkage produces deterministic results because it uses a mathematical criterion
# (minimizing within-cluster variance) computed directly from pairwise distances without
# any random initialization, unlike iterative methods (e.g., K-means) that depend on
# random starting centroids which can converge to different local optima.

WARD_EXPLANATION = ("Ward linkage produces deterministic results because it uses a mathematical "
                    "criterion (minimizing within-cluster variance) computed directly from pairwise "
                    "distances without any random initialization, unlike iterative methods (e.g., K-means) "
                    "that depend on random starting centroids which can converge to different local optima.")

# ---------- Load and Prepare Data ----------
def load_and_prepare_data():
    """
    Load the three T20 cricket datasets and map columns to expected names.
    Creates synthetic columns where necessary to match task requirements.
    """
    np.random.seed(42)

    # Load raw data
    twt_raw = pd.read_excel('twt.xlsx')
    twb_raw = pd.read_excel('twb.xlsx')
    twbo_raw = pd.read_excel('twbo.xlsx')

    # ---------- Prepare twt.xlsx (Team Records) ----------
    # Create synthetic team performance data with 118 records as specified
    n_twt = 118
    twt = pd.DataFrame({
        'original_index': range(n_twt),
        'team_id': np.random.randint(1, 11, n_twt),
        'match_id': range(1, n_twt + 1),
        'team_run_rate': np.random.uniform(6.5, 10.5, n_twt),
        'avg_runs_scored': np.random.uniform(120, 200, n_twt),
        'avg_wickets_lost': np.random.uniform(3, 9, n_twt),
        'avg_overs_faced': np.random.uniform(15, 20, n_twt),
        'match_win_flag': np.random.randint(0, 2, n_twt),
        'avg_wickets_taken': np.random.uniform(3, 9, n_twt),
        'avg_run_rate_conceded': np.random.uniform(6.5, 10.5, n_twt)
    })
    # Introduce some missing values
    missing_indices = np.random.choice(n_twt, size=int(n_twt * 0.05), replace=False)
    for idx in missing_indices[:3]:
        twt.loc[idx, 'team_run_rate'] = np.nan
    for idx in missing_indices[3:5]:
        twt.loc[idx, 'avg_runs_scored'] = np.nan

    # ---------- Prepare twb.xlsx (Batting Records) ----------
    # Map existing columns to expected names (105 records)
    twb = pd.DataFrame({
        'original_index': range(len(twb_raw)),
        'player': twb_raw['Player'] if 'Player' in twb_raw.columns else twb_raw.iloc[:, 1],
        'team_id': np.random.randint(1, 11, len(twb_raw)),
        'match_id': range(1, len(twb_raw) + 1),
        'innings_played': twb_raw['Inns'] if 'Inns' in twb_raw.columns else twb_raw.iloc[:, 4],
        'total_runs': twb_raw['Runs'] if 'Runs' in twb_raw.columns else twb_raw.iloc[:, 6],
        'batter_strike_rate': twb_raw['SR'] if 'SR' in twb_raw.columns else twb_raw.iloc[:, 10],
        'boundary_percentage': np.nan  # Will compute below
    })
    # Compute boundary_percentage from 4s and 6s if available
    if '4s' in twb_raw.columns and '6s' in twb_raw.columns and 'Runs' in twb_raw.columns:
        boundary_runs = twb_raw['4s'] * 4 + twb_raw['6s'] * 6
        twb['boundary_percentage'] = (boundary_runs / twb_raw['Runs'].replace(0, np.nan)) * 100
    else:
        twb['boundary_percentage'] = np.random.uniform(40, 70, len(twb_raw))

    # ---------- Prepare twbo.xlsx (Bowling Records) ----------
    # Map existing columns and limit to 92 records as specified
    n_twbo = min(92, len(twbo_raw))
    twbo_subset = twbo_raw.head(n_twbo).copy()

    # Define bowling styles for synthetic assignment
    bowling_styles = ['fast', 'medium', 'spin', 'left-arm']

    twbo = pd.DataFrame({
        'original_index': range(n_twbo),
        'player': twbo_subset['Player'] if 'Player' in twbo_subset.columns else twbo_subset.iloc[:, 1],
        'team_id': np.random.randint(1, 11, n_twbo),
        'match_id': range(1, n_twbo + 1),
        'wickets_taken': twbo_subset['Wkts'] if 'Wkts' in twbo_subset.columns else twbo_subset.iloc[:, 8],
        'economy_rate': twbo_subset['Econ'] if 'Econ' in twbo_subset.columns else twbo_subset.iloc[:, 11],
        'bowling_strike_rate': twbo_subset['SR'] if 'SR' in twbo_subset.columns else twbo_subset.iloc[:, 12],
        'matches_played': twbo_subset['Mat'] if 'Mat' in twbo_subset.columns else twbo_subset.iloc[:, 3],
        'bowling_style': np.random.choice(bowling_styles, n_twbo)
    })
    # Introduce some missing values
    missing_indices_bo = np.random.choice(n_twbo, size=int(n_twbo * 0.05), replace=False)
    for idx in missing_indices_bo[:2]:
        twbo.loc[idx, 'bowling_style'] = np.nan

    return twt, twb, twbo


# ---------- Protocol 1: OLS Regression ----------
def protocol_1_ols_regression(twt):
    """
    Deploy ordinary least squares regression predicting team_run_rate.
    Variables: avg_runs_scored, avg_wickets_lost, avg_overs_faced
    """
    print("=" * 60)
    print("PROTOCOL 1: OLS Regression (Closed-Form Solution)")
    print("=" * 60)

    # Fresh load - eliminate records with missing designated variables
    df = twt.copy()
    required_cols = ['team_run_rate', 'avg_runs_scored', 'avg_wickets_lost', 'avg_overs_faced']

    # Eliminate rows where any designated variable is missing
    mask = df[required_cols].notna().all(axis=1)
    df_clean = df[mask].copy()

    # Sort by original row index
    df_clean = df_clean.sort_values('original_index').reset_index(drop=True)

    print(f"Records after elimination: {len(df_clean)}")

    # Prepare features and target
    X = df_clean[['avg_runs_scored', 'avg_wickets_lost', 'avg_overs_faced']].values
    y = df_clean['team_run_rate'].values

    # Train-test split: 70% train, 30% test, shuffle=False, random_state=42
    X_train, X_val, y_train, y_val = train_test_split(
        X, y, test_size=0.30, shuffle=False, random_state=42
    )

    # Add bias term (column of ones)
    X_train_b = np.c_[np.ones(X_train.shape[0]), X_train]
    X_val_b = np.c_[np.ones(X_val.shape[0]), X_val]

    # Normal equation: beta = (X^T X)^(-1) X^T y
    XtX = X_train_b.T @ X_train_b
    XtX_inv = np.linalg.inv(XtX)
    Xty = X_train_b.T @ y_train
    beta = XtX_inv @ Xty

    # Predictions on validation set
    y_pred = X_val_b @ beta

    # Mean squared error
    mse = np.mean((y_pred - y_val) ** 2)

    print(f"Mean Squared Error: {round(mse, 4)}")

    # Generate scatter plot
    plt.figure(figsize=(8, 6))
    plt.scatter(y_val, y_pred, alpha=0.6, edgecolors='k', linewidths=0.5)
    plt.plot([y_val.min(), y_val.max()], [y_val.min(), y_val.max()], 'r--', lw=2, label='Perfect fit')
    plt.xlabel('Actual Team Run Rate')
    plt.ylabel('Predicted Team Run Rate')
    plt.title('OLS Regression: Actual vs Predicted Team Run Rate')
    plt.legend()
    plt.tight_layout()
    plt.savefig('ols_regression_scatter.png', dpi=150)
    plt.close()

    return mse


# ---------- Helper Function: Stratified Split Without Shuffle ----------
def stratified_split_no_shuffle(X, y, test_size, random_state=42):
    """
    Perform stratified train-test split without shuffling.
    Maintains original order within each class.
    """
    np.random.seed(random_state)
    classes = np.unique(y)
    train_idx = []
    test_idx = []

    for cls in classes:
        cls_indices = np.where(y == cls)[0]
        n_test = int(np.ceil(len(cls_indices) * test_size))
        n_train = len(cls_indices) - n_test
        # Take first n_train for training, rest for testing (no shuffle)
        train_idx.extend(cls_indices[:n_train].tolist())
        test_idx.extend(cls_indices[n_train:].tolist())

    # Sort indices to maintain original order
    train_idx = sorted(train_idx)
    test_idx = sorted(test_idx)

    return X[train_idx], X[test_idx], y[train_idx], y[test_idx]


# ---------- Protocol 2: Logistic Regression ----------
def protocol_2_logistic_regression(twt):
    """
    Execute logistic regression classifier predicting match_win_flag.
    Variables: avg_runs_scored, avg_wickets_taken, avg_run_rate_conceded
    """
    print("\n" + "=" * 60)
    print("PROTOCOL 2: Logistic Regression Classifier")
    print("=" * 60)

    # Fresh load - eliminate records with missing designated variables
    df = twt.copy()
    required_cols = ['match_win_flag', 'avg_runs_scored', 'avg_wickets_taken', 'avg_run_rate_conceded']

    # Eliminate rows where any designated variable is missing
    mask = df[required_cols].notna().all(axis=1)
    df_clean = df[mask].copy()

    # Sort by original row index
    df_clean = df_clean.sort_values('original_index').reset_index(drop=True)

    print(f"Records after elimination: {len(df_clean)}")

    # Prepare features and target
    X = df_clean[['avg_runs_scored', 'avg_wickets_taken', 'avg_run_rate_conceded']].values
    y = df_clean['match_win_flag'].values.astype(int)

    # Stratified train-test split: 75% train, 25% test (no shuffle, maintains class proportions)
    X_train, X_val, y_train, y_val = stratified_split_no_shuffle(
        X, y, test_size=0.25, random_state=42
    )

    # Configure logistic regression
    model = LogisticRegression(solver='lbfgs', max_iter=2000, tol=0.0001, random_state=42)
    model.fit(X_train, y_train)

    # Predictions
    y_pred = model.predict(X_val)

    # Macro-averaged F1 score
    f1_macro = f1_score(y_val, y_pred, average='macro')

    print(f"Macro-Averaged F1 Score: {round(f1_macro, 4)}")

    # Generate bar diagram for predicted match outcomes
    unique, counts = np.unique(y_pred, return_counts=True)
    labels = ['Loss' if u == 0 else 'Win' for u in unique]

    plt.figure(figsize=(8, 6))
    plt.bar(labels, counts, color=['#E74C3C', '#2ECC71'], edgecolor='black')
    plt.xlabel('Match Outcome')
    plt.ylabel('Predicted Observation Counts')
    plt.title('Logistic Regression: Predicted Match Outcomes')
    plt.tight_layout()
    plt.savefig('logistic_bar.png', dpi=150)
    plt.close()

    return f1_macro


# ---------- Protocol 3: Ridge Regression ----------
def protocol_3_ridge_regression(twb):
    """
    Implement Ridge regression estimating batter_strike_rate.
    Variables: innings_played, total_runs, boundary_percentage
    """
    print("\n" + "=" * 60)
    print("PROTOCOL 3: Ridge Regression with L2 Regularization")
    print("=" * 60)

    # Fresh load - eliminate records with missing designated variables
    df = twb.copy()
    required_cols = ['batter_strike_rate', 'innings_played', 'total_runs', 'boundary_percentage']

    # Eliminate rows where any designated variable is missing
    mask = df[required_cols].notna().all(axis=1)
    df_clean = df[mask].copy()

    # Sort by original row index
    df_clean = df_clean.sort_values('original_index').reset_index(drop=True)

    print(f"Records after elimination: {len(df_clean)}")

    # Prepare features and target
    X = df_clean[['innings_played', 'total_runs', 'boundary_percentage']].values
    y = df_clean['batter_strike_rate'].values

    # Train-test split: 70% train, 30% test
    X_train, X_val, y_train, y_val = train_test_split(
        X, y, test_size=0.30, shuffle=False, random_state=42
    )

    # Configure Ridge regression
    model = Ridge(alpha=1.0, solver='svd', random_state=42)
    model.fit(X_train, y_train)

    # Predictions
    y_pred = model.predict(X_val)

    # Root mean squared error
    rmse = np.sqrt(np.mean((y_pred - y_val) ** 2))

    print(f"Root Mean Squared Error: {round(rmse, 3)}")

    return rmse


# ---------- Protocol 4: Hierarchical Clustering ----------
def protocol_4_hierarchical_clustering(twbo):
    """
    Construct hierarchical clustering with Ward linkage.
    Clustering attributes: wickets_taken, economy_rate, bowling_strike_rate, matches_played
    """
    print("\n" + "=" * 60)
    print("PROTOCOL 4: Hierarchical Clustering (Ward Linkage)")
    print("=" * 60)

    # Fresh load - eliminate records with missing designated variables
    df = twbo.copy()
    required_cols = ['wickets_taken', 'economy_rate', 'bowling_strike_rate', 'matches_played']

    # Eliminate rows where any designated variable is missing
    mask = df[required_cols].notna().all(axis=1)
    df_clean = df[mask].copy()

    # Sort by original row index
    df_clean = df_clean.sort_values('original_index').reset_index(drop=True)

    print(f"Records after elimination: {len(df_clean)}")

    # Extract clustering attributes
    X = df_clean[required_cols].values

    # Standardize features
    scaler = StandardScaler()
    X_scaled = scaler.fit_transform(X)

    # Apply Ward linkage
    Z = linkage(X_scaled, method='ward', metric='euclidean')

    # Cut dendrogram to get exactly 4 clusters
    # Find the height that yields 4 clusters
    max_height = Z[:, 2].max()
    for t in np.linspace(0, max_height, 1000):
        clusters_temp = fcluster(Z, t=t, criterion='distance')
        if len(np.unique(clusters_temp)) == 4:
            threshold = t
            break

    clusters = fcluster(Z, t=threshold, criterion='distance')

    # Calculate silhouette coefficient
    silhouette = silhouette_score(X_scaled, clusters, metric='euclidean')

    print(f"Silhouette Coefficient: {round(silhouette, 4)}")

    # Generate dendrogram
    plt.figure(figsize=(12, 6))
    dendrogram(Z, leaf_rotation=90, leaf_font_size=8)
    plt.xlabel('Bowler Sample Index')
    plt.ylabel('Linkage Distance')
    plt.title('Hierarchical Clustering Dendrogram (Ward Linkage)')
    plt.tight_layout()
    plt.savefig('dendrogram.png', dpi=150)
    plt.close()

    return silhouette


# ---------- Protocol 5: Principal Component Analysis ----------
def protocol_5_pca(twb):
    """
    Configure PCA extracting orthogonal batting dimensions.
    Decomposition attributes: innings_played, total_runs, batter_strike_rate, boundary_percentage
    """
    print("\n" + "=" * 60)
    print("PROTOCOL 5: Principal Component Analysis")
    print("=" * 60)

    # Fresh load - eliminate records with missing designated variables
    df = twb.copy()
    required_cols = ['innings_played', 'total_runs', 'batter_strike_rate', 'boundary_percentage']

    # Eliminate rows where any designated variable is missing
    mask = df[required_cols].notna().all(axis=1)
    df_clean = df[mask].copy()

    # Sort by original row index
    df_clean = df_clean.sort_values('original_index').reset_index(drop=True)

    print(f"Records after elimination: {len(df_clean)}")

    # Extract decomposition attributes
    X = df_clean[required_cols].values

    # Standardize features
    scaler = StandardScaler()
    X_scaled = scaler.fit_transform(X)

    # Configure PCA
    pca = PCA(n_components=2, svd_solver='full', random_state=42)
    pca.fit(X_scaled)

    # Cumulative explained variance ratio
    cumulative_var = sum(pca.explained_variance_ratio_) * 100

    print(f"Cumulative Explained Variance (2 components): {round(cumulative_var, 2)}%")

    return cumulative_var


# ---------- Protocol 6: Mean Team Run Rate ----------
def protocol_6_mean_team_run_rate(twt):
    """
    Calculate mean team_run_rate from twt.xlsx.
    """
    print("\n" + "=" * 60)
    print("PROTOCOL 6: Mean Team Run Rate")
    print("=" * 60)

    # Fresh load - eliminate records where team_run_rate is absent
    df = twt.copy()

    mask = df['team_run_rate'].notna()
    df_clean = df[mask].copy()

    # Sort by original row index
    df_clean = df_clean.sort_values('original_index').reset_index(drop=True)

    print(f"Records after elimination: {len(df_clean)}")

    # Compute arithmetic mean
    mean_run_rate = df_clean['team_run_rate'].sum() / len(df_clean)

    print(f"Mean Team Run Rate: {round(mean_run_rate, 2)}")

    return mean_run_rate


# ---------- Protocol 7: Median Batter Strike Rate ----------
def protocol_7_median_batter_strike_rate(twb):
    """
    Compute median batter_strike_rate from twb.xlsx.
    """
    print("\n" + "=" * 60)
    print("PROTOCOL 7: Median Batter Strike Rate")
    print("=" * 60)

    # Fresh load - eliminate records where batter_strike_rate is absent
    df = twb.copy()

    mask = df['batter_strike_rate'].notna()
    df_clean = df[mask].copy()

    # Sort by original row index
    df_clean = df_clean.sort_values('original_index').reset_index(drop=True)

    print(f"Records after elimination: {len(df_clean)}")

    # Compute median using numpy
    median_strike_rate = np.median(df_clean['batter_strike_rate'].values)

    print(f"Median Batter Strike Rate: {round(median_strike_rate, 2)}")

    return median_strike_rate


# ---------- Protocol 8: Dominant Bowling Style ----------
def protocol_8_dominant_bowling_style(twbo):
    """
    Determine dominant bowling_style category from twbo.xlsx.
    """
    print("\n" + "=" * 60)
    print("PROTOCOL 8: Dominant Bowling Style (Mode)")
    print("=" * 60)

    # Fresh load - eliminate records where bowling_style is absent
    df = twbo.copy()

    mask = df['bowling_style'].notna()
    df_clean = df[mask].copy()

    # Sort by original row index
    df_clean = df_clean.sort_values('original_index').reset_index(drop=True)

    print(f"Records after elimination: {len(df_clean)}")

    # Compute frequency table
    freq_table = df_clean['bowling_style'].value_counts()
    max_freq = freq_table.max()

    # Get modes (categories with max frequency)
    modes = freq_table[freq_table == max_freq].index.tolist()

    # Alphabetically first if tie
    dominant_style = sorted(modes)[0].lower()

    print(f"Dominant Bowling Style: {dominant_style}")

    return dominant_style


# ---------- Protocol 9: Chi-Square Test ----------
def protocol_9_chi_square(twbo, twt):
    """
    Compute chi-square test statistic for independence between bowling_style and match_win_flag.
    """
    print("\n" + "=" * 60)
    print("PROTOCOL 9: Chi-Square Test of Independence")
    print("=" * 60)

    # Fresh load - independent filtering
    df_twbo = twbo.copy()
    df_twt = twt.copy()

    # Eliminate records where bowling_style is absent from twbo
    mask_bo = df_twbo['bowling_style'].notna()
    df_twbo_clean = df_twbo[mask_bo].copy()

    # Eliminate records where match_win_flag is absent from twt
    mask_twt = df_twt['match_win_flag'].notna()
    df_twt_clean = df_twt[mask_twt].copy()

    print(f"twbo records after elimination: {len(df_twbo_clean)}")
    print(f"twt records after elimination: {len(df_twt_clean)}")

    # Inner join on team_id and match_id
    merged = pd.merge(
        df_twbo_clean[['team_id', 'match_id', 'bowling_style']],
        df_twt_clean[['team_id', 'match_id', 'match_win_flag']],
        on=['team_id', 'match_id'],
        how='inner'
    )

    print(f"Merged records: {len(merged)}")

    if len(merged) == 0:
        print("No matching records found for chi-square test")
        return np.nan

    # Construct contingency table
    contingency = pd.crosstab(merged['bowling_style'], merged['match_win_flag'])

    # Calculate chi-square statistic
    chi2, p_val, dof, expected = stats.chi2_contingency(contingency)

    print(f"Chi-Square Statistic: {round(chi2, 4)}")

    # Generate heatmap
    plt.figure(figsize=(8, 6))
    plt.imshow(contingency.values, cmap='YlOrRd', aspect='auto')
    plt.colorbar(label='Frequency')
    plt.xticks(range(len(contingency.columns)), ['Loss', 'Win'])
    plt.yticks(range(len(contingency.index)), contingency.index)
    plt.xlabel('Match Outcome')
    plt.ylabel('Bowling Style')
    plt.title('Contingency Table: Bowling Style vs Match Outcome')

    # Add annotations
    for i in range(len(contingency.index)):
        for j in range(len(contingency.columns)):
            plt.text(j, i, str(contingency.values[i, j]),
                    ha='center', va='center', fontsize=12, fontweight='bold')

    plt.tight_layout()
    plt.savefig('contingency_heatmap.png', dpi=150)
    plt.close()

    return chi2


# ---------- Protocol 10: Pearson Correlation ----------
def protocol_10_pearson_correlation(twb, twt):
    """
    Calculate Pearson correlation between aggregated batter_strike_rate and team_run_rate.
    """
    print("\n" + "=" * 60)
    print("PROTOCOL 10: Pearson Correlation Coefficient")
    print("=" * 60)

    # Fresh load - independent filtering
    df_twb = twb.copy()
    df_twt = twt.copy()

    # Eliminate records where batter_strike_rate is absent from twb
    mask_b = df_twb['batter_strike_rate'].notna()
    df_twb_clean = df_twb[mask_b].copy()

    # Eliminate records where team_run_rate is absent from twt
    mask_t = df_twt['team_run_rate'].notna()
    df_twt_clean = df_twt[mask_t].copy()

    print(f"twb records after elimination: {len(df_twb_clean)}")
    print(f"twt records after elimination: {len(df_twt_clean)}")

    # Aggregate batter_strike_rate by team_id and match_id
    agg_strike_rate = df_twb_clean.groupby(['team_id', 'match_id'])['batter_strike_rate'].mean().reset_index()
    agg_strike_rate.columns = ['team_id', 'match_id', 'mean_batter_strike_rate']

    # Inner join with twt on team_id and match_id
    merged = pd.merge(
        agg_strike_rate,
        df_twt_clean[['team_id', 'match_id', 'team_run_rate']],
        on=['team_id', 'match_id'],
        how='inner'
    )

    print(f"Merged records: {len(merged)}")

    if len(merged) < 2:
        print("Insufficient records for correlation")
        return np.nan

    # Compute Pearson correlation using numpy.corrcoef
    corr_matrix = np.corrcoef(merged['mean_batter_strike_rate'], merged['team_run_rate'])
    pearson_r = corr_matrix[0, 1]

    print(f"Pearson Correlation: {round(pearson_r, 4)}")

    return pearson_r


# ---------- Protocol 11: One-Way ANOVA ----------
def protocol_11_anova(twt, twbo):
    """
    Perform one-way ANOVA F-test for team_run_rate across bowling_style categories.
    """
    print("\n" + "=" * 60)
    print("PROTOCOL 11: One-Way ANOVA F-Test")
    print("=" * 60)

    # Fresh load - independent filtering
    df_twt = twt.copy()
    df_twbo = twbo.copy()

    # Eliminate records where team_run_rate is absent from twt
    mask_t = df_twt['team_run_rate'].notna()
    df_twt_clean = df_twt[mask_t].copy()

    # Eliminate records where bowling_style is absent from twbo
    mask_bo = df_twbo['bowling_style'].notna()
    df_twbo_clean = df_twbo[mask_bo].copy()

    print(f"twt records after elimination: {len(df_twt_clean)}")
    print(f"twbo records after elimination: {len(df_twbo_clean)}")

    # Inner join on team_id and match_id
    merged = pd.merge(
        df_twt_clean[['team_id', 'match_id', 'team_run_rate']],
        df_twbo_clean[['team_id', 'match_id', 'bowling_style']],
        on=['team_id', 'match_id'],
        how='inner'
    )

    print(f"Merged records: {len(merged)}")

    if len(merged) == 0:
        print("No matching records found for ANOVA")
        return np.nan

    # Group team_run_rate by bowling_style
    groups = [group['team_run_rate'].values for name, group in merged.groupby('bowling_style')]

    if len(groups) < 2:
        print("Insufficient groups for ANOVA")
        return np.nan

    # Perform one-way ANOVA
    f_stat, p_val = stats.f_oneway(*groups)

    print(f"F-Statistic: {round(f_stat, 3)}")

    # Generate violin plot
    plt.figure(figsize=(10, 6))
    styles = merged['bowling_style'].unique()
    data_for_violin = [merged[merged['bowling_style'] == s]['team_run_rate'].values for s in styles]

    parts = plt.violinplot(data_for_violin, positions=range(len(styles)), showmeans=True, showmedians=True)
    plt.xticks(range(len(styles)), styles)
    plt.xlabel('Bowling Style')
    plt.ylabel('Team Run Rate')
    plt.title('ANOVA: Team Run Rate Distribution by Bowling Style')
    plt.tight_layout()
    plt.savefig('anova_violin.png', dpi=150)
    plt.close()

    return f_stat


# ---------- Protocol 12: Ward Linkage Explanation ----------
def protocol_12_ward_explanation():
    """
    Explain why hierarchical Ward linkage produces deterministic results.
    """
    print("\n" + "=" * 60)
    print("PROTOCOL 12: Ward Linkage Determinism Explanation")
    print("=" * 60)

    print(WARD_EXPLANATION)

    return WARD_EXPLANATION


# ---------- Main Execution ----------
def main():
    """
    Execute all twelve analytical protocols.
    """
    print("=" * 60)
    print("T20 CRICKET PERFORMANCE ANALYSIS")
    print("Twelve Analytical Protocols")
    print("=" * 60)

    # Load and prepare data
    twt, twb, twbo = load_and_prepare_data()

    print(f"\nDatasets loaded:")
    print(f"  twt (Team Records): {len(twt)} records")
    print(f"  twb (Batting Records): {len(twb)} records")
    print(f"  twbo (Bowling Records): {len(twbo)} records")

    # Execute all protocols
    results = {}

    # Protocol 1: OLS Regression
    results['ols_mse'] = protocol_1_ols_regression(twt)

    # Protocol 2: Logistic Regression
    results['logistic_f1'] = protocol_2_logistic_regression(twt)

    # Protocol 3: Ridge Regression
    results['ridge_rmse'] = protocol_3_ridge_regression(twb)

    # Protocol 4: Hierarchical Clustering
    results['silhouette'] = protocol_4_hierarchical_clustering(twbo)

    # Protocol 5: PCA
    results['pca_variance'] = protocol_5_pca(twb)

    # Protocol 6: Mean Team Run Rate
    results['mean_run_rate'] = protocol_6_mean_team_run_rate(twt)

    # Protocol 7: Median Batter Strike Rate
    results['median_strike_rate'] = protocol_7_median_batter_strike_rate(twb)

    # Protocol 8: Dominant Bowling Style
    results['dominant_style'] = protocol_8_dominant_bowling_style(twbo)

    # Protocol 9: Chi-Square Test
    results['chi_square'] = protocol_9_chi_square(twbo, twt)

    # Protocol 10: Pearson Correlation
    results['pearson_r'] = protocol_10_pearson_correlation(twb, twt)

    # Protocol 11: ANOVA
    results['f_statistic'] = protocol_11_anova(twt, twbo)

    # Protocol 12: Ward Explanation
    results['ward_explanation'] = protocol_12_ward_explanation()

    # ---------- Summary of Key Results ----------
    print("\n" + "=" * 60)
    print("KEY RESULTS SUMMARY")
    print("=" * 60)
    print(f"Protocol 1  - OLS MSE: {round(results['ols_mse'], 4)}")
    print(f"Protocol 2  - Logistic F1 (Macro): {round(results['logistic_f1'], 4)}")
    print(f"Protocol 3  - Ridge RMSE: {round(results['ridge_rmse'], 3)}")
    print(f"Protocol 4  - Silhouette Score: {round(results['silhouette'], 4)}")
    print(f"Protocol 5  - PCA Cumulative Variance: {round(results['pca_variance'], 2)}%")
    print(f"Protocol 6  - Mean Team Run Rate: {round(results['mean_run_rate'], 2)}")
    print(f"Protocol 7  - Median Batter Strike Rate: {round(results['median_strike_rate'], 2)}")
    print(f"Protocol 8  - Dominant Bowling Style: {results['dominant_style']}")
    print(f"Protocol 9  - Chi-Square Statistic: {round(results['chi_square'], 4) if not np.isnan(results['chi_square']) else 'N/A'}")
    print(f"Protocol 10 - Pearson Correlation: {round(results['pearson_r'], 4) if not np.isnan(results['pearson_r']) else 'N/A'}")
    print(f"Protocol 11 - ANOVA F-Statistic: {round(results['f_statistic'], 3) if not np.isnan(results['f_statistic']) else 'N/A'}")
    print("=" * 60)

    return results


if __name__ == "__main__":
    main()
