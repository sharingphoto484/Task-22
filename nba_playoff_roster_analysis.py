# ==========================================
# NBA Playoff Roster Analysis Script
# ==========================================
# Requirements: pandas, numpy, matplotlib, scipy, scikit-learn
# Input files: nba.csv, Regular_Season.csv, Playoffs.csv (in same directory)
# Output files: roster_continuity_heatmap.png
# Key outputs: Printed to console
# ==========================================

import pandas as pd
import numpy as np
import matplotlib.pyplot as plt
import seaborn as sns
from sklearn.linear_model import LogisticRegression
from sklearn.decomposition import PCA
from sklearn.metrics import roc_auc_score, roc_curve
from scipy.stats import ttest_ind
import hashlib
import warnings
warnings.filterwarnings('ignore')

# ---------- Load CSVs Robustly ----------
nba = pd.read_csv('nba.csv')
regular = pd.read_csv('Regular_Season.csv')
playoffs = pd.read_csv('Playoffs.csv')

# ---------- Normalize Season_type ----------
nba['Season_type'] = nba['Season_type'].str.replace('Regular%20Season', 'Regular_Season')
nba['Season_type'] = nba['Season_type'].apply(lambda x: 'Playoffs' if x == 'Playoffs' else x)

regular['Season_type'] = 'Regular_Season'
playoffs['Season_type'] = 'Playoffs'

# ---------- Combine All Data ----------
all_data = pd.concat([nba, regular, playoffs], ignore_index=True)
all_data = all_data.drop_duplicates()

# ---------- Intersect Seasons Across All Three Files ----------
seasons_nba = set(nba['year'].unique())
seasons_regular = set(regular['year'].unique())
seasons_playoffs = set(playoffs['year'].unique())
common_seasons = sorted(seasons_nba & seasons_regular & seasons_playoffs)

all_data = all_data[all_data['year'].isin(common_seasons)].copy()

# ---------- Prepare Regular Season Data for Modeling ----------
regular_data = all_data[all_data['Season_type'] == 'Regular_Season'].copy()
playoffs_data = all_data[all_data['Season_type'] == 'Playoffs'].copy()

# Filter to non-missing required fields for regular season modeling
required_cols = ['year', 'PLAYER_ID', 'PLAYER', 'TEAM', 'GP', 'MIN', 'PTS']
regular_data = regular_data.dropna(subset=required_cols)

# ---------- Create Playoff Participation Label ----------
# Create set of player-team-year combinations that appeared in playoffs
playoffs_appearances = set(
    playoffs_data[['year', 'PLAYER_ID', 'TEAM']].apply(
        lambda x: (x['year'], str(x['PLAYER_ID']), x['TEAM']), axis=1
    )
)

# Label regular season rows
regular_data['playoff_label'] = regular_data.apply(
    lambda x: 1 if (x['year'], str(x['PLAYER_ID']), x['TEAM']) in playoffs_appearances else 0,
    axis=1
)

# ---------- Standardize Predictors Within Year ----------
regular_data['GP_std'] = regular_data.groupby('year')['GP'].transform(
    lambda x: (x - x.mean()) / x.std() if x.std() > 0 else 0
)
regular_data['MIN_std'] = regular_data.groupby('year')['MIN'].transform(
    lambda x: (x - x.mean()) / x.std() if x.std() > 0 else 0
)
regular_data['PTS_std'] = regular_data.groupby('year')['PTS'].transform(
    lambda x: (x - x.mean()) / x.std() if x.std() > 0 else 0
)

# ---------- Assign Folds by Hashing PLAYER_ID ----------
def hash_player_id(player_id):
    """Hash PLAYER_ID modulo 5 to assign fold"""
    return int(hashlib.md5(str(player_id).encode()).hexdigest(), 16) % 5

regular_data['fold'] = regular_data['PLAYER_ID'].apply(hash_player_id)

# ---------- Create Sample Weights ----------
# Use MIN as primary weight, else GP when MIN is missing, else count of player rows (1)
regular_data['weight'] = regular_data['MIN'].fillna(regular_data['GP']).fillna(1)

# ---------- Cross-Validated Logistic Regression ----------
X_cols = ['GP_std', 'MIN_std', 'PTS_std']
regular_data_clean = regular_data.dropna(subset=X_cols + ['playoff_label']).copy()
regular_data_clean.reset_index(drop=True, inplace=True)

oof_predictions = np.zeros(len(regular_data_clean))
coefficients = []

for fold in range(5):
    train_mask = regular_data_clean['fold'] != fold
    val_mask = regular_data_clean['fold'] == fold

    X_train = regular_data_clean.loc[train_mask, X_cols]
    y_train = regular_data_clean.loc[train_mask, 'playoff_label']
    w_train = regular_data_clean.loc[train_mask, 'weight']
    X_val = regular_data_clean.loc[val_mask, X_cols]

    lr = LogisticRegression(max_iter=1000, random_state=42)
    lr.fit(X_train, y_train, sample_weight=w_train)

    oof_predictions[val_mask] = lr.predict_proba(X_val)[:, 1]
    coefficients.append(lr.coef_[0])

# Calculate cross-validated AUC
y_true = regular_data_clean['playoff_label'].values
weights = regular_data_clean['weight'].values
cv_auc = roc_auc_score(y_true, oof_predictions, sample_weight=weights)

# Average coefficient on MIN_std
avg_coef_min = np.mean([c[1] for c in coefficients])  # MIN_std is second column

# ---------- PCA Analysis on Regular Season Data ----------
pca_data = regular_data[['GP_std', 'MIN_std', 'PTS_std', 'playoff_label', 'weight']].dropna()

X_pca = pca_data[['GP_std', 'MIN_std', 'PTS_std']].values
y_pca = pca_data['playoff_label'].values
w_pca = pca_data['weight'].values

# Fit PCA
pca = PCA(n_components=1)
pca_scores = pca.fit_transform(X_pca)

# Constrain first component to unit norm (already done by sklearn)
# Ensure loading on MIN is positive
loadings = pca.components_[0]
if loadings[1] < 0:  # MIN is second component
    loadings = -loadings
    pca_scores = -pca_scores

# Variance explained
variance_explained_pct = pca.explained_variance_ratio_[0] * 100

# Fit logistic regression with single PC score
lr_pca = LogisticRegression(max_iter=1000, random_state=42)
lr_pca.fit(pca_scores, y_pca, sample_weight=w_pca)
pca_predictions = lr_pca.predict_proba(pca_scores)[:, 1]
pca_auc = roc_auc_score(y_pca, pca_predictions, sample_weight=w_pca)

# ---------- Roster Continuity Analysis ----------

# Get playoff team-seasons from nba.csv
nba_playoffs = nba[nba['Season_type'] == 'Playoffs'].copy()
nba_regular = nba[nba['Season_type'] == 'Regular_Season'].copy()

# Filter to common seasons
nba_playoffs = nba_playoffs[nba_playoffs['year'].isin(common_seasons)]
nba_regular = nba_regular[nba_regular['year'].isin(common_seasons)]

# For each playoff team-season, calculate continuity ratio
continuity_ratios = []
team_season_records = []

for year in common_seasons:
    year_playoffs = nba_playoffs[nba_playoffs['year'] == year]
    year_regular = nba_regular[nba_regular['year'] == year]

    teams_in_playoffs = year_playoffs['TEAM'].unique()

    for team in teams_in_playoffs:
        team_playoff = year_playoffs[year_playoffs['TEAM'] == team].copy()
        team_regular = year_regular[year_regular['TEAM'] == team].copy()

        # Skip if no MIN data
        if team_playoff['MIN'].isna().all():
            continue

        # Get players who played in both regular season and playoffs for this team
        regular_players = set(team_regular['PLAYER_ID'].unique())

        # Calculate MIN for players who also played regular season
        team_playoff['in_regular'] = team_playoff['PLAYER_ID'].isin(regular_players)

        continued_min = team_playoff[team_playoff['in_regular']]['MIN'].sum()
        total_playoff_min = team_playoff['MIN'].sum()

        if total_playoff_min > 0:
            ratio = continued_min / total_playoff_min
            continuity_ratios.append(ratio)
            team_season_records.append({
                'year': year,
                'team': team,
                'continuity_ratio': ratio
            })

# Average continuity ratio
avg_continuity = np.mean(continuity_ratios)

# ---------- Gini Index Calculation ----------

def gini_index(values):
    """Calculate Gini coefficient"""
    values = np.array(values)
    values = values[~np.isnan(values)]
    if len(values) == 0 or values.sum() == 0:
        return 0
    sorted_values = np.sort(values)
    n = len(sorted_values)
    index = np.arange(1, n + 1)
    return (2 * np.sum(index * sorted_values)) / (n * np.sum(sorted_values)) - (n + 1) / n

gini_values = []

for year in common_seasons:
    year_playoffs = nba_playoffs[nba_playoffs['year'] == year]
    teams_in_playoffs = year_playoffs['TEAM'].unique()

    for team in teams_in_playoffs:
        team_playoff = year_playoffs[year_playoffs['TEAM'] == team]
        min_values = team_playoff['MIN'].dropna().values

        if len(min_values) > 0:
            gini = gini_index(min_values)
            gini_values.append(gini)

avg_gini = np.mean(gini_values)

# ---------- Create Heatmap of Roster Continuity ----------

# Create pivot table for heatmap
continuity_df = pd.DataFrame(team_season_records)
heatmap_pivot = continuity_df.pivot(index='team', columns='year', values='continuity_ratio')

# Sort by year (columns) and teams (rows)
heatmap_pivot = heatmap_pivot.sort_index(axis=1)  # Sort columns (years)
heatmap_pivot = heatmap_pivot.sort_index(axis=0)  # Sort rows (teams)

# Get maximum continuity ratio
max_continuity = continuity_df['continuity_ratio'].max()

# Create heatmap
plt.figure(figsize=(14, 10))
sns.heatmap(heatmap_pivot, annot=False, cmap='YlOrRd', cbar_kws={'label': 'Continuity Ratio'})
plt.title('Roster Continuity Heatmap by Team and Year')
plt.xlabel('Year')
plt.ylabel('Team')
plt.tight_layout()
plt.savefig('roster_continuity_heatmap.png', dpi=300, bbox_inches='tight')
plt.close()

# ---------- Test for Recent Stability ----------

if len(common_seasons) >= 6:
    # Sort seasons
    sorted_seasons = sorted(common_seasons)
    recent_3 = sorted_seasons[-3:]
    prior_3 = sorted_seasons[-6:-3]

    # Get continuity ratios for recent and prior periods
    recent_ratios = continuity_df[continuity_df['year'].isin(recent_3)]['continuity_ratio'].values
    prior_ratios = continuity_df[continuity_df['year'].isin(prior_3)]['continuity_ratio'].values

    # Perform two-sided Welch t-test
    t_stat, p_value = ttest_ind(recent_ratios, prior_ratios, equal_var=False)

    # Check if recent mean is significantly larger at 5% level
    recent_mean = np.mean(recent_ratios)
    prior_mean = np.mean(prior_ratios)

    # For a two-sided test, we check if recent > prior AND p-value < 0.05
    # But we need to ensure we're testing the right direction
    if recent_mean > prior_mean and p_value < 0.05:
        stability_verdict = 1
    else:
        stability_verdict = 0
else:
    stability_verdict = 0

# ---------- Final Output ----------
print(f"Cross-Validated AUC:             {cv_auc:.4f}")
print(f"Standardized Coefficient on MIN: {avg_coef_min:.4f}")
print(f"Variance Explained by First PC:  {variance_explained_pct:.4f}%")
print(f"AUC for Single Score Model:      {pca_auc:.4f}")
print(f"Average Continuity Ratio:        {avg_continuity:.4f}")
print(f"Average Gini Index:              {avg_gini:.4f}")
print(f"Highest Continuity Ratio:        {max_continuity:.4f}")
print(f"Stability Test Verdict:          {stability_verdict}")
