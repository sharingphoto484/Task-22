# ==========================================
# NBA Playoff Roster Analysis Script - Fresh Implementation
# ==========================================
# Requirements: pandas, numpy, matplotlib, scipy, scikit-learn
# Input files: nba.csv, Regular_Season.csv, Playoffs.csv (in same directory)
# ==========================================

import pandas as pd
import numpy as np
import matplotlib.pyplot as plt
import seaborn as sns
from sklearn.linear_model import LogisticRegression
from sklearn.decomposition import PCA
from sklearn.metrics import roc_auc_score
from scipy.stats import ttest_ind
import hashlib
import warnings
warnings.filterwarnings('ignore')

print("="*70)
print("FRESH NBA PLAYOFF ANALYSIS - INDEPENDENT VERIFICATION")
print("="*70)

# ---------- Load CSVs ----------
print("\n1. Loading CSV files...")
df_nba = pd.read_csv('nba.csv')
df_regular = pd.read_csv('Regular_Season.csv')
df_playoffs = pd.read_csv('Playoffs.csv')

print(f"   nba.csv: {len(df_nba)} rows")
print(f"   Regular_Season.csv: {len(df_regular)} rows")
print(f"   Playoffs.csv: {len(df_playoffs)} rows")

# ---------- Normalize Season_type ----------
print("\n2. Normalizing Season_type...")
df_nba['Season_type'] = df_nba['Season_type'].replace('Regular%20Season', 'Regular_Season')
df_regular['Season_type'] = 'Regular_Season'
df_playoffs['Season_type'] = 'Playoffs'

# ---------- Find Intersected Seasons ----------
print("\n3. Finding seasons present in ALL three files...")
years_nba = set(df_nba['year'].unique())
years_regular = set(df_regular['year'].unique())
years_playoffs = set(df_playoffs['year'].unique())
common_years = sorted(years_nba & years_regular & years_playoffs)

print(f"   Common seasons: {common_years}")
print(f"   Number of seasons: {len(common_years)}")

# ---------- Prepare Regular Season Data ----------
print("\n4. Preparing Regular Season modeling data...")

# Combine all data and filter to common years
all_data = pd.concat([df_nba, df_regular, df_playoffs], ignore_index=True)
all_data = all_data[all_data['year'].isin(common_years)].copy()

# Get Regular Season and Playoffs subsets
regular_season = all_data[all_data['Season_type'] == 'Regular_Season'].copy()
playoffs = all_data[all_data['Season_type'] == 'Playoffs'].copy()

# Filter to non-missing required fields
required_fields = ['year', 'PLAYER_ID', 'PLAYER', 'TEAM', 'GP', 'MIN', 'PTS']
regular_season = regular_season.dropna(subset=required_fields).copy()

print(f"   Regular season rows (after filtering): {len(regular_season)}")

# ---------- Create Playoff Participation Label ----------
print("\n5. Creating playoff participation labels...")

# Build set of (year, PLAYER_ID, TEAM) that appeared in playoffs
playoff_keys = set()
for _, row in playoffs.iterrows():
    key = (row['year'], str(row['PLAYER_ID']), row['TEAM'])
    playoff_keys.add(key)

print(f"   Unique playoff appearances: {len(playoff_keys)}")

# Label regular season rows
def check_playoff_participation(row):
    key = (row['year'], str(row['PLAYER_ID']), row['TEAM'])
    return 1 if key in playoff_keys else 0

regular_season['made_playoffs'] = regular_season.apply(check_playoff_participation, axis=1)

print(f"   Players who made playoffs: {regular_season['made_playoffs'].sum()}")
print(f"   Players who did not: {len(regular_season) - regular_season['made_playoffs'].sum()}")
print(f"   Playoff rate: {regular_season['made_playoffs'].mean():.4f}")

# ---------- Standardize Predictors Within Year ----------
print("\n6. Standardizing GP, MIN, PTS within each year...")

def standardize_within_year(df, col):
    """Standardize column within each year group"""
    result = df.groupby('year')[col].transform(
        lambda x: (x - x.mean()) / x.std() if x.std() > 0 else 0
    )
    return result

regular_season['GP_std'] = standardize_within_year(regular_season, 'GP')
regular_season['MIN_std'] = standardize_within_year(regular_season, 'MIN')
regular_season['PTS_std'] = standardize_within_year(regular_season, 'PTS')

# Verify standardization
print(f"   GP_std mean: {regular_season['GP_std'].mean():.6f} (should be ~0)")
print(f"   MIN_std mean: {regular_season['MIN_std'].mean():.6f} (should be ~0)")
print(f"   PTS_std mean: {regular_season['PTS_std'].mean():.6f} (should be ~0)")

# ---------- Assign Folds by Hashing PLAYER_ID Modulo 5 ----------
print("\n7. Assigning cross-validation folds...")

def assign_fold(player_id):
    """Hash PLAYER_ID modulo 5"""
    hash_obj = hashlib.md5(str(player_id).encode())
    hash_int = int(hash_obj.hexdigest(), 16)
    return hash_int % 5

regular_season['fold'] = regular_season['PLAYER_ID'].apply(assign_fold)

for fold_num in range(5):
    count = (regular_season['fold'] == fold_num).sum()
    print(f"   Fold {fold_num}: {count} observations")

# ---------- Cross-Validated Logistic Regression ----------
print("\n8. Performing 5-fold cross-validated logistic regression...")

feature_cols = ['GP_std', 'MIN_std', 'PTS_std']
X = regular_season[feature_cols].values
y = regular_season['made_playoffs'].values
folds = regular_season['fold'].values

# Initialize out-of-fold predictions
oof_preds = np.zeros(len(regular_season))
fold_coefficients = []

for fold_id in range(5):
    # Split into train and validation
    train_idx = folds != fold_id
    val_idx = folds == fold_id

    X_train, y_train = X[train_idx], y[train_idx]
    X_val = X[val_idx]

    # Fit logistic regression
    model = LogisticRegression(max_iter=1000, random_state=42)
    model.fit(X_train, y_train)

    # Store out-of-fold predictions
    oof_preds[val_idx] = model.predict_proba(X_val)[:, 1]

    # Store coefficients
    fold_coefficients.append(model.coef_[0])

    print(f"   Fold {fold_id}: Coefs = {model.coef_[0]}")

# Calculate cross-validated AUC
cv_auc = roc_auc_score(y, oof_preds)

# Calculate mean coefficient on MIN (index 1)
mean_coef_min = np.mean([coef[1] for coef in fold_coefficients])

print(f"\n   Cross-Validated AUC: {cv_auc:.4f}")
print(f"   Mean MIN coefficient across folds: {mean_coef_min:.4f}")

# Alternative: Fit on full data to get single coefficient
print("\n   Fitting single model on all data for coefficient reporting...")
model_full = LogisticRegression(max_iter=1000, random_state=42)
model_full.fit(X, y)
full_coef_min = model_full.coef_[0][1]
print(f"   MIN coefficient (full data): {full_coef_min:.4f}")

# ---------- PCA Analysis ----------
print("\n9. Performing PCA on standardized GP, MIN, PTS...")

X_pca = regular_season[['GP_std', 'MIN_std', 'PTS_std']].values
y_pca = regular_season['made_playoffs'].values

# Fit PCA
pca = PCA(n_components=3)
pca_all = pca.fit_transform(X_pca)

# Extract first component
pc1_scores = pca_all[:, 0]
pc1_loadings = pca.components_[0]

print(f"   PC1 loadings (before sign correction): {pc1_loadings}")

# Ensure loading on MIN is positive
if pc1_loadings[1] < 0:
    pc1_loadings = -pc1_loadings
    pc1_scores = -pc1_scores

print(f"   PC1 loadings (after sign correction): {pc1_loadings}")
print(f"   Variance explained by PC1: {pca.explained_variance_ratio_[0] * 100:.4f}%")

# Fit logistic regression with PC1 score
pc1_model = LogisticRegression(max_iter=1000, random_state=42)
pc1_model.fit(pc1_scores.reshape(-1, 1), y_pca)
pc1_preds = pc1_model.predict_proba(pc1_scores.reshape(-1, 1))[:, 1]
pc1_auc = roc_auc_score(y_pca, pc1_preds)

print(f"   AUC for PC1 score model: {pc1_auc:.4f}")

# ---------- Roster Continuity Analysis ----------
print("\n10. Analyzing roster continuity...")

# Get playoff data from nba.csv only
nba_playoff = df_nba[df_nba['Season_type'] == 'Playoffs'].copy()
nba_regular = df_nba[df_nba['Season_type'] == 'Regular_Season'].copy()

# Filter to common years
nba_playoff = nba_playoff[nba_playoff['year'].isin(common_years)]
nba_regular = nba_regular[nba_regular['year'].isin(common_years)]

continuity_ratios = []
team_year_continuity = []

for year in common_years:
    year_playoff = nba_playoff[nba_playoff['year'] == year]
    year_regular = nba_regular[nba_regular['year'] == year]

    playoff_teams = year_playoff['TEAM'].unique()

    for team in playoff_teams:
        team_playoff = year_playoff[year_playoff['TEAM'] == team]
        team_regular = year_regular[year_regular['TEAM'] == team]

        # Skip if no MIN data
        if team_playoff['MIN'].isna().all():
            continue

        # Get players in regular season for this team
        regular_players = set(team_regular['PLAYER_ID'].dropna().astype(str))

        # Calculate continuity
        team_playoff_clean = team_playoff.dropna(subset=['MIN', 'PLAYER_ID'])

        continued_min = 0
        total_min = 0

        for _, prow in team_playoff_clean.iterrows():
            player_min = prow['MIN']
            total_min += player_min

            if str(prow['PLAYER_ID']) in regular_players:
                continued_min += player_min

        if total_min > 0:
            ratio = continued_min / total_min
            continuity_ratios.append(ratio)
            team_year_continuity.append({
                'year': year,
                'team': team,
                'ratio': ratio
            })

avg_continuity = np.mean(continuity_ratios)
print(f"   Average continuity ratio: {avg_continuity:.4f}")
print(f"   Number of team-seasons: {len(continuity_ratios)}")

# ---------- Gini Index Calculation ----------
print("\n11. Calculating Gini index for playoff MIN concentration...")

def gini_coefficient(values):
    """Calculate Gini coefficient"""
    values = np.array(values, dtype=float)
    values = values[~np.isnan(values)]
    if len(values) == 0 or values.sum() == 0:
        return 0.0
    sorted_vals = np.sort(values)
    n = len(sorted_vals)
    index = np.arange(1, n + 1)
    return (2.0 * np.sum(index * sorted_vals)) / (n * np.sum(sorted_vals)) - (n + 1.0) / n

gini_values = []

for year in common_years:
    year_playoff = nba_playoff[nba_playoff['year'] == year]
    playoff_teams = year_playoff['TEAM'].unique()

    for team in playoff_teams:
        team_playoff = year_playoff[year_playoff['TEAM'] == team]
        min_vals = team_playoff['MIN'].dropna().values

        if len(min_vals) > 0:
            gini = gini_coefficient(min_vals)
            gini_values.append(gini)

avg_gini = np.mean(gini_values)
print(f"   Average Gini index: {avg_gini:.4f}")

# ---------- Heatmap and Maximum Continuity ----------
print("\n12. Creating heatmap and finding maximum continuity...")

continuity_df = pd.DataFrame(team_year_continuity)
max_continuity = continuity_df['ratio'].max()

print(f"   Maximum continuity ratio: {max_continuity:.4f}")

# Create heatmap
pivot_table = continuity_df.pivot(index='team', columns='year', values='ratio')
pivot_table = pivot_table.sort_index(axis=1).sort_index(axis=0)

plt.figure(figsize=(14, 10))
sns.heatmap(pivot_table, annot=False, cmap='YlOrRd', cbar_kws={'label': 'Continuity Ratio'})
plt.title('Roster Continuity Heatmap')
plt.xlabel('Year')
plt.ylabel('Team')
plt.tight_layout()
plt.savefig('heatmap_fresh.png', dpi=300, bbox_inches='tight')
print(f"   Heatmap saved as 'heatmap_fresh.png'")

# ---------- Recent Stability Test ----------
print("\n13. Testing for recent stability...")

verdict = 0

if len(common_years) >= 6:
    sorted_years = sorted(common_years)
    recent_3_years = sorted_years[-3:]
    prior_3_years = sorted_years[-6:-3]

    recent_ratios = continuity_df[continuity_df['year'].isin(recent_3_years)]['ratio'].values
    prior_ratios = continuity_df[continuity_df['year'].isin(prior_3_years)]['ratio'].values

    print(f"   Recent 3 years: {recent_3_years}")
    print(f"   Prior 3 years: {prior_3_years}")
    print(f"   Recent mean: {np.mean(recent_ratios):.6f}")
    print(f"   Prior mean: {np.mean(prior_ratios):.6f}")

    # Two-sided Welch t-test
    t_stat, p_value = ttest_ind(recent_ratios, prior_ratios, equal_var=False)

    print(f"   t-statistic: {t_stat:.6f}")
    print(f"   p-value: {p_value:.6f}")

    # Check if recent > prior AND significant at 5% level
    if np.mean(recent_ratios) > np.mean(prior_ratios) and p_value < 0.05:
        verdict = 1
else:
    print(f"   Fewer than 6 seasons, verdict = 0 by rule")

print(f"   Stability verdict: {verdict}")

# ---------- FINAL SUMMARY ----------
print("\n" + "="*70)
print("FINAL KEY OUTPUTS")
print("="*70)
print(f"Cross-Validated AUC: {cv_auc:.4f}")
print(f"Standardized Coefficient on MIN (mean across folds): {mean_coef_min:.4f}")
print(f"Standardized Coefficient on MIN (full data): {full_coef_min:.4f}")
print(f"Variance Explained by First PC: {pca.explained_variance_ratio_[0] * 100:.4f}%")
print(f"AUC for Single Score Model: {pc1_auc:.4f}")
print(f"Average Continuity Ratio: {avg_continuity:.4f}")
print(f"Average Gini Index: {avg_gini:.4f}")
print(f"Highest Continuity Ratio: {max_continuity:.4f}")
print(f"Stability Test Verdict: {verdict}")
print("="*70)
