import pandas as pd
import numpy as np
from sklearn.linear_model import LogisticRegression
from sklearn.metrics import roc_auc_score

# Load data
print("Loading data...")
nba = pd.read_csv('nba.csv')
regular = pd.read_csv('Regular_Season.csv')
playoffs = pd.read_csv('Playoffs.csv')

# Normalize Season_type
nba['Season_type'] = nba['Season_type'].str.replace('Regular%20Season', 'Regular_Season')
regular['Season_type'] = 'Regular_Season'
playoffs['Season_type'] = 'Playoffs'

# Get common seasons
seasons_nba = set(nba['year'].unique())
seasons_regular = set(regular['year'].unique())
seasons_playoffs = set(playoffs['year'].unique())
common_seasons = sorted(seasons_nba & seasons_regular & seasons_playoffs)

print(f"\nCommon seasons: {common_seasons}")
print(f"Number of common seasons: {len(common_seasons)}")

# Check correlation in regular season data
all_data = pd.concat([nba, regular, playoffs], ignore_index=True)
all_data = all_data[all_data['year'].isin(common_seasons)].copy()
regular_data = all_data[all_data['Season_type'] == 'Regular_Season'].copy()

required_cols = ['year', 'PLAYER_ID', 'PLAYER', 'TEAM', 'GP', 'MIN', 'PTS']
regular_data = regular_data.dropna(subset=required_cols)

print(f"\nRegular season rows: {len(regular_data)}")
print(f"\nCorrelation matrix for GP, MIN, PTS:")
print(regular_data[['GP', 'MIN', 'PTS']].corr())

# Check playoff participation rates
playoffs_data = all_data[all_data['Season_type'] == 'Playoffs'].copy()
playoffs_appearances = set(
    playoffs_data[['year', 'PLAYER_ID', 'TEAM']].apply(
        lambda x: (x['year'], str(x['PLAYER_ID']), x['TEAM']), axis=1
    )
)

regular_data['playoff_label'] = regular_data.apply(
    lambda x: 1 if (x['year'], str(x['PLAYER_ID']), x['TEAM']) in playoffs_appearances else 0,
    axis=1
)

print(f"\nPlayoff participation rate: {regular_data['playoff_label'].mean():.4f}")
print(f"Players making playoffs: {regular_data['playoff_label'].sum()}")
print(f"Players not making playoffs: {(1-regular_data['playoff_label']).sum()}")

# Standardize within year
regular_data['GP_std'] = regular_data.groupby('year')['GP'].transform(
    lambda x: (x - x.mean()) / x.std() if x.std() > 0 else 0
)
regular_data['MIN_std'] = regular_data.groupby('year')['MIN'].transform(
    lambda x: (x - x.mean()) / x.std() if x.std() > 0 else 0
)
regular_data['PTS_std'] = regular_data.groupby('year')['PTS'].transform(
    lambda x: (x - x.mean()) / x.std() if x.std() > 0 else 0
)

print(f"\nCorrelation matrix for standardized GP, MIN, PTS:")
print(regular_data[['GP_std', 'MIN_std', 'PTS_std']].corr())

# Fit simple model with just MIN
X_min_only = regular_data[['MIN_std']].values
y = regular_data['playoff_label'].values

lr_min = LogisticRegression(max_iter=1000, random_state=42)
lr_min.fit(X_min_only, y)
print(f"\nCoefficient on MIN_std (univariate model): {lr_min.coef_[0][0]:.4f}")

# Fit model with all three
X_all = regular_data[['GP_std', 'MIN_std', 'PTS_std']].values
lr_all = LogisticRegression(max_iter=1000, random_state=42)
lr_all.fit(X_all, y)
print(f"\nCoefficients (multivariate model):")
print(f"  GP_std: {lr_all.coef_[0][0]:.4f}")
print(f"  MIN_std: {lr_all.coef_[0][1]:.4f}")
print(f"  PTS_std: {lr_all.coef_[0][2]:.4f}")

# Check continuity calculation - sample a few team-seasons
print(f"\n\n{'='*60}")
print("CHECKING CONTINUITY CALCULATION")
print(f"{'='*60}")

nba_playoffs = nba[nba['Season_type'] == 'Playoffs'].copy()
nba_regular = nba[nba['Season_type'] == 'Regular_Season'].copy()
nba_playoffs = nba_playoffs[nba_playoffs['year'].isin(common_seasons)]
nba_regular = nba_regular[nba_regular['year'].isin(common_seasons)]

sample_count = 0
for year in common_seasons[:3]:  # Check first 3 years
    year_playoffs = nba_playoffs[nba_playoffs['year'] == year]
    year_regular = nba_regular[nba_regular['year'] == year]

    teams = year_playoffs['TEAM'].unique()[:2]  # First 2 teams per year

    for team in teams:
        team_playoff = year_playoffs[year_playoffs['TEAM'] == team].copy()
        team_regular = year_regular[year_regular['TEAM'] == team].copy()

        if team_playoff['MIN'].isna().all():
            continue

        regular_players = set(team_regular['PLAYER_ID'].unique())
        playoff_players = set(team_playoff['PLAYER_ID'].unique())

        team_playoff['in_regular'] = team_playoff['PLAYER_ID'].isin(regular_players)
        continued_min = team_playoff[team_playoff['in_regular']]['MIN'].sum()
        total_playoff_min = team_playoff['MIN'].sum()

        ratio = continued_min / total_playoff_min if total_playoff_min > 0 else 0

        print(f"\n{year} {team}:")
        print(f"  Playoff players: {len(playoff_players)}")
        print(f"  Regular season players: {len(regular_players)}")
        print(f"  Playoff players also in regular: {len(playoff_players & regular_players)}")
        print(f"  Continuity ratio: {ratio:.4f}")

        sample_count += 1
        if sample_count >= 6:
            break
    if sample_count >= 6:
        break
