import pandas as pd
import numpy as np
from sklearn.linear_model import LogisticRegression
from sklearn.metrics import roc_auc_score
import hashlib
import warnings
warnings.filterwarnings('ignore')

# Load data
regular_raw = pd.read_csv('Regular_Season.csv')
playoffs_raw = pd.read_csv('Playoffs.csv')
nba_raw = pd.read_csv('nba.csv')

nba_raw['Season_type'] = nba_raw['Season_type'].replace('Regular%20Season', 'Regular_Season')
regular_raw['Season_type'] = 'Regular_Season'
playoffs_raw['Season_type'] = 'Playoffs'

# Common years
years_nba = set(nba_raw['year'].unique())
years_regular = set(regular_raw['year'].unique())
years_playoffs = set(playoffs_raw['year'].unique())
common_years = sorted(years_nba & years_regular & years_playoffs)

# Combine and filter
all_data = pd.concat([nba_raw, regular_raw, playoffs_raw], ignore_index=True)
all_data = all_data[all_data['year'].isin(common_years)].copy()

regular = all_data[all_data['Season_type'] == 'Regular_Season'].copy()
playoffs = all_data[all_data['Season_type'] == 'Playoffs'].copy()

required_fields = ['year', 'PLAYER_ID', 'PLAYER', 'TEAM', 'GP', 'MIN', 'PTS']
regular = regular.dropna(subset=required_fields).copy()

# Create labels
playoff_keys = set()
for _, row in playoffs.iterrows():
    if pd.notna(row.get('PLAYER_ID')) and pd.notna(row.get('TEAM')):
        key = (row['year'], str(row['PLAYER_ID']), row['TEAM'])
        playoff_keys.add(key)

regular['made_playoffs'] = regular.apply(
    lambda row: 1 if (row['year'], str(row['PLAYER_ID']), row['TEAM']) in playoff_keys else 0,
    axis=1
)

# Standardize
def standardize_within_year(df, col):
    return df.groupby('year')[col].transform(
        lambda x: (x - x.mean()) / x.std() if x.std() > 0 else 0
    )

regular['GP_std'] = standardize_within_year(regular, 'GP')
regular['MIN_std'] = standardize_within_year(regular, 'MIN')
regular['PTS_std'] = standardize_within_year(regular, 'PTS')

X = regular[['GP_std', 'MIN_std', 'PTS_std']].values
y = regular['made_playoffs'].values

print("="*70)
print("TESTING DIFFERENT SOLVER AND PENALTY COMBINATIONS")
print("="*70)

# Test different combinations
configs = [
    {'penalty': 'l2', 'solver': 'lbfgs', 'C': 1.0},
    {'penalty': 'l2', 'solver': 'liblinear', 'C': 1.0},
    {'penalty': 'l2', 'solver': 'newton-cg', 'C': 1.0},
    {'penalty': 'l2', 'solver': 'sag', 'C': 1.0},
    {'penalty': 'l2', 'solver': 'saga', 'C': 1.0},
    {'penalty': 'l1', 'solver': 'liblinear', 'C': 1.0},
    {'penalty': 'l1', 'solver': 'saga', 'C': 1.0},
    {'penalty': None, 'solver': 'lbfgs', 'C': None},
    {'penalty': None, 'solver': 'newton-cg', 'C': None},
    {'penalty': 'l2', 'solver': 'lbfgs', 'C': 0.5},
    {'penalty': 'l2', 'solver': 'lbfgs', 'C': 2.0},
    {'penalty': 'l2', 'solver': 'lbfgs', 'C': 10.0},
]

for config in configs:
    try:
        if config['C'] is None:
            model = LogisticRegression(
                penalty=config['penalty'],
                solver=config['solver'],
                max_iter=1000,
                random_state=42
            )
        else:
            model = LogisticRegression(
                penalty=config['penalty'],
                solver=config['solver'],
                C=config['C'],
                max_iter=1000,
                random_state=42
            )

        model.fit(X, y)
        preds = model.predict_proba(X)[:, 1]
        auc = roc_auc_score(y, preds)

        print(f"\nPenalty: {config['penalty']:<4}, Solver: {config['solver']:<12}, C: {config['C']}")
        print(f"  GP={model.coef_[0][0]:7.4f}, MIN={model.coef_[0][1]:7.4f}, PTS={model.coef_[0][2]:7.4f}")
        print(f"  AUC: {auc:.4f}")

    except Exception as e:
        print(f"\nPenalty: {config['penalty']:<4}, Solver: {config['solver']:<12}, C: {config['C']}")
        print(f"  Error: {e}")
