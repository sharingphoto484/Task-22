import pandas as pd
import numpy as np
from sklearn.linear_model import LogisticRegression
from sklearn.metrics import roc_auc_score
import hashlib

# Load and prepare data
df_nba = pd.read_csv('nba.csv')
df_regular = pd.read_csv('Regular_Season.csv')
df_playoffs = pd.read_csv('Playoffs.csv')

# Normalize
df_nba['Season_type'] = df_nba['Season_type'].replace('Regular%20Season', 'Regular_Season')
df_regular['Season_type'] = 'Regular_Season'
df_playoffs['Season_type'] = 'Playoffs'

# Find common years
years_nba = set(df_nba['year'].unique())
years_regular = set(df_regular['year'].unique())
years_playoffs = set(df_playoffs['year'].unique())
common_years = sorted(years_nba & years_regular & years_playoffs)

# Combine and filter
all_data = pd.concat([df_nba, df_regular, df_playoffs], ignore_index=True)
all_data = all_data[all_data['year'].isin(common_years)].copy()

regular_season = all_data[all_data['Season_type'] == 'Regular_Season'].copy()
playoffs = all_data[all_data['Season_type'] == 'Playoffs'].copy()

required_fields = ['year', 'PLAYER_ID', 'PLAYER', 'TEAM', 'GP', 'MIN', 'PTS']
regular_season = regular_season.dropna(subset=required_fields).copy()

# Create labels
playoff_keys = set()
for _, row in playoffs.iterrows():
    key = (row['year'], str(row['PLAYER_ID']), row['TEAM'])
    playoff_keys.add(key)

regular_season['made_playoffs'] = regular_season.apply(
    lambda row: 1 if (row['year'], str(row['PLAYER_ID']), row['TEAM']) in playoff_keys else 0,
    axis=1
)

# Standardize within year
def standardize_within_year(df, col):
    return df.groupby('year')[col].transform(
        lambda x: (x - x.mean()) / x.std() if x.std() > 0 else 0
    )

regular_season['GP_std'] = standardize_within_year(regular_season, 'GP')
regular_season['MIN_std'] = standardize_within_year(regular_season, 'MIN')
regular_season['PTS_std'] = standardize_within_year(regular_season, 'PTS')

print("="*70)
print("DIAGNOSTIC: Testing different modeling approaches")
print("="*70)

X = regular_season[['GP_std', 'MIN_std', 'PTS_std']].values
y = regular_season['made_playoffs'].values

# Test 1: Simple model with all data, no CV
print("\nTest 1: Logistic regression on ALL data (no cross-validation)")
model1 = LogisticRegression(max_iter=1000, random_state=42)
model1.fit(X, y)
preds1 = model1.predict_proba(X)[:, 1]
auc1 = roc_auc_score(y, preds1)
print(f"  Coefficients: GP={model1.coef_[0][0]:.4f}, MIN={model1.coef_[0][1]:.4f}, PTS={model1.coef_[0][2]:.4f}")
print(f"  AUC (training): {auc1:.4f}")

# Test 2: Cross-validation with mean coefficient reporting
print("\nTest 2: Cross-validation with mean coefficient across folds")

def assign_fold(player_id):
    hash_obj = hashlib.md5(str(player_id).encode())
    hash_int = int(hash_obj.hexdigest(), 16)
    return hash_int % 5

regular_season['fold'] = regular_season['PLAYER_ID'].apply(assign_fold)
folds = regular_season['fold'].values

oof_preds = np.zeros(len(regular_season))
fold_coefficients = []

for fold_id in range(5):
    train_idx = folds != fold_id
    val_idx = folds == fold_id

    X_train, y_train = X[train_idx], y[train_idx]
    X_val = X[val_idx]

    model = LogisticRegression(max_iter=1000, random_state=42)
    model.fit(X_train, y_train)

    oof_preds[val_idx] = model.predict_proba(X_val)[:, 1]
    fold_coefficients.append(model.coef_[0])

cv_auc = roc_auc_score(y, oof_preds)
mean_coefs = np.mean(fold_coefficients, axis=0)

print(f"  Mean coefficients: GP={mean_coefs[0]:.4f}, MIN={mean_coefs[1]:.4f}, PTS={mean_coefs[2]:.4f}")
print(f"  Cross-validated AUC: {cv_auc:.4f}")

# Test 3: Try different random states
print("\nTest 3: Testing different random states")
for rs in [None, 0, 1, 42, 100]:
    model_rs = LogisticRegression(max_iter=1000, random_state=rs)
    model_rs.fit(X, y)
    print(f"  Random state {rs}: MIN coef = {model_rs.coef_[0][1]:.4f}")

# Test 4: Try with L2 penalty variations
print("\nTest 4: Testing different regularization strengths")
for C in [0.1, 1.0, 10.0, 100.0]:
    model_c = LogisticRegression(max_iter=1000, C=C, random_state=42)
    model_c.fit(X, y)
    print(f"  C={C}: MIN coef = {model_c.coef_[0][1]:.4f}")

# Test 5: Check correlation
print("\nTest 5: Correlation between predictors")
corr_matrix = regular_season[['GP_std', 'MIN_std', 'PTS_std']].corr()
print(corr_matrix)

# Test 6: VIF for multicollinearity
print("\nTest 6: Variance Inflation Factors")
from sklearn.linear_model import LinearRegression

for i, col in enumerate(['GP_std', 'MIN_std', 'PTS_std']):
    other_cols = [c for j, c in enumerate(['GP_std', 'MIN_std', 'PTS_std']) if j != i]
    X_other = regular_season[other_cols].values
    y_target = regular_season[col].values

    lr = LinearRegression()
    lr.fit(X_other, y_target)
    r_squared = lr.score(X_other, y_target)

    if r_squared < 1:
        vif = 1 / (1 - r_squared)
    else:
        vif = np.inf

    print(f"  {col}: VIF = {vif:.4f}")
