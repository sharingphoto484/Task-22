import pandas as pd
import numpy as np
import matplotlib.pyplot as plt
from scipy import stats
from sklearn.linear_model import LinearRegression
import warnings
warnings.filterwarnings('ignore')

# Load datasets
print("="*80)
print("LOADING DATASETS")
print("="*80)

df_players = pd.read_csv('df_players.csv')
df_batting = pd.read_csv('df_batting.csv')
df_bowling = pd.read_csv('df_bowling.csv')

print(f"Players dataset: {df_players.shape}")
print(f"Batting dataset: {df_batting.shape}")
print(f"Bowling dataset: {df_bowling.shape}")

# ============================================================================
# DATA MERGING AND CLEANING
# ============================================================================
print("\n" + "="*80)
print("DATA MERGING AND CLEANING")
print("="*80)

# Get unique player names from batting and bowling
batting_players = set(df_batting['batsmanName'].unique())
bowling_players = set(df_bowling['bowlerName'].unique())

# Find players appearing in both datasets
common_players = batting_players.intersection(bowling_players)
print(f"\nPlayers appearing in both batting and bowling: {len(common_players)}")

# Filter datasets to keep only common players
df_batting_filtered = df_batting[df_batting['batsmanName'].isin(common_players)].copy()
df_bowling_filtered = df_bowling[df_bowling['bowlerName'].isin(common_players)].copy()

print(f"Batting records after filtering: {df_batting_filtered.shape[0]}")
print(f"Bowling records after filtering: {df_bowling_filtered.shape[0]}")

# Clean batting data - remove missing/non-numeric values in runs, balls, 4s, 6s
print("\nCleaning batting data...")
df_batting_clean = df_batting_filtered.copy()

# Convert columns to numeric, coercing errors to NaN
for col in ['runs', 'balls', '4s', '6s']:
    df_batting_clean[col] = pd.to_numeric(df_batting_clean[col], errors='coerce')

# Remove rows with missing values in these columns
initial_batting_count = len(df_batting_clean)
df_batting_clean = df_batting_clean.dropna(subset=['runs', 'balls', '4s', '6s'])
print(f"Removed {initial_batting_count - len(df_batting_clean)} batting records with missing/non-numeric values")

# Clean bowling data - remove missing/non-numeric values in overs, runs, wickets, economy
print("\nCleaning bowling data...")
df_bowling_clean = df_bowling_filtered.copy()

# Convert columns to numeric, coercing errors to NaN
for col in ['overs', 'runs', 'wickets', 'economy']:
    df_bowling_clean[col] = pd.to_numeric(df_bowling_clean[col], errors='coerce')

# Remove rows with missing values in these columns
initial_bowling_count = len(df_bowling_clean)
df_bowling_clean = df_bowling_clean.dropna(subset=['overs', 'runs', 'wickets', 'economy'])
print(f"Removed {initial_bowling_count - len(df_bowling_clean)} bowling records with missing/non-numeric values")

print(f"\nFinal batting records: {len(df_batting_clean)}")
print(f"Final bowling records: {len(df_bowling_clean)}")

# ============================================================================
# REGRESSION MODEL: runs ~ balls + 4s + 6s
# ============================================================================
print("\n" + "="*80)
print("REGRESSION MODEL: runs ~ balls + 4s + 6s")
print("="*80)

# Prepare data for regression
X = df_batting_clean[['balls', '4s', '6s']].values
y = df_batting_clean['runs'].values

# Fit linear regression model
reg_model = LinearRegression()
reg_model.fit(X, y)

# Get coefficients
coef_balls = reg_model.coef_[0]
coef_4s = reg_model.coef_[1]
coef_6s = reg_model.coef_[2]

# Calculate R-squared
y_pred = reg_model.predict(X)
r_squared = reg_model.score(X, y)

print(f"\nRegression Coefficients:")
print(f"  Coefficient for balls: {coef_balls:.4f}")
print(f"  Coefficient for 4s: {coef_4s:.4f}")
print(f"  Coefficient for 6s: {coef_6s:.4f}")
print(f"\nCoefficient of Determination (R²): {r_squared:.3f}")

# ============================================================================
# STRIKE RATE STATISTICS
# ============================================================================
print("\n" + "="*80)
print("STRIKE RATE STATISTICS")
print("="*80)

# Calculate SR statistics from all valid batting entries (original filtered data)
# SR column needs to be cleaned
df_batting_sr = df_batting_filtered.copy()
df_batting_sr['SR'] = pd.to_numeric(df_batting_sr['SR'], errors='coerce')
df_batting_sr_clean = df_batting_sr.dropna(subset=['SR'])

mean_sr = df_batting_sr_clean['SR'].mean()
std_sr = df_batting_sr_clean['SR'].std()

print(f"\nOverall Mean Strike Rate: {mean_sr:.3f}")
print(f"Standard Deviation of Strike Rate: {std_sr:.3f}")

# ============================================================================
# PEARSON CORRELATION: overs vs wickets
# ============================================================================
print("\n" + "="*80)
print("PEARSON CORRELATION: overs vs wickets")
print("="*80)

# Calculate Pearson correlation
correlation, p_value = stats.pearsonr(df_bowling_clean['overs'], df_bowling_clean['wickets'])

print(f"\nCorrelation Coefficient: {correlation:.4f}")
print(f"P-value: {p_value:.6f}")

# ============================================================================
# TWO-SAMPLE T-TEST: India vs Australia economy rates
# ============================================================================
print("\n" + "="*80)
print("TWO-SAMPLE T-TEST: India vs Australia Economy Rates")
print("="*80)

# Get economy rates for India and Australia
india_economy = df_bowling_clean[df_bowling_clean['bowlingTeam'] == 'India']['economy']
australia_economy = df_bowling_clean[df_bowling_clean['bowlingTeam'] == 'Australia']['economy']

print(f"\nIndia bowling records: {len(india_economy)}")
print(f"Australia bowling records: {len(australia_economy)}")

if len(india_economy) > 0 and len(australia_economy) > 0:
    # Perform two-sample t-test
    t_statistic, p_value_ttest = stats.ttest_ind(india_economy, australia_economy)

    print(f"\nMean economy rate (India): {india_economy.mean():.3f}")
    print(f"Mean economy rate (Australia): {australia_economy.mean():.3f}")
    print(f"\nT-statistic: {t_statistic:.3f}")
    print(f"P-value: {p_value_ttest:.4f}")
else:
    print("\nInsufficient data for t-test")

# ============================================================================
# CAUSAL INFERENCE: playingRole effect on runs
# ============================================================================
print("\n" + "="*80)
print("CAUSAL INFERENCE: playingRole Effect on Runs")
print("="*80)

# Merge batting data with player data to get playingRole and team
df_batting_with_role = df_batting_clean.merge(
    df_players[['name', 'playingRole', 'team']],
    left_on='batsmanName',
    right_on='name',
    how='left'
)

# Remove rows with missing playingRole or team
df_batting_with_role = df_batting_with_role.dropna(subset=['playingRole', 'team'])

print(f"\nRecords with playingRole and team information: {len(df_batting_with_role)}")

# Create dummy variables for playingRole and team
df_iv = pd.get_dummies(df_batting_with_role[['playingRole', 'team']], drop_first=True)

# Prepare data for regression
X_causal = df_iv.values
y_causal = df_batting_with_role['runs'].values

# Fit linear regression model
causal_model = LinearRegression()
causal_model.fit(X_causal, y_causal)

# The causal coefficient is represented by the coefficients for playingRole variables
# Let's get the mean causal effect
playingRole_cols = [col for col in df_iv.columns if 'playingRole' in col]
if len(playingRole_cols) > 0:
    causal_coefs = causal_model.coef_[:len(playingRole_cols)]
    mean_causal_effect = np.mean(causal_coefs)
    print(f"\nEstimated Causal Coefficient (mean effect): {mean_causal_effect:.3f}")
else:
    print("\nNo playingRole variables found for causal inference")

# ============================================================================
# SCATTER PLOT WITH REGRESSION LINE: runs vs balls
# ============================================================================
print("\n" + "="*80)
print("SCATTER PLOT WITH REGRESSION LINE: runs vs balls")
print("="*80)

# Fit simple linear regression for runs vs balls
X_simple = df_batting_clean['balls'].values.reshape(-1, 1)
y_simple = df_batting_clean['runs'].values

simple_reg = LinearRegression()
simple_reg.fit(X_simple, y_simple)

slope = simple_reg.coef_[0]
intercept = simple_reg.intercept_

print(f"\nSlope of regression line: {slope:.4f}")
print(f"Intercept: {intercept:.4f}")

# Create scatter plot
plt.figure(figsize=(10, 6))
plt.scatter(X_simple, y_simple, alpha=0.3, s=20, label='Observed')

# Plot regression line
x_line = np.linspace(X_simple.min(), X_simple.max(), 100)
y_line = slope * x_line + intercept
plt.plot(x_line, y_line, 'r-', linewidth=2, label=f'Regression Line (slope={slope:.4f})')

plt.xlabel('Balls Faced', fontsize=12)
plt.ylabel('Runs Scored', fontsize=12)
plt.title('Relationship Between Runs and Balls\n(Predicted vs Observed Scoring Intensity)', fontsize=14)
plt.legend()
plt.grid(True, alpha=0.3)
plt.tight_layout()
plt.savefig('runs_vs_balls_regression.png', dpi=300, bbox_inches='tight')
print("\nScatter plot saved as 'runs_vs_balls_regression.png'")

# ============================================================================
# PERFORMANCE INDEX: (total runs + total wickets) / total matches
# ============================================================================
print("\n" + "="*80)
print("PERFORMANCE INDEX CALCULATION")
print("="*80)

# Calculate total runs per player from batting data
player_batting_stats = df_batting_clean.groupby('batsmanName').agg({
    'runs': 'sum',
    'match': 'count'
}).rename(columns={'match': 'batting_matches'})

# Calculate total wickets per player from bowling data
player_bowling_stats = df_bowling_clean.groupby('bowlerName').agg({
    'wickets': 'sum'
})

# Merge the stats
player_performance = player_batting_stats.join(player_bowling_stats, how='inner')

# Calculate performance index
player_performance['performance_index'] = (
    player_performance['runs'] + player_performance['wickets']
) / player_performance['batting_matches']

# Find player with highest performance index
best_player_idx = player_performance['performance_index'].idxmax()
best_player_index = player_performance.loc[best_player_idx, 'performance_index']

print(f"\nTop 10 Players by Performance Index:")
print(player_performance.nlargest(10, 'performance_index')[['runs', 'wickets', 'batting_matches', 'performance_index']])

print(f"\n{'='*80}")
print(f"PLAYER WITH HIGHEST PERFORMANCE INDEX:")
print(f"  Player Name: {best_player_idx}")
print(f"  Performance Index: {best_player_index:.3f}")
print(f"  Total Runs: {player_performance.loc[best_player_idx, 'runs']}")
print(f"  Total Wickets: {player_performance.loc[best_player_idx, 'wickets']}")
print(f"  Total Matches: {player_performance.loc[best_player_idx, 'batting_matches']}")
print(f"{'='*80}")

# ============================================================================
# SUMMARY OF ALL RESULTS
# ============================================================================
print("\n" + "="*80)
print("SUMMARY OF ALL RESULTS")
print("="*80)

print("\n1. REGRESSION MODEL (runs ~ balls + 4s + 6s):")
print(f"   - Coefficient for balls: {coef_balls:.4f}")
print(f"   - Coefficient for 4s: {coef_4s:.4f}")
print(f"   - Coefficient for 6s: {coef_6s:.4f}")
print(f"   - R² (Coefficient of Determination): {r_squared:.3f}")

print("\n2. STRIKE RATE STATISTICS:")
print(f"   - Mean Strike Rate: {mean_sr:.3f}")
print(f"   - Standard Deviation: {std_sr:.3f}")

print("\n3. PEARSON CORRELATION (overs vs wickets):")
print(f"   - Correlation Coefficient: {correlation:.4f}")

print("\n4. T-TEST (India vs Australia Economy Rates):")
if len(india_economy) > 0 and len(australia_economy) > 0:
    print(f"   - T-statistic: {t_statistic:.3f}")
    print(f"   - P-value: {p_value_ttest:.4f}")

print("\n5. CAUSAL INFERENCE (playingRole effect on runs):")
if len(playingRole_cols) > 0:
    print(f"   - Estimated Causal Coefficient: {mean_causal_effect:.3f}")

print("\n6. REGRESSION LINE (runs vs balls):")
print(f"   - Slope: {slope:.4f}")

print("\n7. PERFORMANCE INDEX:")
print(f"   - Best Player: {best_player_idx}")
print(f"   - Performance Index: {best_player_index:.3f}")

print("\n" + "="*80)
print("ANALYSIS COMPLETE!")
print("="*80)
