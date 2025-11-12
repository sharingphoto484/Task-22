"""
FIFA World Cup Scoring Patterns Analysis
Comprehensive analysis of historical FIFA World Cup data
"""

import pandas as pd
import numpy as np
import matplotlib.pyplot as plt
from statsmodels.discrete.discrete_model import Poisson
from scipy.stats import pearsonr
import warnings
warnings.filterwarnings('ignore')

print("=" * 80)
print("FIFA WORLD CUP SCORING PATTERNS ANALYSIS")
print("=" * 80)

# ============================================================================
# STEP 1: Load and Clean WorldCupMatches.csv
# ============================================================================
print("\n[1/11] Loading and cleaning WorldCupMatches.csv...")

matches = pd.read_csv('WorldCupMatches.csv')
print(f"  Initial shape: {matches.shape}")

# Remove rows with missing Year
matches = matches.dropna(subset=['Year'])
print(f"  After removing missing Year: {matches.shape}")

# Clean Attendance column (remove thousands separators)
if 'Attendance' in matches.columns:
    matches['Attendance'] = matches['Attendance'].astype(str).str.replace('.', '').str.replace(',', '')
    matches['Attendance'] = pd.to_numeric(matches['Attendance'], errors='coerce')

# Convert Home Team Goals and Away Team Goals to numeric
matches['Home Team Goals'] = pd.to_numeric(matches['Home Team Goals'], errors='coerce')
matches['Away Team Goals'] = pd.to_numeric(matches['Away Team Goals'], errors='coerce')

# Convert Year to numeric
matches['Year'] = pd.to_numeric(matches['Year'], errors='coerce')

print(f"  Final shape: {matches.shape}")

# ============================================================================
# STEP 2: Load and Clean WorldCups.csv
# ============================================================================
print("\n[2/11] Loading and cleaning WorldCups.csv...")

worldcups = pd.read_csv('WorldCups.csv')
print(f"  Initial shape: {worldcups.shape}")

# Confirm Year is numeric
worldcups['Year'] = pd.to_numeric(worldcups['Year'], errors='coerce')

# Clean Attendance column (remove thousands separators)
if 'Attendance' in worldcups.columns:
    worldcups['Attendance'] = worldcups['Attendance'].astype(str).str.replace('.', '').str.replace(',', '')
    worldcups['Attendance'] = pd.to_numeric(worldcups['Attendance'], errors='coerce')

# Rename Attendance in worldcups to avoid conflict
worldcups = worldcups.rename(columns={'Attendance': 'tournament_attendance'})

print(f"  Final shape: {worldcups.shape}")

# ============================================================================
# STEP 3: Process WorldCupPlayers.csv for Card Counts
# ============================================================================
print("\n[3/11] Processing WorldCupPlayers.csv for card counts...")

players = pd.read_csv('WorldCupPlayers.csv')
print(f"  Initial shape: {players.shape}")

# Debug: Check what events are present
print(f"  Unique events: {players['Event'].unique()[:20]}")

# Extract card events from Event column (cards are marked in the Event field)
# Yellow cards often appear as 'Y' followed by minute, e.g., 'Y45'
# Red cards often appear as 'R' followed by minute, e.g., 'R78'
# Also check for patterns

players['Event_str'] = players['Event'].astype(str).str.strip()

# Count yellow cards (Event contains Y but not Y2)
yellow_pattern = players['Event_str'].str.contains('Y', case=False, na=False) & ~players['Event_str'].str.contains('Y2', case=False, na=False)
yellow_cards = players[yellow_pattern].groupby('MatchID').size().reset_index(name='yellow_cards')

# Count red cards (Event contains R or Y2)
red_pattern = players['Event_str'].str.contains('R', case=False, na=False) | players['Event_str'].str.contains('Y2', case=False, na=False)
red_cards = players[red_pattern].groupby('MatchID').size().reset_index(name='red_cards')

# Merge yellow and red cards
cards_summary = yellow_cards.merge(red_cards, on='MatchID', how='outer').fillna(0)
cards_summary['total_cards'] = cards_summary['yellow_cards'] + cards_summary['red_cards']

print(f"  Cards summary shape: {cards_summary.shape}")
print(f"  Total yellow cards: {cards_summary['yellow_cards'].sum():.0f}")
print(f"  Total red cards: {cards_summary['red_cards'].sum():.0f}")

# ============================================================================
# STEP 4: Merge Datasets
# ============================================================================
print("\n[4/11] Merging datasets...")

# Join cards with matches
matches = matches.merge(cards_summary, on='MatchID', how='left')
matches['yellow_cards'] = matches['yellow_cards'].fillna(0)
matches['red_cards'] = matches['red_cards'].fillna(0)
matches['total_cards'] = matches['total_cards'].fillna(0)

print(f"  After merging cards: {matches.shape}")

# Merge with WorldCups on Year
matches = matches.merge(
    worldcups[['Year', 'Country', 'QualifiedTeams', 'MatchesPlayed', 'GoalsScored', 'tournament_attendance']],
    on='Year',
    how='left'
)

print(f"  After merging with WorldCups: {matches.shape}")

# ============================================================================
# STEP 5: Engineer Features
# ============================================================================
print("\n[5/11] Engineering features...")

# Total goals
matches['total_goals'] = matches['Home Team Goals'] + matches['Away Team Goals']

# Goal difference (absolute)
matches['goal_diff'] = abs(matches['Home Team Goals'] - matches['Away Team Goals'])

# Outcome (1 = home win, 0 = draw, -1 = away win)
matches['outcome'] = np.where(
    matches['Home Team Goals'] > matches['Away Team Goals'], 1,
    np.where(matches['Home Team Goals'] == matches['Away Team Goals'], 0, -1)
)

# Halftime goal difference (signed)
matches['Half-time Home Goals'] = pd.to_numeric(matches['Half-time Home Goals'], errors='coerce')
matches['Half-time Away Goals'] = pd.to_numeric(matches['Half-time Away Goals'], errors='coerce')
matches['halftime_goal_diff'] = matches['Half-time Home Goals'] - matches['Half-time Away Goals']

print(f"  Features engineered successfully")

# ============================================================================
# STEP 6: Create Era Classification and Dummy Variables
# ============================================================================
print("\n[6/11] Creating era classification and dummy variables...")

# Define era bins
matches['era'] = pd.cut(
    matches['Year'],
    bins=[1929, 1966, 1989, 2005, 2018],
    labels=['1930-1966', '1970-1989', '1990-2005', '2006-2018']
)

# Create dummy variables (1930-1966 as reference)
era_dummies = pd.get_dummies(matches['era'], prefix='era', drop_first=True, dtype=int)
matches = pd.concat([matches, era_dummies], axis=1)

print(f"  Era distribution:")
print(matches['era'].value_counts().sort_index())

# ============================================================================
# STEP 7: Normalize and Encode Tournament Stage Intensity
# ============================================================================
print("\n[7/11] Normalizing and encoding tournament stage intensity...")

# Normalize Stage
matches['stage_normalized'] = matches['Stage'].astype(str).str.lower().str.strip()

# Map to intensity levels
def map_stage_intensity(stage):
    stage = str(stage).lower().strip()

    # Group stage (intensity 1)
    group_stages = ['group stage', 'group 1', 'group 2', 'group 3', 'group 4',
                    'group 5', 'group 6', 'group 7', 'group 8',
                    'preliminary round', 'first round']

    # Knockout stages (intensity 2)
    knockout_stages = ['round of 16', 'quarter-finals', 'quarter-final',
                       'third place', 'play-off for third place', 'match for third place']

    # Final stages (intensity 3)
    final_stages = ['semi-finals', 'semi-final', 'final']

    for gs in group_stages:
        if gs in stage:
            return 1

    for ks in knockout_stages:
        if ks in stage:
            return 2

    for fs in final_stages:
        if fs in stage:
            return 3

    # Default to 1 for unmatched stages
    return 1

matches['tournament_stage_intensity'] = matches['stage_normalized'].apply(map_stage_intensity)

print(f"  Stage intensity distribution:")
print(matches['tournament_stage_intensity'].value_counts().sort_index())

# ============================================================================
# STEP 8: Prepare Data for Poisson Regression
# ============================================================================
print("\n[8/11] Preparing data for Poisson regression...")

# Create regression dataset (drop rows with missing values in key variables)
regression_data = matches.dropna(subset=[
    'total_goals', 'tournament_stage_intensity', 'halftime_goal_diff',
    'QualifiedTeams', 'total_cards'
]).copy()

# Ensure era dummies exist
era_cols = [col for col in regression_data.columns if col.startswith('era_')]
print(f"  Era dummy columns: {era_cols}")
print(f"  Regression data shape: {regression_data.shape}")

# ============================================================================
# STEP 9: Fit Poisson Regression Model
# ============================================================================
print("\n[9/11] Fitting Poisson regression model...")

# Prepare predictors
X_cols = ['tournament_stage_intensity', 'halftime_goal_diff', 'QualifiedTeams', 'total_cards'] + era_cols
X = regression_data[X_cols].copy()

# Convert all to float to ensure numeric types
for col in X_cols:
    X[col] = pd.to_numeric(X[col], errors='coerce')

# Drop any rows with NaN after conversion
X = X.dropna()
y = regression_data.loc[X.index, 'total_goals']

# Convert to numpy arrays and ensure proper dtypes
X_array = X.values.astype(float)
y_array = y.values.astype(float)

# Add constant column
X_with_const = np.column_stack([X_array, np.ones(len(X_array))])

# Create column names
col_names = list(X.columns) + ['const']

print(f"  Final X shape: {X_with_const.shape}")
print(f"  Data types are all numeric: {X_with_const.dtype}")

# Fit Poisson model with numpy arrays
poisson_model = Poisson(y_array, X_with_const).fit(maxiter=1000, disp=False)

# Calculate McFadden's pseudo R-squared
# McFadden R² = 1 - (log-likelihood of full model / log-likelihood of null model)
null_model = Poisson(y_array, np.ones((len(y_array), 1))).fit(maxiter=1000, disp=False)
mcfadden_r2 = 1 - (poisson_model.llf / null_model.llf)

print(f"\n  Poisson Regression Results:")
print(f"  McFadden's Pseudo R²: {mcfadden_r2:.4f}")
print(f"\n  Coefficients:")
for i, var in enumerate(col_names):
    coef = poisson_model.params[i]
    pval = poisson_model.pvalues[i]
    print(f"    {var:35s}: {coef:10.4f} (p={pval:.4f})")

# Extract specific coefficients and p-values (by column index)
coef_stage_intensity = poisson_model.params[0]  # first column
coef_total_cards = poisson_model.params[3]  # fourth column (after stage, halftime_diff, qualified_teams)
pval_stage_intensity = poisson_model.pvalues[0]

# ============================================================================
# STEP 10: Model Prediction for tournament_stage_intensity = 3
# ============================================================================
print("\n[10/11] Calculating model prediction for tournament_stage_intensity = 3...")

# Create prediction data (using mean values for other predictors)
pred_array = np.array([[
    3,  # tournament_stage_intensity = 3
    X_array[:, 1].mean(),  # halftime_goal_diff mean
    X_array[:, 2].mean(),  # QualifiedTeams mean
    X_array[:, 3].mean(),  # total_cards mean
    X_array[:, 4].mean() if X_array.shape[1] > 4 else 0,  # era_1970-1989 mean
    X_array[:, 5].mean() if X_array.shape[1] > 5 else 0,  # era_1990-2005 mean
    X_array[:, 6].mean() if X_array.shape[1] > 6 else 0,  # era_2006-2018 mean
    1  # const
]])

# Predict
predicted_goals_stage3 = poisson_model.predict(pred_array)[0]

print(f"  Predicted mean goals when tournament_stage_intensity = 3: {predicted_goals_stage3:.4f}")

# ============================================================================
# STEP 11: Calculate Summary Statistics
# ============================================================================
print("\n[11/11] Calculating summary statistics and creating visualization...")

# Total number of matches
total_matches = len(matches)

# Mean total goals
mean_total_goals = matches['total_goals'].mean()

# Mean goal difference
mean_goal_diff = matches['goal_diff'].mean()

# Number of distinct referees
distinct_referees = matches['Referee'].nunique()

# Maximum tournament attendance
max_tournament_attendance = worldcups['tournament_attendance'].max()

# Calculate tournament goals per match in WorldCups
worldcups['goals_per_match'] = worldcups['GoalsScored'] / worldcups['MatchesPlayed']

# Pearson correlation between Year and tournament goals per match
corr_year_goals, _ = pearsonr(worldcups['Year'].dropna(), worldcups['goals_per_match'].dropna())

# ============================================================================
# Visualization: Tournament Goals per Match vs Year
# ============================================================================
plt.figure(figsize=(12, 6))
plt.plot(worldcups['Year'], worldcups['goals_per_match'], marker='o', linewidth=2, markersize=6)
plt.xlabel('Year', fontsize=12)
plt.ylabel('Tournament Goals per Match', fontsize=12)
plt.title('FIFA World Cup: Goals per Match Over Time', fontsize=14, fontweight='bold')
plt.grid(True, alpha=0.3)
plt.tight_layout()
plt.savefig('fifa_goals_per_match.png', dpi=300, bbox_inches='tight')
print("  Visualization saved as 'fifa_goals_per_match.png'")

# ============================================================================
# FINAL REPORT
# ============================================================================
print("\n" + "=" * 80)
print("FINAL REPORT - FIFA WORLD CUP SCORING PATTERNS ANALYSIS")
print("=" * 80)

print(f"\nTotal number of matches: {total_matches}")
print(f"Mean total goals: {mean_total_goals:.4f}")
print(f"Mean goal difference: {mean_goal_diff:.4f}")
print(f"Pearson correlation (Year vs tournament goals per match): {corr_year_goals:.4f}")
print(f"\nPoisson Regression Results:")
print(f"  Coefficient for tournament_stage_intensity: {coef_stage_intensity:.4f}")
print(f"  Coefficient for total_cards: {coef_total_cards:.4f}")
print(f"  P-value for tournament_stage_intensity: {pval_stage_intensity:.4f}")
print(f"  McFadden pseudo R-squared: {mcfadden_r2:.4f}")
print(f"\nNumber of distinct referees: {distinct_referees}")
print(f"Maximum tournament attendance: {max_tournament_attendance:.0f}")
print(f"Model-predicted mean goals (tournament_stage_intensity = 3): {predicted_goals_stage3:.4f}")

print("\n" + "=" * 80)
print("ANALYSIS COMPLETE")
print("=" * 80)

# Save results to a summary file
with open('fifa_analysis_results.txt', 'w') as f:
    f.write("FIFA WORLD CUP SCORING PATTERNS ANALYSIS - RESULTS\n")
    f.write("=" * 80 + "\n\n")
    f.write(f"Total number of matches: {total_matches}\n")
    f.write(f"Mean total goals: {mean_total_goals:.4f}\n")
    f.write(f"Mean goal difference: {mean_goal_diff:.4f}\n")
    f.write(f"Pearson correlation (Year vs tournament goals per match): {corr_year_goals:.4f}\n")
    f.write(f"\nPoisson Regression Results:\n")
    f.write(f"  Coefficient for tournament_stage_intensity: {coef_stage_intensity:.4f}\n")
    f.write(f"  Coefficient for total_cards: {coef_total_cards:.4f}\n")
    f.write(f"  P-value for tournament_stage_intensity: {pval_stage_intensity:.4f}\n")
    f.write(f"  McFadden pseudo R-squared: {mcfadden_r2:.4f}\n")
    f.write(f"\nNumber of distinct referees: {distinct_referees}\n")
    f.write(f"Maximum tournament attendance: {max_tournament_attendance:.0f}\n")
    f.write(f"Model-predicted mean goals (tournament_stage_intensity = 3): {predicted_goals_stage3:.4f}\n")

print("\nResults saved to 'fifa_analysis_results.txt'")
