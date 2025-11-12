# ==========================================
# FIFA World Cup Scoring Patterns Analysis
# ==========================================
# Requirements: pandas, numpy, matplotlib, scipy, statsmodels
# Input files: WorldCupMatches.csv, WorldCups.csv, WorldCupPlayers.csv (in same directory)
# Output files: fifa_analysis_results.txt, fifa_goals_per_match.png
# Description: Analyzes historical FIFA World Cup scoring patterns across eras,
#              tournament stages, and referee discipline using Poisson regression
# ==========================================

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

# ---------- Load Matches CSV ----------
matches = pd.read_csv('WorldCupMatches.csv')
print(f"  Initial shape: {matches.shape}")

# ---------- Remove Missing Year Rows ----------
matches = matches.dropna(subset=['Year'])
print(f"  After removing missing Year: {matches.shape}")

# ---------- Clean Attendance Column ----------
# Remove thousands separators (dots and commas) to make numeric
if 'Attendance' in matches.columns:
    matches['Attendance'] = matches['Attendance'].astype(str).str.replace('.', '').str.replace(',', '')
    matches['Attendance'] = pd.to_numeric(matches['Attendance'], errors='coerce')

# ---------- Convert Goals to Numeric ----------
matches['Home Team Goals'] = pd.to_numeric(matches['Home Team Goals'], errors='coerce')
matches['Away Team Goals'] = pd.to_numeric(matches['Away Team Goals'], errors='coerce')

# ---------- Convert Year to Numeric ----------
matches['Year'] = pd.to_numeric(matches['Year'], errors='coerce')

print(f"  Final shape: {matches.shape}")

# ============================================================================
# STEP 2: Load and Clean WorldCups.csv
# ============================================================================
print("\n[2/11] Loading and cleaning WorldCups.csv...")

# ---------- Load WorldCups CSV ----------
worldcups = pd.read_csv('WorldCups.csv')
print(f"  Initial shape: {worldcups.shape}")

# ---------- Ensure Year is Numeric (Merge Key) ----------
worldcups['Year'] = pd.to_numeric(worldcups['Year'], errors='coerce')

# ---------- Clean Attendance Column ----------
# Remove thousands separators to make numeric
if 'Attendance' in worldcups.columns:
    worldcups['Attendance'] = worldcups['Attendance'].astype(str).str.replace('.', '').str.replace(',', '')
    worldcups['Attendance'] = pd.to_numeric(worldcups['Attendance'], errors='coerce')

# Rename to avoid conflict with matches Attendance
worldcups = worldcups.rename(columns={'Attendance': 'tournament_attendance'})

print(f"  Final shape: {worldcups.shape}")

# ============================================================================
# STEP 3: Process WorldCupPlayers.csv for Card Counts
# ============================================================================
print("\n[3/11] Processing WorldCupPlayers.csv for card counts...")

# ---------- Load Players CSV ----------
players = pd.read_csv('WorldCupPlayers.csv')
print(f"  Initial shape: {players.shape}")

# ---------- Count Yellow Cards (Event equals 'Y' EXACTLY) ----------
# Per prompt: "count yellow card events where Event equals Y"
# Using exact match only (not Y45, not Y90, only exactly 'Y')
yellow_cards = players[players['Event'] == 'Y'].groupby('MatchID').size().reset_index(name='yellow_cards')

# ---------- Count Red Cards (Event equals 'Y2' or 'R' EXACTLY) ----------
# Per prompt: "count red card events where Event equals Y2 or R"
# Using exact match only (not Y2', not R89', only exactly 'Y2' or 'R')
red_cards = players[players['Event'].isin(['Y2', 'R'])].groupby('MatchID').size().reset_index(name='red_cards')

# ---------- Merge Yellow and Red Cards ----------
cards_summary = yellow_cards.merge(red_cards, on='MatchID', how='outer').fillna(0)
cards_summary['total_cards'] = cards_summary['yellow_cards'] + cards_summary['red_cards']

print(f"  Cards summary shape: {cards_summary.shape}")
print(f"  Total yellow cards: {cards_summary['yellow_cards'].sum():.0f}")
print(f"  Total red cards: {cards_summary['red_cards'].sum():.0f}")

# ============================================================================
# STEP 4: Merge Datasets
# ============================================================================
print("\n[4/11] Merging datasets...")

# ---------- Join Cards with Matches on MatchID ----------
matches = matches.merge(cards_summary, on='MatchID', how='left')
matches['yellow_cards'] = matches['yellow_cards'].fillna(0)
matches['red_cards'] = matches['red_cards'].fillna(0)
matches['total_cards'] = matches['total_cards'].fillna(0)

print(f"  After merging cards: {matches.shape}")

# ---------- Merge with WorldCups on Year ----------
# Append Country, QualifiedTeams, MatchesPlayed, GoalsScored, tournament_attendance
matches = matches.merge(
    worldcups[['Year', 'Country', 'QualifiedTeams', 'MatchesPlayed', 'GoalsScored', 'tournament_attendance']],
    on='Year',
    how='left'
)

print(f"  After merging with WorldCups: {matches.shape}")

# ============================================================================
# STEP 5: Engineer Features - Total Goals, Goal Diff, Outcome, Halftime Diff
# ============================================================================
print("\n[5/11] Engineering features...")

# ---------- Total Goals ----------
matches['total_goals'] = matches['Home Team Goals'] + matches['Away Team Goals']

# ---------- Goal Difference (Absolute) ----------
matches['goal_diff'] = abs(matches['Home Team Goals'] - matches['Away Team Goals'])

# ---------- Outcome (1=home win, 0=draw, -1=away win) ----------
matches['outcome'] = np.where(
    matches['Home Team Goals'] > matches['Away Team Goals'], 1,
    np.where(matches['Home Team Goals'] == matches['Away Team Goals'], 0, -1)
)

# ---------- Halftime Goal Difference (Signed) ----------
matches['Half-time Home Goals'] = pd.to_numeric(matches['Half-time Home Goals'], errors='coerce')
matches['Half-time Away Goals'] = pd.to_numeric(matches['Half-time Away Goals'], errors='coerce')
matches['halftime_goal_diff'] = matches['Half-time Home Goals'] - matches['Half-time Away Goals']

print(f"  Features engineered successfully")

# ============================================================================
# STEP 6: Create Era Classification and Dummy Variables
# ============================================================================
print("\n[6/11] Creating era classification and dummy variables...")

# ---------- Define Era Bins ----------
# Four eras: 1930-1966, 1970-1989, 1990-2005, 2006-2018
matches['era'] = pd.cut(
    matches['Year'],
    bins=[1929, 1966, 1989, 2005, 2018],
    labels=['1930-1966', '1970-1989', '1990-2005', '2006-2018']
)

# ---------- Create Dummy Variables (1930-1966 as reference) ----------
era_dummies = pd.get_dummies(matches['era'], prefix='era', drop_first=True, dtype=int)
matches = pd.concat([matches, era_dummies], axis=1)

print(f"  Era distribution:")
print(matches['era'].value_counts().sort_index())

# ============================================================================
# STEP 7: Normalize and Encode Tournament Stage Intensity
# ============================================================================
print("\n[7/11] Normalizing and encoding tournament stage intensity...")

# ---------- Normalize Stage Text ----------
matches['stage_normalized'] = matches['Stage'].astype(str).str.lower().str.strip()

# ---------- Map Stage to Intensity Levels ----------
def map_stage_intensity(stage):
    """
    Maps tournament stage to intensity level:
    - Group stages (1): group stage, group 1-8, preliminary round, first round
    - Knockout stages (2): round of 16, quarter-finals, third place
    - Final stages (3): semi-finals, final
    """
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

# ---------- Create Regression Dataset ----------
# Drop rows with missing values in key variables
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

# ---------- Prepare Predictors ----------
# Predictors: tournament_stage_intensity, halftime_goal_diff, QualifiedTeams,
#             total_cards, and era dummies
X_cols_initial = ['tournament_stage_intensity', 'halftime_goal_diff', 'QualifiedTeams', 'total_cards'] + era_cols
X = regression_data[X_cols_initial].copy()

# ---------- Convert to Numeric and Handle Missing ----------
for col in X_cols_initial:
    X[col] = pd.to_numeric(X[col], errors='coerce')

# Drop any rows with NaN after conversion
X = X.dropna()
y = regression_data.loc[X.index, 'total_goals']

# ---------- Remove Zero-Variance Predictors ----------
# If total_cards is all zeros (exact match found no cards), remove it to avoid singular matrix
X_cols = []
for col in X_cols_initial:
    if X[col].std() > 0:  # Only keep columns with variance
        X_cols.append(col)
    else:
        print(f"  Warning: Removing '{col}' (zero variance)")

X = X[X_cols]

# ---------- Convert to NumPy Arrays ----------
X_array = X.values.astype(float)
y_array = y.values.astype(float)

# ---------- Add Constant Column ----------
X_with_const = np.column_stack([X_array, np.ones(len(X_array))])

# Create column names
col_names = list(X.columns) + ['const']

print(f"  Final X shape: {X_with_const.shape}")
print(f"  Data types are all numeric: {X_with_const.dtype}")

# ---------- Fit Poisson Model ----------
poisson_model = Poisson(y_array, X_with_const).fit(maxiter=1000, disp=False)

# ---------- Calculate McFadden's Pseudo R-squared ----------
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

# ---------- Extract Specific Coefficients and P-values ----------
coef_stage_intensity = poisson_model.params[0]  # first column
pval_stage_intensity = poisson_model.pvalues[0]

# Find total_cards coefficient if it exists in the model
if 'total_cards' in col_names:
    cards_idx = col_names.index('total_cards')
    coef_total_cards = poisson_model.params[cards_idx]
else:
    coef_total_cards = 0.0  # Not in model (zero variance)

# ============================================================================
# STEP 10: Model Prediction for tournament_stage_intensity = 3
# ============================================================================
print("\n[10/11] Calculating model prediction for tournament_stage_intensity = 3...")

# ---------- Create Prediction Data ----------
# Set tournament_stage_intensity = 3, use mean values for other predictors
# Build prediction array based on actual columns in model
pred_list = [3]  # tournament_stage_intensity = 3

# Add other predictors in order based on col_names
for i, col in enumerate(col_names[1:-1]):  # Skip first (stage_intensity, already added) and last (const)
    if i < X_array.shape[1]:
        pred_list.append(X_array[:, i+1].mean())
    else:
        pred_list.append(0)

pred_list.append(1)  # const
pred_array = np.array([pred_list])

# ---------- Predict Mean Goals ----------
predicted_goals_stage3 = poisson_model.predict(pred_array)[0]

print(f"  Predicted mean goals when tournament_stage_intensity = 3: {predicted_goals_stage3:.4f}")

# ============================================================================
# STEP 11: Calculate Summary Statistics
# ============================================================================
print("\n[11/11] Calculating summary statistics and creating visualization...")

# ---------- Total Number of Matches ----------
total_matches = len(matches)

# ---------- Mean Total Goals ----------
mean_total_goals = matches['total_goals'].mean()

# ---------- Mean Goal Difference ----------
mean_goal_diff = matches['goal_diff'].mean()

# ---------- Number of Distinct Referees ----------
distinct_referees = matches['Referee'].nunique()

# ---------- Maximum Tournament Attendance ----------
max_tournament_attendance = worldcups['tournament_attendance'].max()

# ---------- Calculate Tournament Goals per Match ----------
worldcups['goals_per_match'] = worldcups['GoalsScored'] / worldcups['MatchesPlayed']

# ---------- Pearson Correlation Between Year and Goals per Match ----------
corr_year_goals, _ = pearsonr(worldcups['Year'].dropna(), worldcups['goals_per_match'].dropna())

# ============================================================================
# Visualization: Tournament Goals per Match vs Year
# ============================================================================

# ---------- Create Line Plot ----------
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
# FINAL REPORT - Round All Metrics to 4 Decimal Places
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

# ---------- Save Results to Summary File ----------
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
