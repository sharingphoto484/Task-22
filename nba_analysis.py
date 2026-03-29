# ==========================================
# NBA Team Ratings Analysis Script
# Requirements: pandas, numpy, matplotlib, scipy, scikit-learn
# Input files: nba.csv, Regular_Season.csv, Playoffs.csv (in same directory)
# Output files: srs_ratings.csv, playoff_shifts.csv, heatmap.png, analysis_summary.json
# ==========================================

import pandas as pd
import numpy as np
import matplotlib.pyplot as plt
from scipy import stats
from scipy.sparse import csr_matrix
from scipy.sparse.linalg import lsqr
import json
import warnings
warnings.filterwarnings('ignore')

# ---------- Load CSVs Robustly ----------
def load_data():
    """Load and validate the three required CSV files."""
    print("Loading data files...")

    # Load Regular Season game logs
    try:
        regular_season = pd.read_csv('Regular_Season.csv')
        print(f"  Regular_Season.csv loaded: {len(regular_season)} rows")
    except FileNotFoundError:
        raise FileNotFoundError("Regular_Season.csv not found in current directory")

    # Load Playoffs game logs
    try:
        playoffs = pd.read_csv('Playoffs.csv')
        print(f"  Playoffs.csv loaded: {len(playoffs)} rows")
    except FileNotFoundError:
        raise FileNotFoundError("Playoffs.csv not found in current directory")

    # Load player data
    try:
        nba = pd.read_csv('nba.csv')
        print(f"  nba.csv loaded: {len(nba)} rows")
    except FileNotFoundError:
        raise FileNotFoundError("nba.csv not found in current directory")

    return regular_season, playoffs, nba

# ---------- Validate Data Columns ----------
def validate_data(regular_season, playoffs, nba):
    """Validate that required columns exist in each dataset."""
    print("\nValidating data structure...")

    # Required columns for game logs
    required_game_cols = ['Season', 'Date', 'HomeTeam', 'AwayTeam', 'HomePTS', 'AwayPTS']

    # Check Regular Season columns
    missing_regular = [col for col in required_game_cols if col not in regular_season.columns]
    if missing_regular:
        raise ValueError(f"Regular_Season.csv missing required columns: {missing_regular}")

    # Check Playoffs columns
    missing_playoffs = [col for col in required_game_cols if col not in playoffs.columns]
    if missing_playoffs:
        raise ValueError(f"Playoffs.csv missing required columns: {missing_playoffs}")

    # Check nba.csv columns
    if 'Player' not in nba.columns and 'PLAYER' not in nba.columns:
        raise ValueError("nba.csv missing Player column")
    if 'Team' not in nba.columns and 'TEAM' not in nba.columns:
        raise ValueError("nba.csv missing Team column")
    if 'Season' not in nba.columns and 'year' not in nba.columns:
        raise ValueError("nba.csv missing Season column")

    # Check for at least one of Minutes or Games
    has_minutes = 'Minutes' in nba.columns or 'MIN' in nba.columns
    has_games = 'Games' in nba.columns or 'GP' in nba.columns
    if not (has_minutes or has_games):
        raise ValueError("nba.csv missing both Minutes and Games columns")

    print("  Data validation passed")

# ---------- Identify Intersected Seasons ----------
def get_intersected_seasons(regular_season, playoffs, nba):
    """Find seasons present in all three datasets."""
    print("\nIdentifying intersected seasons...")

    # Normalize season column names
    rs_seasons = set(regular_season['Season'].unique())
    po_seasons = set(playoffs['Season'].unique())

    # Handle different possible column names for nba.csv
    if 'Season' in nba.columns:
        nba_seasons = set(nba['Season'].unique())
    elif 'year' in nba.columns:
        nba_seasons = set(nba['year'].unique())
    else:
        raise ValueError("Cannot find Season column in nba.csv")

    # Get intersection
    intersected = rs_seasons & po_seasons & nba_seasons
    intersected = sorted(list(intersected))

    print(f"  Found {len(intersected)} intersected seasons: {intersected}")
    return intersected

# ---------- Simple Rating System Implementation ----------
def compute_srs_ratings(games_df, season):
    """
    Compute Simple Rating System ratings for teams in a given season.
    SRS solves: margin = home_rating - away_rating + season_intercept
    """
    season_games = games_df[games_df['Season'] == season].copy()

    if len(season_games) == 0:
        return {}

    # Calculate margin
    season_games['Margin'] = season_games['HomePTS'] - season_games['AwayPTS']

    # Get unique teams
    teams = sorted(list(set(season_games['HomeTeam'].unique()) | set(season_games['AwayTeam'].unique())))
    team_to_idx = {team: idx for idx, team in enumerate(teams)}
    n_teams = len(teams)

    # Build design matrix for least squares
    # Each row is a game, columns are [team_ratings..., intercept]
    n_games = len(season_games)
    A = np.zeros((n_games, n_teams + 1))
    b = season_games['Margin'].values

    for game_idx, (_, game) in enumerate(season_games.iterrows()):
        home_idx = team_to_idx[game['HomeTeam']]
        away_idx = team_to_idx[game['AwayTeam']]

        A[game_idx, home_idx] = 1
        A[game_idx, away_idx] = -1
        A[game_idx, n_teams] = 1  # intercept

    # Solve least squares: A @ x = b
    # Add constraint that sum of ratings = 0 for identifiability
    A_augmented = np.vstack([A, np.concatenate([np.ones(n_teams), [0]])])
    b_augmented = np.concatenate([b, [0]])

    x, residuals, rank, s = np.linalg.lstsq(A_augmented, b_augmented, rcond=None)

    # Extract ratings and intercept
    ratings = {teams[i]: x[i] for i in range(n_teams)}
    intercept = x[n_teams]

    return ratings

# ---------- Calculate Win Percentage ----------
def calculate_win_percentage(games_df, season, team):
    """Calculate regular season win percentage for a team."""
    season_games = games_df[games_df['Season'] == season]

    # Games where team was home
    home_games = season_games[season_games['HomeTeam'] == team].copy()
    home_games['Win'] = (home_games['HomePTS'] > home_games['AwayPTS']).astype(int)

    # Games where team was away
    away_games = season_games[season_games['AwayTeam'] == team].copy()
    away_games['Win'] = (away_games['AwayPTS'] > away_games['HomePTS']).astype(int)

    total_wins = home_games['Win'].sum() + away_games['Win'].sum()
    total_games = len(home_games) + len(away_games)

    if total_games == 0:
        return 0.0

    return total_wins / total_games

# ---------- Regular Season SRS Analysis ----------
def analyze_regular_season_srs(regular_season, intersected_seasons):
    """
    Compute SRS ratings for regular season and analyze correlation with win percentage.
    """
    print("\n" + "="*60)
    print("REGULAR SEASON SIMPLE RATING SYSTEM ANALYSIS")
    print("="*60)

    all_ratings = []
    all_win_pcts = []
    season_ratings = {}

    for season in intersected_seasons:
        print(f"\nProcessing season {season}...")
        ratings = compute_srs_ratings(regular_season, season)
        season_ratings[season] = ratings

        for team, rating in ratings.items():
            win_pct = calculate_win_percentage(regular_season, season, team)
            all_ratings.append(rating)
            all_win_pcts.append(win_pct)

    # Calculate league-wide standard deviation of ratings
    ratings_std = np.std(all_ratings, ddof=1)

    # Calculate Pearson correlation between ratings and win percentage
    correlation, p_value = stats.pearsonr(all_ratings, all_win_pcts)

    print(f"\n{'='*60}")
    print("REGULAR SEASON SRS RESULTS:")
    print(f"{'='*60}")
    print(f"League-wide standard deviation of team ratings: {ratings_std:.4f}")
    print(f"Pearson correlation (ratings vs win %):          {correlation:.4f}")
    print(f"{'='*60}\n")

    return season_ratings, ratings_std, correlation

# ---------- Playoff SRS Analysis ----------
def analyze_playoff_srs(playoffs, regular_season_ratings, intersected_seasons):
    """
    Compute playoff SRS and calculate playoff shifts.
    """
    print("\n" + "="*60)
    print("PLAYOFF SIMPLE RATING SYSTEM ANALYSIS")
    print("="*60)

    playoff_shifts = []
    teams_with_positive_shift = 0
    playoff_ratings_dict = {}

    for season in intersected_seasons:
        print(f"\nProcessing playoff season {season}...")
        playoff_ratings = compute_srs_ratings(playoffs, season)
        playoff_ratings_dict[season] = playoff_ratings

        regular_ratings = regular_season_ratings[season]

        # Calculate shifts for teams that appear in both
        for team in playoff_ratings:
            if team in regular_ratings:
                shift = playoff_ratings[team] - regular_ratings[team]
                playoff_shifts.append(shift)
                if shift > 0:
                    teams_with_positive_shift += 1

    # Calculate mean playoff shift
    mean_shift = np.mean(playoff_shifts) if playoff_shifts else 0.0

    print(f"\n{'='*60}")
    print("PLAYOFF SHIFT RESULTS:")
    print(f"{'='*60}")
    print(f"League-wide mean playoff shift:        {mean_shift:.4f}")
    print(f"Number of teams with positive shift:   {teams_with_positive_shift}")
    print(f"{'='*60}\n")

    return playoff_ratings_dict, mean_shift, teams_with_positive_shift

# ---------- Home Court Advantage Analysis ----------
def analyze_home_advantage(regular_season, intersected_seasons):
    """
    Calculate average home court advantage across all games.
    """
    print("\n" + "="*60)
    print("HOME COURT ADVANTAGE ANALYSIS")
    print("="*60)

    # Filter to intersected seasons
    season_games = regular_season[regular_season['Season'].isin(intersected_seasons)].copy()

    # Calculate home margin for each game
    season_games['HomeMargin'] = season_games['HomePTS'] - season_games['AwayPTS']

    # Calculate grand mean
    mean_home_margin = season_games['HomeMargin'].mean()

    print(f"\n{'='*60}")
    print("HOME COURT ADVANTAGE RESULTS:")
    print(f"{'='*60}")
    print(f"Grand mean home margin: {mean_home_margin:.4f}")
    print(f"{'='*60}\n")

    return mean_home_margin

# ---------- Roster Continuity Analysis ----------
def analyze_roster_continuity(nba, intersected_seasons):
    """
    Measure roster continuity by tracking player-team matches across seasons.
    """
    print("\n" + "="*60)
    print("ROSTER CONTINUITY ANALYSIS")
    print("="*60)

    # Normalize column names
    nba_df = nba.copy()

    # Map columns to standard names
    if 'Player' in nba_df.columns:
        nba_df['player'] = nba_df['Player']
    elif 'PLAYER' in nba_df.columns:
        nba_df['player'] = nba_df['PLAYER']

    if 'Team' in nba_df.columns:
        nba_df['team'] = nba_df['Team']
    elif 'TEAM' in nba_df.columns:
        nba_df['team'] = nba_df['TEAM']

    if 'Season' in nba_df.columns:
        nba_df['season'] = nba_df['Season']
    elif 'year' in nba_df.columns:
        nba_df['season'] = nba_df['year']

    # Determine weight column
    if 'Minutes' in nba_df.columns:
        nba_df['weight'] = nba_df['Minutes']
    elif 'MIN' in nba_df.columns:
        nba_df['weight'] = nba_df['MIN']
    elif 'Games' in nba_df.columns:
        nba_df['weight'] = nba_df['Games']
    elif 'GP' in nba_df.columns:
        nba_df['weight'] = nba_df['GP']
    else:
        nba_df['weight'] = 1  # Count of rows

    # Filter to intersected seasons
    nba_df = nba_df[nba_df['season'].isin(intersected_seasons)]

    # Calculate continuity for each team-season
    continuity_scores = []

    # Get unique seasons sorted
    seasons_sorted = sorted(nba_df['season'].unique())

    for i in range(1, len(seasons_sorted)):
        current_season = seasons_sorted[i]
        prev_season = seasons_sorted[i-1]

        current_data = nba_df[nba_df['season'] == current_season]
        prev_data = nba_df[nba_df['season'] == prev_season]

        # For each team in current season
        for team in current_data['team'].unique():
            team_current = current_data[current_data['team'] == team]
            team_prev = prev_data[prev_data['team'] == team]

            # Get players from both seasons
            current_players = set(team_current['player'].unique())
            prev_players = set(team_prev['player'].unique())

            # Find players who were on team in both seasons
            returning_players = current_players & prev_players

            # Calculate weighted continuity
            total_weight = team_current['weight'].sum()
            returning_weight = team_current[team_current['player'].isin(returning_players)]['weight'].sum()

            if total_weight > 0:
                continuity = returning_weight / total_weight
                continuity_scores.append(continuity)

    # Calculate league-wide average
    avg_continuity = np.mean(continuity_scores) if continuity_scores else 0.0

    print(f"\n{'='*60}")
    print("ROSTER CONTINUITY RESULTS:")
    print(f"{'='*60}")
    print(f"League-wide average continuity share: {avg_continuity:.4f}")
    print(f"{'='*60}\n")

    return avg_continuity

# ---------- Playoff Margin Heatmap ----------
def create_playoff_heatmap(playoffs, intersected_seasons):
    """
    Create head-to-head playoff margin heatmap.
    """
    print("\n" + "="*60)
    print("PLAYOFF MARGIN HEATMAP ANALYSIS")
    print("="*60)

    # Filter to intersected seasons
    playoff_games = playoffs[playoffs['Season'].isin(intersected_seasons)].copy()

    # Get all unique teams
    all_teams = sorted(list(set(playoff_games['HomeTeam'].unique()) |
                            set(playoff_games['AwayTeam'].unique())))

    # Initialize margin matrix
    n_teams = len(all_teams)
    margin_sums = np.zeros((n_teams, n_teams))
    game_counts = np.zeros((n_teams, n_teams))

    team_to_idx = {team: idx for idx, team in enumerate(all_teams)}

    # Process each game
    for _, game in playoff_games.iterrows():
        home_team = game['HomeTeam']
        away_team = game['AwayTeam']
        home_pts = game['HomePTS']
        away_pts = game['AwayPTS']

        home_idx = team_to_idx[home_team]
        away_idx = team_to_idx[away_team]

        # Home team's perspective
        margin_home = home_pts - away_pts
        margin_sums[home_idx, away_idx] += margin_home
        game_counts[home_idx, away_idx] += 1

        # Away team's perspective
        margin_away = away_pts - home_pts
        margin_sums[away_idx, home_idx] += margin_away
        game_counts[away_idx, home_idx] += 1

    # Calculate average margins
    avg_margins = np.divide(margin_sums, game_counts,
                           out=np.zeros_like(margin_sums),
                           where=game_counts > 0)

    # Find max absolute value
    max_abs_margin = np.max(np.abs(avg_margins[game_counts > 0]))

    # Create heatmap
    plt.figure(figsize=(14, 12))
    im = plt.imshow(avg_margins, cmap='RdBu_r', aspect='auto', vmin=-max_abs_margin, vmax=max_abs_margin)
    plt.colorbar(im, label='Average Point Margin')

    plt.xticks(range(n_teams), all_teams, rotation=90, ha='right', fontsize=8)
    plt.yticks(range(n_teams), all_teams, fontsize=8)

    plt.xlabel('Opponent Team')
    plt.ylabel('Team')
    plt.title('Head-to-Head Playoff Margin Heatmap\n(Average point margin for team vs opponent)')
    plt.tight_layout()
    plt.savefig('playoff_margin_heatmap.png', dpi=150, bbox_inches='tight')
    print("  Heatmap saved as 'playoff_margin_heatmap.png'")

    print(f"\n{'='*60}")
    print("HEATMAP RESULTS:")
    print(f"{'='*60}")
    print(f"Maximum absolute cell value: {max_abs_margin:.4f}")
    print(f"{'='*60}\n")

    return max_abs_margin

# ---------- Statistical Test: Regular Season Strength vs Playoff Performance ----------
def test_regular_season_playoff_correlation(regular_season, playoffs,
                                           regular_season_ratings,
                                           intersected_seasons):
    """
    Test whether regular season ratings correlate with playoff win percentage.
    """
    print("\n" + "="*60)
    print("REGULAR SEASON STRENGTH VS PLAYOFF OUTCOMES TEST")
    print("="*60)

    ratings_list = []
    playoff_win_pcts = []

    for season in intersected_seasons:
        regular_ratings = regular_season_ratings[season]

        # Get teams that made playoffs
        playoff_teams = set(playoffs[playoffs['Season'] == season]['HomeTeam'].unique()) | \
                       set(playoffs[playoffs['Season'] == season]['AwayTeam'].unique())

        for team in playoff_teams:
            if team in regular_ratings:
                rating = regular_ratings[team]

                # Calculate playoff win percentage
                playoff_win_pct = calculate_win_percentage(playoffs, season, team)

                ratings_list.append(rating)
                playoff_win_pcts.append(playoff_win_pct)

    # Calculate correlation
    if len(ratings_list) >= 3:
        correlation, p_value = stats.pearsonr(ratings_list, playoff_win_pcts)

        # Determine verdict
        verdict = 1 if (correlation > 0 and p_value < 0.05) else 0

        print(f"\nAnalyzed {len(ratings_list)} team-season observations")
        print(f"\nPearson correlation: {correlation:.4f}")
        print(f"P-value: {p_value:.6f}")
        print(f"\n{'='*60}")
        print("VERDICT:")
        print(f"{'='*60}")
        print(f"Binary indicator (positive correlation with p < 0.05): {verdict}")
        print(f"{'='*60}\n")
    else:
        correlation = 0.0
        p_value = 1.0
        verdict = 0
        print("Insufficient data for correlation test")

    return verdict, correlation, p_value

# ---------- Main Analysis Pipeline ----------
def main():
    """Execute complete NBA analysis pipeline."""
    print("\n" + "="*70)
    print(" "*15 + "NBA TEAM RATINGS ANALYSIS")
    print("="*70)

    # Load data
    regular_season, playoffs, nba = load_data()

    # Validate data
    validate_data(regular_season, playoffs, nba)

    # Get intersected seasons
    intersected_seasons = get_intersected_seasons(regular_season, playoffs, nba)

    # 1. Regular Season SRS Analysis
    regular_season_ratings, ratings_std, ratings_winpct_corr = \
        analyze_regular_season_srs(regular_season, intersected_seasons)

    # 2. Playoff SRS Analysis
    playoff_ratings, mean_playoff_shift, positive_shift_count = \
        analyze_playoff_srs(playoffs, regular_season_ratings, intersected_seasons)

    # 3. Home Court Advantage
    mean_home_margin = analyze_home_advantage(regular_season, intersected_seasons)

    # 4. Roster Continuity
    avg_continuity = analyze_roster_continuity(nba, intersected_seasons)

    # 5. Playoff Margin Heatmap
    max_abs_heatmap_value = create_playoff_heatmap(playoffs, intersected_seasons)

    # 6. Statistical Test
    verdict, test_correlation, test_pvalue = \
        test_regular_season_playoff_correlation(regular_season, playoffs,
                                               regular_season_ratings,
                                               intersected_seasons)

    # Compile results
    results = {
        'intersected_seasons': intersected_seasons,
        'regular_season_srs': {
            'league_std_dev_ratings': round(ratings_std, 4),
            'pearson_corr_ratings_winpct': round(ratings_winpct_corr, 4)
        },
        'playoff_srs': {
            'mean_playoff_shift': round(mean_playoff_shift, 4),
            'teams_with_positive_shift': int(positive_shift_count)
        },
        'home_court_advantage': {
            'grand_mean_home_margin': round(mean_home_margin, 4)
        },
        'roster_continuity': {
            'league_avg_continuity_share': round(avg_continuity, 4)
        },
        'playoff_heatmap': {
            'max_absolute_cell_value': round(max_abs_heatmap_value, 4)
        },
        'statistical_test': {
            'verdict': int(verdict),
            'correlation': round(test_correlation, 4),
            'p_value': round(test_pvalue, 6)
        }
    }

    # Save results to JSON
    with open('analysis_summary.json', 'w') as f:
        json.dump(results, f, indent=2)

    print("\n" + "="*70)
    print(" "*20 + "FINAL SUMMARY")
    print("="*70)
    print(f"\nRegular Season SRS:")
    print(f"  - League-wide std dev of team ratings:      {results['regular_season_srs']['league_std_dev_ratings']}")
    print(f"  - Pearson correlation (ratings vs win %):   {results['regular_season_srs']['pearson_corr_ratings_winpct']}")

    print(f"\nPlayoff SRS:")
    print(f"  - League-wide mean playoff shift:           {results['playoff_srs']['mean_playoff_shift']}")
    print(f"  - Teams with positive playoff shift:        {results['playoff_srs']['teams_with_positive_shift']}")

    print(f"\nHome Court Advantage:")
    print(f"  - Grand mean home margin:                   {results['home_court_advantage']['grand_mean_home_margin']}")

    print(f"\nRoster Continuity:")
    print(f"  - League-wide average continuity share:     {results['roster_continuity']['league_avg_continuity_share']}")

    print(f"\nPlayoff Heatmap:")
    print(f"  - Max absolute cell value:                  {results['playoff_heatmap']['max_absolute_cell_value']}")

    print(f"\nStatistical Test (Regular Season → Playoff Performance):")
    print(f"  - Binary verdict (positive corr, p<0.05):   {results['statistical_test']['verdict']}")

    print(f"\n{'='*70}")
    print("Analysis complete! Results saved to 'analysis_summary.json'")
    print("="*70 + "\n")

    return results

if __name__ == "__main__":
    main()
