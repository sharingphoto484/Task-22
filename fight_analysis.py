# ==========================================
# Integrated Fight Analysis Script
# Requirements: pandas, numpy, matplotlib, scipy, scikit-learn, lifelines, networkx
# Input files: Fighters.xlsx, Fights.xlsx, Fighters Stats.xlsx
# Output files: elo_progression.png, survival_curve.png, style_similarity_heatmap.png,
#               network_dominance.png, roc_curve.png, lorenz_curve.png,
#               lda_weight_class.png
# ==========================================

import pandas as pd
import numpy as np
import matplotlib
matplotlib.use('Agg')
import matplotlib.pyplot as plt
from sklearn.ensemble import GradientBoostingClassifier
from sklearn.metrics import roc_curve, auc, accuracy_score
from sklearn.preprocessing import StandardScaler
from sklearn.discriminant_analysis import LinearDiscriminantAnalysis
from sklearn.metrics.pairwise import cosine_similarity
from lifelines import CoxPHFitter, KaplanMeierFitter
import networkx as nx
import warnings
warnings.filterwarnings('ignore')


# ---------- Helper: Load Raw Datasets with Prompt-Aligned Column Names ----------
def load_fighters():
    df = pd.read_excel('Fighters.xlsx')
    df = df.rename(columns={
        'Fighter_Id': 'fighter_id',
        'Full Name': 'fighter_name',
        'Reach': 'reach_cm',
        'Ht.': 'height_cm',
        'Wt.': 'weight_kg',
        'Weight_Class': 'weight_class',
    })
    return df

def load_fights():
    df = pd.read_excel('Fights.xlsx')
    df = df.rename(columns={
        'Fight_Id': 'fight_id',
        'Fighter_Id_1': 'red_fighter_id',
        'Fighter_Id_2': 'blue_fighter_id',
        'Method': 'result_method',
        'Weight_Class': 'weight_class',
    })
    return df

def load_stats():
    df = pd.read_excel('Fighters Stats.xlsx')
    df = df.rename(columns={
        'Fighter_Id': 'fighter_id',
        'Full Name': 'fighter_name',
        'STR': 'total_strikes_landed',
        'TD': 'takedowns_landed',
        'Ctrl': 'control_time_seconds',
        'Weight_Class': 'weight_class',
    })
    return df


# ---------- Helper: Derive Winner ID from Fights ----------
def derive_winner_id(fights):
    conditions = [
        fights['Result_1'] == 'W',
        fights['Result_2'] == 'W',
    ]
    choices = [fights['red_fighter_id'], fights['blue_fighter_id']]
    fights = fights.copy()
    fights['winner_id'] = np.select(conditions, choices, default=np.nan)
    return fights


# ---------- Helper: Derive Loser ID from Fights ----------
def derive_loser_id(fights):
    conditions = [
        fights['Result_1'] == 'L',
        fights['Result_2'] == 'L',
    ]
    choices = [fights['red_fighter_id'], fights['blue_fighter_id']]
    fights = fights.copy()
    fights['loser_id'] = np.select(conditions, choices, default=np.nan)
    return fights


# ---------- Helper: Compute Fight Duration in Seconds ----------
def compute_fight_duration(fights):
    fights = fights.copy()

    def parse_round_duration(time_format):
        if pd.isna(time_format) or time_format == 'No Time Limit':
            return None
        import re
        match = re.search(r'\(([^)]+)\)', str(time_format))
        if match:
            durations = [int(x) for x in match.group(1).split('-')]
            return durations
        return None

    def calc_duration(row):
        durations = parse_round_duration(row['Time Format'])
        if durations is None:
            return np.nan
        fight_round = int(row['Round'])
        fight_time = str(row['Fight_Time'])
        parts = fight_time.split(':')
        if len(parts) == 2:
            minutes = int(parts[0])
            seconds = int(parts[1])
        else:
            return np.nan
        completed_seconds = 0
        for i in range(min(fight_round - 1, len(durations))):
            completed_seconds += durations[i] * 60
        completed_seconds += minutes * 60 + seconds
        return completed_seconds

    fights['fight_duration_seconds'] = fights.apply(calc_duration, axis=1)
    return fights


# ---------- Helper: Identify Decision Fights ----------
def is_decision(method):
    if pd.isna(method):
        return False
    return 'DEC' in str(method)


# ============================================================
# OPERATION 1: Elo Rating System
# ============================================================
def operation_1_elo():
    print("=" * 60)
    print("OPERATION 1: Elo Rating System")
    print("=" * 60)

    # ---------- Load Data Independently ----------
    fights = load_fights()

    # ---------- Derive Winner and Loser ----------
    fights = derive_winner_id(fights)
    fights = derive_loser_id(fights)

    # ---------- Exclude Draws and No-Contest ----------
    fights = fights[
        (fights['Result_1'].isin(['W', 'L'])) &
        (fights['Result_2'].isin(['W', 'L']))
    ].copy()

    # ---------- Sort by fight_id Ascending ----------
    fights = fights.sort_values('fight_id').reset_index(drop=True)

    # ---------- Initialize Elo Ratings ----------
    elo_ratings = {}
    K = 32

    # ---------- Track Elo Progression for Plot ----------
    elo_history = {}

    # ---------- Process Fights in Order ----------
    for idx, row in fights.iterrows():
        winner = row['winner_id']
        loser = row['loser_id']

        if pd.isna(winner) or pd.isna(loser):
            continue

        # Initialize if new
        if winner not in elo_ratings:
            elo_ratings[winner] = 1500.0
        if loser not in elo_ratings:
            elo_ratings[loser] = 1500.0

        # Expected scores
        r_w = elo_ratings[winner]
        r_l = elo_ratings[loser]
        e_w = 1.0 / (1.0 + 10.0 ** ((r_l - r_w) / 400.0))
        e_l = 1.0 / (1.0 + 10.0 ** ((r_w - r_l) / 400.0))

        # Update
        elo_ratings[winner] = r_w + K * (1.0 - e_w)
        elo_ratings[loser] = r_l + K * (0.0 - e_l)

        # Track history
        fight_seq = idx
        if winner not in elo_history:
            elo_history[winner] = []
        elo_history[winner].append((fight_seq, elo_ratings[winner]))

        if loser not in elo_history:
            elo_history[loser] = []
        elo_history[loser].append((fight_seq, elo_ratings[loser]))

    # ---------- Report Maximum Final Elo ----------
    max_elo = max(elo_ratings.values())
    max_elo_rounded = round(max_elo, 2)
    print(f"Maximum Final Elo Rating: {max_elo_rounded}")

    # ---------- Generate Elo Progression Line Plot ----------
    top_fighters = sorted(elo_ratings, key=elo_ratings.get, reverse=True)[:5]

    fig, ax = plt.subplots(figsize=(12, 6))
    for fid in top_fighters:
        if fid in elo_history:
            indices = [x[0] for x in elo_history[fid]]
            elos = [x[1] for x in elo_history[fid]]
            ax.plot(indices, elos, label=f'Fighter {fid[:8]}...', alpha=0.7)
    ax.set_xlabel('Fight Sequence Index')
    ax.set_ylabel('Elo Rating')
    ax.set_title('Elo Rating Progression (Top 5 Fighters)')
    ax.legend(fontsize=8)
    ax.grid(True, alpha=0.3)
    plt.tight_layout()
    plt.savefig('elo_progression.png', dpi=150)
    plt.close()
    print("Plot saved: elo_progression.png")

    return max_elo_rounded


# ============================================================
# OPERATION 2: Cox Proportional Hazards (Reach Difference)
# ============================================================
def operation_2_cox():
    print("\n" + "=" * 60)
    print("OPERATION 2: Cox Proportional Hazards Model")
    print("=" * 60)

    # ---------- Load Data Independently ----------
    fights = load_fights()
    fighters = load_fighters()

    # ---------- Compute Fight Duration ----------
    fights = compute_fight_duration(fights)

    # ---------- Define Event Variable ----------
    fights['event'] = fights['result_method'].apply(lambda x: 0 if is_decision(x) else 1)

    # ---------- Join reach_cm for Red Fighter ----------
    fighters_reach_red = fighters[['fighter_id', 'reach_cm']].rename(
        columns={'fighter_id': 'red_fighter_id', 'reach_cm': 'reach_cm_red'}
    )
    fights = fights.merge(fighters_reach_red, on='red_fighter_id', how='left')

    # ---------- Join reach_cm for Blue Fighter ----------
    fighters_reach_blue = fighters[['fighter_id', 'reach_cm']].rename(
        columns={'fighter_id': 'blue_fighter_id', 'reach_cm': 'reach_cm_blue'}
    )
    fights = fights.merge(fighters_reach_blue, on='blue_fighter_id', how='left')

    # ---------- Compute reach_cm Difference ----------
    fights['reach_cm_diff'] = (fights['reach_cm_red'] - fights['reach_cm_blue']).abs()

    # ---------- Drop Rows with Missing Referenced Variables ----------
    referenced_cols = ['event', 'fight_duration_seconds', 'reach_cm_diff']
    fights = fights.dropna(subset=referenced_cols).copy()

    # ---------- Fit Cox PH Model ----------
    cox_df = fights[['fight_duration_seconds', 'event', 'reach_cm_diff']].copy()
    cph = CoxPHFitter()
    cph.fit(cox_df, duration_col='fight_duration_seconds', event_col='event')

    # ---------- Report Hazard Ratio ----------
    hazard_ratio = np.exp(cph.params_['reach_cm_diff'])
    hazard_ratio_rounded = round(hazard_ratio, 3)
    print(f"Hazard Ratio for reach_cm difference: {hazard_ratio_rounded}")

    # ---------- Generate Survival Curve Plot ----------
    fig, ax = plt.subplots(figsize=(10, 6))
    kmf = KaplanMeierFitter()
    kmf.fit(cox_df['fight_duration_seconds'], event_observed=cox_df['event'])
    kmf.plot_survival_function(ax=ax)
    ax.set_xlabel('Fight Duration (seconds)')
    ax.set_ylabel('Survival Probability')
    ax.set_title('Survival Curve for Bout Duration Outcomes')
    ax.grid(True, alpha=0.3)
    plt.tight_layout()
    plt.savefig('survival_curve.png', dpi=150)
    plt.close()
    print("Plot saved: survival_curve.png")

    return hazard_ratio_rounded


# ============================================================
# OPERATION 3: Fighter Style Similarity (Cosine Similarity)
# ============================================================
def operation_3_similarity():
    print("\n" + "=" * 60)
    print("OPERATION 3: Fighter Style Similarity")
    print("=" * 60)

    # ---------- Load Data Independently ----------
    stats = load_stats()

    # ---------- Select Referenced Variables ----------
    style_cols = ['total_strikes_landed', 'takedowns_landed', 'control_time_seconds']
    subset = stats[['fighter_id'] + style_cols].copy()

    # ---------- Drop Rows with Missing Referenced Variables ----------
    subset = subset.dropna(subset=style_cols)

    # ---------- Standardize Features ----------
    scaler = StandardScaler()
    scaled_values = scaler.fit_transform(subset[style_cols])

    # ---------- Compute Cosine Similarity ----------
    sim_matrix = cosine_similarity(scaled_values)

    # ---------- Exclude Self-Comparisons ----------
    np.fill_diagonal(sim_matrix, -np.inf)

    # ---------- Report Maximum Similarity ----------
    max_sim = np.max(sim_matrix)
    max_sim_rounded = round(max_sim, 3)
    print(f"Maximum Cosine Similarity (excluding self): {max_sim_rounded}")

    # ---------- Generate Heatmap Plot ----------
    fig, ax = plt.subplots(figsize=(12, 10))
    n_display = 30
    display_sim = cosine_similarity(scaled_values[:n_display])
    np.fill_diagonal(display_sim, 0)

    im = ax.imshow(display_sim, cmap='YlOrRd', aspect='auto')
    ax.set_xticks(range(n_display))
    ax.set_yticks(range(n_display))
    fighter_labels = subset['fighter_id'].iloc[:n_display].str[:8].tolist()
    ax.set_xticklabels(fighter_labels, rotation=90, fontsize=6)
    ax.set_yticklabels(fighter_labels, fontsize=6)
    ax.set_xlabel('Fighter Identifiers')
    ax.set_ylabel('Fighter Identifiers')
    ax.set_title('Fighter Style Similarity Heatmap')
    plt.colorbar(im, ax=ax, label='Cosine Similarity')
    plt.tight_layout()
    plt.savefig('style_similarity_heatmap.png', dpi=150)
    plt.close()
    print("Plot saved: style_similarity_heatmap.png")

    return max_sim_rounded


# ============================================================
# OPERATION 4: PageRank Network Dominance
# ============================================================
def operation_4_pagerank():
    print("\n" + "=" * 60)
    print("OPERATION 4: PageRank Network Dominance")
    print("=" * 60)

    # ---------- Load Data Independently ----------
    fights = load_fights()

    # ---------- Derive Winner and Loser ----------
    fights = derive_winner_id(fights)
    fights = derive_loser_id(fights)

    # ---------- Filter Valid Win/Loss Fights ----------
    fights = fights[
        (fights['Result_1'].isin(['W', 'L'])) &
        (fights['Result_2'].isin(['W', 'L']))
    ].copy()

    # ---------- Construct Directed Graph (Winner -> Loser) ----------
    G = nx.DiGraph()
    for _, row in fights.iterrows():
        winner = row['winner_id']
        loser = row['loser_id']
        if pd.notna(winner) and pd.notna(loser):
            G.add_edge(winner, loser)

    # ---------- Compute PageRank ----------
    pagerank = nx.pagerank(G, alpha=0.85)

    # ---------- Report Maximum PageRank ----------
    max_pr = max(pagerank.values())
    max_pr_rounded = round(max_pr, 4)
    print(f"Maximum PageRank Score: {max_pr_rounded}")

    # ---------- Compute In-Degree ----------
    in_degrees = dict(G.in_degree())

    # ---------- Generate Scatter Plot ----------
    fig, ax = plt.subplots(figsize=(10, 6))
    fighters_in_both = set(pagerank.keys()) & set(in_degrees.keys())
    x_vals = [in_degrees[f] for f in fighters_in_both]
    y_vals = [pagerank[f] for f in fighters_in_both]
    ax.scatter(x_vals, y_vals, alpha=0.4, s=10)
    ax.set_xlabel('In-Degree')
    ax.set_ylabel('PageRank Score')
    ax.set_title('Network Dominance: In-Degree vs PageRank')
    ax.grid(True, alpha=0.3)
    plt.tight_layout()
    plt.savefig('network_dominance.png', dpi=150)
    plt.close()
    print("Plot saved: network_dominance.png")

    return max_pr_rounded


# ============================================================
# OPERATION 5: Win Prediction (GradientBoostingClassifier)
# ============================================================
def operation_5_win_prediction():
    print("\n" + "=" * 60)
    print("OPERATION 5: Win Prediction Model")
    print("=" * 60)

    # ---------- Load Data Independently ----------
    fights = load_fights()
    stats = load_stats()

    # ---------- Derive Winner ----------
    fights = derive_winner_id(fights)

    # ---------- Filter Valid Win/Loss Fights ----------
    fights = fights[
        (fights['Result_1'].isin(['W', 'L'])) &
        (fights['Result_2'].isin(['W', 'L']))
    ].copy()

    # ---------- Join Fighter Stats for Red Fighter ----------
    stats_red = stats[['fighter_id', 'total_strikes_landed', 'takedowns_landed', 'control_time_seconds']].rename(
        columns={
            'fighter_id': 'red_fighter_id',
            'total_strikes_landed': 'total_strikes_landed_red',
            'takedowns_landed': 'takedowns_landed_red',
            'control_time_seconds': 'control_time_seconds_red',
        }
    )
    fights = fights.merge(stats_red, on='red_fighter_id', how='left')

    # ---------- Join Fighter Stats for Blue Fighter ----------
    stats_blue = stats[['fighter_id', 'total_strikes_landed', 'takedowns_landed', 'control_time_seconds']].rename(
        columns={
            'fighter_id': 'blue_fighter_id',
            'total_strikes_landed': 'total_strikes_landed_blue',
            'takedowns_landed': 'takedowns_landed_blue',
            'control_time_seconds': 'control_time_seconds_blue',
        }
    )
    fights = fights.merge(stats_blue, on='blue_fighter_id', how='left')

    # ---------- Compute Feature Differences (Red minus Blue) ----------
    fights['total_strikes_landed'] = fights['total_strikes_landed_red'] - fights['total_strikes_landed_blue']
    fights['takedowns_landed'] = fights['takedowns_landed_red'] - fights['takedowns_landed_blue']
    fights['control_time_seconds'] = fights['control_time_seconds_red'] - fights['control_time_seconds_blue']

    # ---------- Define Target: Win equals 1 if Red Fighter Wins ----------
    fights['win'] = (fights['winner_id'] == fights['red_fighter_id']).astype(int)

    # ---------- Drop Rows with Missing Referenced Variables ----------
    feature_cols = ['total_strikes_landed', 'takedowns_landed', 'control_time_seconds']
    referenced = feature_cols + ['win']
    fights = fights.dropna(subset=referenced)

    # ---------- Prepare Features and Target ----------
    X = fights[feature_cols].values
    y = fights['win'].values

    # ---------- Train GradientBoostingClassifier ----------
    gbc = GradientBoostingClassifier(
        learning_rate=0.1,
        max_depth=3,
        n_estimators=100,
        random_state=42
    )
    gbc.fit(X, y)

    # ---------- Evaluate Accuracy ----------
    y_prob = gbc.predict_proba(X)[:, 1]
    y_pred = (y_prob >= 0.5).astype(int)
    acc = accuracy_score(y, y_pred)
    acc_rounded = round(acc, 3)
    print(f"Accuracy: {acc_rounded}")

    # ---------- Generate ROC Curve ----------
    fpr, tpr, _ = roc_curve(y, y_prob)
    roc_auc = auc(fpr, tpr)

    fig, ax = plt.subplots(figsize=(8, 6))
    ax.plot(fpr, tpr, color='blue', lw=2, label=f'ROC Curve (AUC = {roc_auc:.3f})')
    ax.plot([0, 1], [0, 1], color='gray', linestyle='--', lw=1)
    ax.set_xlabel('False Positive Rate')
    ax.set_ylabel('True Positive Rate')
    ax.set_title('ROC Curve for Win Prediction Performance')
    ax.legend(loc='lower right')
    ax.grid(True, alpha=0.3)
    plt.tight_layout()
    plt.savefig('roc_curve.png', dpi=150)
    plt.close()
    print("Plot saved: roc_curve.png")

    return acc_rounded


# ============================================================
# OPERATION 6: Competitive Balance (Gini Coefficient)
# ============================================================
def operation_6_gini():
    print("\n" + "=" * 60)
    print("OPERATION 6: Competitive Balance (Gini Coefficient)")
    print("=" * 60)

    # ---------- Load Data Independently ----------
    fights = load_fights()

    # ---------- Derive Winner ----------
    fights = derive_winner_id(fights)

    # ---------- Filter Valid Win/Loss Fights ----------
    fights = fights[
        (fights['Result_1'].isin(['W', 'L'])) &
        (fights['Result_2'].isin(['W', 'L']))
    ].copy()

    # ---------- Compute Total Wins Per Fighter ----------
    winner_counts = fights['winner_id'].value_counts().reset_index()
    winner_counts.columns = ['fighter_id', 'wins']

    # ---------- Include All Fighters with Zero Wins ----------
    fights = derive_loser_id(fights)
    all_fighters = set(fights['winner_id'].dropna()) | set(fights['loser_id'].dropna())
    win_dict = dict(zip(winner_counts['fighter_id'], winner_counts['wins']))
    win_distribution = np.array([win_dict.get(f, 0) for f in all_fighters])

    # ---------- Calculate Gini Coefficient ----------
    def gini(array):
        array = np.sort(array).astype(float)
        n = len(array)
        if n == 0 or array.sum() == 0:
            return 0.0
        index = np.arange(1, n + 1)
        return (2 * np.sum(index * array) - (n + 1) * np.sum(array)) / (n * np.sum(array))

    gini_coeff = gini(win_distribution)
    gini_rounded = round(gini_coeff, 3)
    print(f"Gini Coefficient: {gini_rounded}")

    # ---------- Generate Lorenz Curve ----------
    sorted_wins = np.sort(win_distribution)
    cumulative_wins = np.cumsum(sorted_wins) / np.sum(sorted_wins)
    cumulative_fighters = np.arange(1, len(sorted_wins) + 1) / len(sorted_wins)

    fig, ax = plt.subplots(figsize=(8, 6))
    ax.plot(cumulative_fighters, cumulative_wins, color='blue', lw=2, label='Lorenz Curve')
    ax.plot([0, 1], [0, 1], color='gray', linestyle='--', lw=1, label='Perfect Equality')
    ax.set_xlabel('Cumulative Share of Fighters')
    ax.set_ylabel('Cumulative Share of Wins')
    ax.set_title('Lorenz Curve for Competitive Balance')
    ax.legend()
    ax.grid(True, alpha=0.3)
    plt.tight_layout()
    plt.savefig('lorenz_curve.png', dpi=150)
    plt.close()
    print("Plot saved: lorenz_curve.png")

    return gini_rounded


# ============================================================
# OPERATION 7: Linear Discriminant Analysis (Weight Class)
# ============================================================
def operation_7_lda():
    print("\n" + "=" * 60)
    print("OPERATION 7: Linear Discriminant Analysis")
    print("=" * 60)

    # ---------- Load Data Independently ----------
    stats = load_stats()

    # ---------- Select Referenced Variables ----------
    feature_cols = ['total_strikes_landed', 'takedowns_landed', 'control_time_seconds']
    target_col = 'weight_class'
    subset = stats[['fighter_id'] + feature_cols + [target_col]].copy()

    # ---------- Drop Rows with Missing Referenced Variables ----------
    subset = subset.dropna(subset=feature_cols + [target_col])

    # ---------- Prepare Features and Target ----------
    X = subset[feature_cols].values
    y = subset[target_col].values

    # ---------- Fit LDA ----------
    lda = LinearDiscriminantAnalysis(solver='svd')
    lda.fit(X, y)
    X_lda = lda.transform(X)

    # ---------- Proportion of Variance Explained by First Component ----------
    explained_variance_ratio = lda.explained_variance_ratio_
    first_component_var = round(explained_variance_ratio[0], 3)
    print(f"Proportion of Variance Explained (LD1): {first_component_var}")

    # ---------- Generate Scatter Plot ----------
    fig, ax = plt.subplots(figsize=(10, 8))
    unique_classes = np.unique(y)
    colors = plt.cm.tab10(np.linspace(0, 1, len(unique_classes)))

    for cls, color in zip(unique_classes, colors):
        mask = y == cls
        ax.scatter(X_lda[mask, 0], X_lda[mask, 1], c=[color], label=cls,
                   alpha=0.5, s=15)

    ax.set_xlabel('First Discriminant Component (LD1)')
    ax.set_ylabel('Second Discriminant Component (LD2)')
    ax.set_title('LDA: Weight-Class Separation')
    ax.legend(fontsize=7, loc='best', ncol=2)
    ax.grid(True, alpha=0.3)
    plt.tight_layout()
    plt.savefig('lda_weight_class.png', dpi=150)
    plt.close()
    print("Plot saved: lda_weight_class.png")

    return first_component_var


# ============================================================
# OPERATION 8: Striking Consistency (Coefficient of Variation)
# NOTE: The prompt specifies total_strikes_landed from Fighters Stats.xlsx,
# but Fighters Stats contains only one aggregated row per fighter, making
# per-fight CV computation infeasible. The CV requires multiple observations
# per fighter across recorded fights, so per-fight striking values from
# Fights.xlsx are used as the necessary deviation to fulfill the analytical
# intent of computing within-fighter variability across bouts.
# ============================================================
def operation_8_cv():
    print("\n" + "=" * 60)
    print("OPERATION 8: Striking Consistency (Coefficient of Variation)")
    print("=" * 60)

    # ---------- Load Data Independently ----------
    fights = load_fights()

    # ---------- Build Per-Fighter Per-Fight Striking Data ----------
    # Per-fight total_strikes_landed sourced from fight-level columns in Fights.xlsx
    # because Fighters Stats.xlsx contains only one averaged row per fighter
    fighter_red = fights[['red_fighter_id', 'STR_1']].rename(
        columns={'red_fighter_id': 'fighter_id', 'STR_1': 'total_strikes_landed'}
    )
    fighter_blue = fights[['blue_fighter_id', 'STR_2']].rename(
        columns={'blue_fighter_id': 'fighter_id', 'STR_2': 'total_strikes_landed'}
    )
    all_strikes = pd.concat([fighter_red, fighter_blue], ignore_index=True)

    # ---------- Drop Rows with Missing Referenced Variables ----------
    all_strikes = all_strikes.dropna(subset=['total_strikes_landed'])

    # ---------- Compute CV Per Fighter ----------
    grouped = all_strikes.groupby('fighter_id')['total_strikes_landed']
    cv_per_fighter = grouped.std() / grouped.mean()

    # ---------- Drop NaN (fighters with only 1 fight have undefined std) ----------
    cv_per_fighter = cv_per_fighter.dropna()

    # ---------- Report Maximum CV ----------
    max_cv = cv_per_fighter.max()
    max_cv_rounded = round(max_cv, 3)
    print(f"Maximum Coefficient of Variation: {max_cv_rounded}")

    return max_cv_rounded


# ============================================================
# OPERATION 9: Elo-PageRank Dominance Gap
# ============================================================
def operation_9_dominance_gap():
    print("\n" + "=" * 60)
    print("OPERATION 9: Elo-PageRank Dominance Gap")
    print("=" * 60)

    # ---------- Load Data Independently ----------
    fights = load_fights()
    fighters = load_fighters()

    # ---------- Derive Winner and Loser ----------
    fights = derive_winner_id(fights)
    fights = derive_loser_id(fights)

    # ---------- Filter Valid Win/Loss Fights ----------
    valid_fights = fights[
        (fights['Result_1'].isin(['W', 'L'])) &
        (fights['Result_2'].isin(['W', 'L']))
    ].copy()

    # ---------- Sort by fight_id Ascending ----------
    valid_fights = valid_fights.sort_values('fight_id').reset_index(drop=True)

    # ---------- Recompute Elo Ratings ----------
    elo_ratings = {}
    K = 32

    for _, row in valid_fights.iterrows():
        winner = row['winner_id']
        loser = row['loser_id']
        if pd.isna(winner) or pd.isna(loser):
            continue
        if winner not in elo_ratings:
            elo_ratings[winner] = 1500.0
        if loser not in elo_ratings:
            elo_ratings[loser] = 1500.0

        r_w = elo_ratings[winner]
        r_l = elo_ratings[loser]
        e_w = 1.0 / (1.0 + 10.0 ** ((r_l - r_w) / 400.0))
        e_l = 1.0 / (1.0 + 10.0 ** ((r_w - r_l) / 400.0))

        elo_ratings[winner] = r_w + K * (1.0 - e_w)
        elo_ratings[loser] = r_l + K * (0.0 - e_l)

    # ---------- Recompute PageRank ----------
    G = nx.DiGraph()
    for _, row in valid_fights.iterrows():
        winner = row['winner_id']
        loser = row['loser_id']
        if pd.notna(winner) and pd.notna(loser):
            G.add_edge(winner, loser)

    pagerank = nx.pagerank(G, alpha=0.85)

    # ---------- Rank by Elo (Descending) ----------
    elo_series = pd.Series(elo_ratings)
    elo_rank = elo_series.rank(ascending=False, method='min').astype(int)

    # ---------- Rank by PageRank (Descending) ----------
    pr_series = pd.Series(pagerank)
    pr_rank = pr_series.rank(ascending=False, method='min').astype(int)

    # ---------- Compute Dominance Gap (Elo Rank minus PageRank Rank) ----------
    common_fighters = set(elo_rank.index) & set(pr_rank.index)
    dominance_gap = {}
    for fid in common_fighters:
        dominance_gap[fid] = elo_rank[fid] - pr_rank[fid]

    # ---------- Filter for Positive Gaps Only ----------
    gap_df = pd.DataFrame({
        'fighter_id': list(dominance_gap.keys()),
        'gap': list(dominance_gap.values())
    })
    gap_df = gap_df[gap_df['gap'] > 0]

    # ---------- Find Fighter with Largest Positive Dominance Gap ----------
    gap_df = gap_df.sort_values(['gap', 'fighter_id'], ascending=[False, True])
    top_fighter_id = gap_df.iloc[0]['fighter_id']

    # ---------- Look Up Fighter Name ----------
    fighter_name = fighters.loc[
        fighters['fighter_id'] == top_fighter_id, 'fighter_name'
    ].values[0]

    print(f"Fighter with Largest Positive Dominance Gap: {fighter_name}")

    return fighter_name


# ============================================================
# MAIN EXECUTION
# ============================================================
if __name__ == '__main__':
    print("=" * 60)
    print("INTEGRATED FIGHT ANALYSIS")
    print("=" * 60)
    print()

    # ---------- Run All Operations Independently ----------
    result_1 = operation_1_elo()
    result_2 = operation_2_cox()
    result_3 = operation_3_similarity()
    result_4 = operation_4_pagerank()
    result_5 = operation_5_win_prediction()
    result_6 = operation_6_gini()
    result_7 = operation_7_lda()
    result_8 = operation_8_cv()
    result_9 = operation_9_dominance_gap()

    # ---------- Final Summary ----------
    print("\n" + "=" * 60)
    print("KEY OUTPUTS")
    print("=" * 60)
    print(f"Op 1 - Maximum Final Elo Rating:          {result_1}")
    print(f"Op 2 - Hazard Ratio (reach_cm diff):      {result_2}")
    print(f"Op 3 - Max Cosine Similarity:             {result_3}")
    print(f"Op 4 - Maximum PageRank Score:             {result_4}")
    print(f"Op 5 - Accuracy:                           {result_5}")
    print(f"Op 6 - Gini Coefficient:                   {result_6}")
    print(f"Op 7 - Variance Explained (LD1):           {result_7}")
    print(f"Op 8 - Max Coefficient of Variation:       {result_8}")
    print(f"Op 9 - Dominance Gap Fighter:              {result_9}")
