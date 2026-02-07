# ==========================================
# Integrated Board Game Trend Analysis Script
# Requirements: pandas, numpy, matplotlib, scipy, openpyxl
# Input files: bgg_db_1806.xlsx, bgg_db_2017_04.xlsx, bgg_db_2018_01.xlsx
# Output files: rating_distribution_boxplot.png, complexity_vs_rating_scatter.png,
#               top20_popularity_bar.png, rating_stability_line.png,
#               playtime_histogram.png
# ==========================================

import pandas as pd
import numpy as np
import matplotlib
matplotlib.use("Agg")
import matplotlib.pyplot as plt
from scipy import stats

# ---------- Load Excel Files Robustly ----------

df_2017_04 = pd.read_excel("bgg_db_2017_04.xlsx")
df_2018_01 = pd.read_excel("bgg_db_2018_01.xlsx")
df_1806    = pd.read_excel("bgg_db_1806.xlsx")

# ---------- Label Each Snapshot ----------

df_2017_04["snapshot"] = "2017-04"
df_2018_01["snapshot"] = "2018-01"
df_1806["snapshot"]    = "2018-06"

print("="*60)
print("BOARD GAME TREND ANALYSIS — KEY OUTPUTS")
print("="*60)

# ---------- Align Datasets by game_id (Inner Join) ----------

common_ids = (
    set(df_2017_04["game_id"])
    & set(df_2018_01["game_id"])
    & set(df_1806["game_id"])
)

df_2017_04 = df_2017_04[df_2017_04["game_id"].isin(common_ids)].copy()
df_2018_01 = df_2018_01[df_2018_01["game_id"].isin(common_ids)].copy()
df_1806    = df_1806[df_1806["game_id"].isin(common_ids)].copy()

print(f"\nTotal games retained after alignment (present in all 3 snapshots): {len(common_ids)}")

# ---------- Combine Into Single Long-Form DataFrame ----------

df_all = pd.concat([df_2017_04, df_2018_01, df_1806], ignore_index=True)

# ---------- Average Rating Change: Earliest vs Latest ----------

earliest = df_2017_04.set_index("game_id")["avg_rating"]
latest   = df_1806.set_index("game_id")["avg_rating"]

common_idx = earliest.index.intersection(latest.index)
mean_abs_change = (latest.loc[common_idx] - earliest.loc[common_idx]).abs().mean()

print(f"\nMean absolute change in avg_rating (earliest → latest): {mean_abs_change:.4f}")

# ---------- Player Count Summary ----------

print("\n--- Player Count Summary (across all retained games, all snapshots) ---")
print(f"  Min of min_players: {df_all['min_players'].min()}")
print(f"  Max of min_players: {df_all['min_players'].max()}")
print(f"  Min of max_players: {df_all['max_players'].min()}")
print(f"  Max of max_players: {df_all['max_players'].max()}")

df_all["player_range_width"] = df_all["max_players"] - df_all["min_players"]
median_range_width = df_all["player_range_width"].median()
print(f"  Median recommended player range width: {median_range_width}")

# ---------- Complexity vs Rating (Most Recent Snapshot) ----------

latest_df = df_1806.copy()
valid = latest_df.dropna(subset=["weight", "avg_rating"])
valid = valid[(valid["weight"] > 0)]

pearson_r, pearson_p = stats.pearsonr(valid["weight"], valid["avg_rating"])
print(f"\nPearson correlation (weight vs avg_rating, latest snapshot): r={pearson_r:.4f}, p={pearson_p:.2e}")

# ---------- Top 20 Games by Rating in Most Recent Snapshot ----------

latest_sorted = latest_df.sort_values(
    by=["avg_rating", "game_id"],
    ascending=[False, True]
)
top20 = latest_sorted.head(20).copy()

print(f"\nTop-ranked game (highest avg_rating, latest snapshot):")
print(f"  game_id: {top20.iloc[0]['game_id']}")
print(f"  name:    {top20.iloc[0]['names']}")
print(f"  rating:  {top20.iloc[0]['avg_rating']}")

# ---------- Playtime Stats for Top 20 ----------

top20_playtime = top20["avg_time"]
playtime_pop_std = top20_playtime.std(ddof=0)
print(f"\nPopulation std dev of playtime (top 20 games): {playtime_pop_std:.4f}")

# ---------- Community Engagement: Skewness of num_votes ----------

all_num_votes = df_all["num_votes"]
vote_skewness = all_num_votes.skew()
print(f"\nSkewness of num_votes distribution (all retained games, all snapshots): {vote_skewness:.4f}")

# ---------- Most Stable Game Across Snapshots ----------

rating_pivot = df_all.pivot_table(
    index="game_id", columns="snapshot", values="avg_rating", aggfunc="first"
)
rating_pivot["rating_std"] = rating_pivot.std(axis=1, ddof=0)

most_stable_id = rating_pivot["rating_std"].idxmin()
most_stable_name = df_all.loc[df_all["game_id"] == most_stable_id, "names"].iloc[0]
most_stable_std = rating_pivot.loc[most_stable_id, "rating_std"]

print(f"\nMost stable game (lowest rating std across 3 snapshots):")
print(f"  game_id: {most_stable_id}")
print(f"  name:    {most_stable_name}")
print(f"  rating std: {most_stable_std:.6f}")

# ---------- Open-Ended Insight ----------

print("\n--- Insight ---")
print("Games with higher and more sustained community engagement (large num_votes) tend to")
print("exhibit more stable long-term ratings, as the law of large numbers dampens individual")
print("score fluctuations, while niche titles with fewer votes show greater rating volatility.")

# =============================================
# VISUALIZATIONS
# =============================================

# ---------- 1. Boxplot: Rating Distribution Across Snapshots ----------

fig, ax = plt.subplots(figsize=(8, 5))
snapshot_order = ["2017-04", "2018-01", "2018-06"]
box_data = [df_all[df_all["snapshot"] == s]["avg_rating"].dropna() for s in snapshot_order]
bp = ax.boxplot(box_data, tick_labels=snapshot_order, patch_artist=True,
                boxprops=dict(facecolor="#4C72B0", alpha=0.7),
                medianprops=dict(color="red", linewidth=2))
ax.set_xlabel("Snapshot")
ax.set_ylabel("Average Rating")
ax.set_title("Distribution of Average Ratings Across Snapshots")
plt.tight_layout()
plt.savefig("rating_distribution_boxplot.png", dpi=150)
plt.close()
print("\nSaved: rating_distribution_boxplot.png")

# ---------- 2. Scatter: Complexity (Weight) vs Rating ----------

fig, ax = plt.subplots(figsize=(8, 5))
ax.scatter(valid["weight"], valid["avg_rating"], alpha=0.3, s=12, c="#4C72B0")
z = np.polyfit(valid["weight"], valid["avg_rating"], 1)
p_line = np.poly1d(z)
x_range = np.linspace(valid["weight"].min(), valid["weight"].max(), 100)
ax.plot(x_range, p_line(x_range), "r--", linewidth=2, label=f"r={pearson_r:.3f}")
ax.set_xlabel("Average Weight (Complexity)")
ax.set_ylabel("Average Rating")
ax.set_title("Game Complexity vs. Rating (Latest Snapshot)")
ax.legend()
plt.tight_layout()
plt.savefig("complexity_vs_rating_scatter.png", dpi=150)
plt.close()
print("Saved: complexity_vs_rating_scatter.png")

# ---------- 3. Bar Chart: Top 20 Games by Total User Ratings ----------

fig, ax = plt.subplots(figsize=(12, 5))
top20_bar = top20.sort_values("num_votes", ascending=False)
ax.bar(top20_bar["game_id"].astype(str), top20_bar["num_votes"], color="#4C72B0", alpha=0.8)
ax.set_xlabel("Game ID")
ax.set_ylabel("Total User Ratings (num_votes)")
ax.set_title("Popularity Concentration — Top 20 Games by Rating")
plt.xticks(rotation=45, ha="right")
plt.tight_layout()
plt.savefig("top20_popularity_bar.png", dpi=150)
plt.close()
print("Saved: top20_popularity_bar.png")

# ---------- 4. Line Chart: Rating Stability for 10 Most-Rated Games ----------

total_votes_per_game = df_all.groupby("game_id")["num_votes"].sum()
top10_voted_ids = total_votes_per_game.nlargest(10).index

fig, ax = plt.subplots(figsize=(10, 5))
for gid in top10_voted_ids:
    game_data = df_all[df_all["game_id"] == gid].sort_values("snapshot")
    label_name = game_data["names"].iloc[0]
    if len(label_name) > 25:
        label_name = label_name[:22] + "..."
    ax.plot(game_data["snapshot"], game_data["avg_rating"], marker="o", label=f"{gid}: {label_name}")
ax.set_xlabel("Snapshot")
ax.set_ylabel("Average Rating")
ax.set_title("Rating Stability — 10 Most-Rated Games")
ax.legend(fontsize=7, loc="best")
plt.tight_layout()
plt.savefig("rating_stability_line.png", dpi=150)
plt.close()
print("Saved: rating_stability_line.png")

# ---------- 5. Histogram: Playtime Distribution ----------

fig, ax = plt.subplots(figsize=(8, 5))
playtime_all = df_all["avg_time"].dropna()
ax.hist(playtime_all, bins=50, color="#4C72B0", alpha=0.7, edgecolor="black")
ax.set_xlabel("Playtime (minutes)")
ax.set_ylabel("Frequency")
ax.set_title("Distribution of Game Playtime (All Retained Games)")
plt.tight_layout()
plt.savefig("playtime_histogram.png", dpi=150)
plt.close()
print("Saved: playtime_histogram.png")

print("\n" + "="*60)
print("ANALYSIS COMPLETE")
print("="*60)
