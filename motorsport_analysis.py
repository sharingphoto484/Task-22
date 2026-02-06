# ==========================================
# Integrated Motorsport Data Analysis Script
# Requirements: pandas, numpy, matplotlib, scipy, scikit-learn
# Input files: qualifying.xlsx, results.xlsx, pit_stops.xlsx
# Output files: race_scoring_trends.png, qualifying_vs_race_outcome.png,
#               pit_stop_duration_histogram.png, pit_stop_duration_boxplot.png
# ==========================================

import pandas as pd
import numpy as np
import matplotlib
matplotlib.use("Agg")
import matplotlib.pyplot as plt
from scipy import stats
from sklearn.linear_model import LinearRegression
from sklearn.metrics import mean_absolute_error

# ---------- Load Excel Files ----------
results_raw = pd.read_excel("results.xlsx")
qualifying_raw = pd.read_excel("qualifying.xlsx")
pit_stops_raw = pd.read_excel("pit_stops.xlsx")

# ---------- Helper: Convert to Numeric and Drop Invalid Rows ----------
def clean_numeric(df, cols):
    """Convert specified columns to numeric; drop rows where any
    referenced column is missing or non-numeric."""
    df = df.copy()
    for c in cols:
        df[c] = pd.to_numeric(df[c], errors="coerce")
    return df.dropna(subset=cols)

# =====================================================
# Task 1: Pearson Correlation – Grid vs Finishing Position
# (results.xlsx only)
# =====================================================
# ---------- Sort by raceId ----------
res1 = results_raw.sort_values("raceId").copy()
# ---------- Clean referenced columns ----------
res1 = clean_numeric(res1, ["grid", "position"])
# ---------- Compute Pearson r ----------
corr1, _ = stats.pearsonr(res1["grid"], res1["position"])
print(f"Task 1  – Pearson r (grid vs finishing): {corr1:.4f}")

# =====================================================
# Task 2: OLS Regression MAE – Finishing ~ Grid
# (results.xlsx only)
# =====================================================
# ---------- Sort by raceId ----------
res2 = results_raw.sort_values("raceId").copy()
# ---------- Clean referenced columns ----------
res2 = clean_numeric(res2, ["grid", "position"])
# ---------- Fit OLS and compute MAE ----------
lr2 = LinearRegression().fit(res2[["grid"]], res2["position"])
mae2 = mean_absolute_error(res2["position"], lr2.predict(res2[["grid"]]))
print(f"Task 2  – MAE (finishing ~ grid OLS): {mae2:.4f}")

# =====================================================
# Task 3: Median Points for Race Winners
# (results.xlsx only)
# =====================================================
# ---------- Sort by raceId ----------
res3 = results_raw.sort_values("raceId").copy()
# ---------- Clean referenced columns ----------
res3 = clean_numeric(res3, ["position", "points"])
# ---------- Filter winners and compute median ----------
median3 = res3.loc[res3["position"] == 1, "points"].median()
print(f"Task 3  – Median points (finishing == 1): {median3:.4f}")

# =====================================================
# Task 4: Maximum Laps Completed
# (results.xlsx only)
# =====================================================
# ---------- Sort by raceId ----------
res4 = results_raw.sort_values("raceId").copy()
# ---------- Clean referenced columns ----------
res4 = clean_numeric(res4, ["laps"])
# ---------- Compute max ----------
max4 = int(res4["laps"].max())
print(f"Task 4  – Max laps completed: {max4}")

# =====================================================
# Task 5: Min of Constructor-Level Mean Qualifying Position
# (qualifying.xlsx only)
# =====================================================
# ---------- Sort by raceId ----------
qual5 = qualifying_raw.sort_values("raceId").copy()
# ---------- Clean referenced columns ----------
qual5 = clean_numeric(qual5, ["position"])
# ---------- Mean per constructor, then min ----------
min5 = qual5.groupby("constructorId")["position"].mean().min()
print(f"Task 5  – Min constructor mean qualifying position: {min5:.4f}")

# =====================================================
# Task 6: DriverId with Smallest Mean Qualifying Position
# (qualifying.xlsx only)
# =====================================================
# ---------- Sort by raceId ----------
qual6 = qualifying_raw.sort_values("raceId").copy()
# ---------- Clean referenced columns ----------
qual6 = clean_numeric(qual6, ["position"])
# ---------- Mean per driver, pick smallest (ties: smallest driverId) ----------
mean_qual6 = qual6.groupby("driverId")["position"].mean()
best_val6 = mean_qual6.min()
driver6 = int(mean_qual6[mean_qual6 == best_val6].index.min())
print(f"Task 6  – DriverId with smallest mean qualifying pos: {driver6}")

# =====================================================
# Task 7: Mean Pit Stop Count per Driver-Race
# (pit_stops.xlsx only)
# =====================================================
# ---------- Sort by raceId ----------
pit7 = pit_stops_raw.sort_values("raceId").copy()
# ---------- Count stops per (raceId, driverId) then mean ----------
counts7 = pit7.groupby(["raceId", "driverId"]).size()
mean7 = counts7.mean()
print(f"Task 7  – Mean pit stop count: {mean7:.4f}")

# =====================================================
# Task 8: RaceId with Largest Total Pit Stop Duration
# (pit_stops.xlsx only)
# =====================================================
# ---------- Sort by raceId ----------
pit8 = pit_stops_raw.sort_values("raceId").copy()
# ---------- Clean referenced columns ----------
pit8 = clean_numeric(pit8, ["milliseconds"])
# ---------- Sum per race, pick max (ties: smallest raceId) ----------
total8 = pit8.groupby("raceId")["milliseconds"].sum()
max_total8 = total8.max()
race8 = int(total8[total8 == max_total8].index.min())
print(f"Task 8  – RaceId with largest total pit duration: {race8}")

# =====================================================
# Task 9: Correlation – Total Pit Duration vs Finishing Position
# (pit_stops.xlsx + results.xlsx)
# =====================================================
# ---------- Aggregate pit stops per driver-race ----------
pit9 = pit_stops_raw.sort_values("raceId").copy()
pit9 = clean_numeric(pit9, ["milliseconds"])
pit9_agg = pit9.groupby(["raceId", "driverId"])["milliseconds"].sum().reset_index(
    name="total_pit_ms"
)
# ---------- Prepare results ----------
res9 = results_raw.sort_values("raceId").copy()
res9 = clean_numeric(res9, ["position"])
# ---------- Inner join ----------
m9 = pd.merge(
    pit9_agg,
    res9[["raceId", "driverId", "position"]],
    on=["raceId", "driverId"],
)
m9 = clean_numeric(m9, ["total_pit_ms", "position"])
# ---------- Pearson r ----------
corr9, _ = stats.pearsonr(m9["total_pit_ms"], m9["position"])
print(f"Task 9  – Pearson r (total pit duration vs finishing): {corr9:.4f}")

# =====================================================
# Task 10: Mean of (Grid Position – Qualifying Position)
# (qualifying.xlsx + results.xlsx)
# =====================================================
# ---------- Prepare qualifying ----------
qual10 = qualifying_raw.sort_values("raceId").copy()
qual10 = clean_numeric(qual10, ["position"])
qual10 = qual10.rename(columns={"position": "qual_position"})
# ---------- Prepare results ----------
res10 = results_raw.sort_values("raceId").copy()
res10 = clean_numeric(res10, ["grid"])
# ---------- Inner join ----------
m10 = pd.merge(
    qual10[["raceId", "driverId", "qual_position"]],
    res10[["raceId", "driverId", "grid"]],
    on=["raceId", "driverId"],
)
m10 = clean_numeric(m10, ["grid", "qual_position"])
# ---------- Compute mean difference ----------
mean10 = (m10["grid"] - m10["qual_position"]).mean()
print(f"Task 10 – Mean (grid − qualifying position): {mean10:.4f}")

# =====================================================
# Task 11: DriverId with Highest Mean Improvement Rate
# (qualifying.xlsx + results.xlsx)
# =====================================================
# ---------- Prepare qualifying ----------
qual11 = qualifying_raw.sort_values("raceId").copy()
qual11 = clean_numeric(qual11, ["position"])
qual11 = qual11.rename(columns={"position": "qual_position"})
# ---------- Prepare results ----------
res11 = results_raw.sort_values("raceId").copy()
res11 = clean_numeric(res11, ["grid", "position"])
res11 = res11.rename(columns={"position": "fin_position"})
# ---------- Inner join ----------
m11 = pd.merge(
    qual11[["raceId", "driverId", "qual_position"]],
    res11[["raceId", "driverId", "grid", "fin_position"]],
    on=["raceId", "driverId"],
)
m11 = clean_numeric(m11, ["grid", "fin_position"])
# ---------- Improvement indicator ----------
m11["improved"] = (m11["fin_position"] < m11["grid"]).astype(int)
# ---------- Mean per driver, pick highest (ties: smallest driverId) ----------
imp_rate = m11.groupby("driverId")["improved"].mean()
best_rate = imp_rate.max()
driver11 = int(imp_rate[imp_rate == best_rate].index.min())
print(f"Task 11 – DriverId with highest improvement rate: {driver11}")

# =====================================================
# Task 12: Three-File OLS – Predictor with Larger |Standardized β|
# (qualifying.xlsx + results.xlsx + pit_stops.xlsx)
# =====================================================
# ---------- Prepare qualifying ----------
qual12 = qualifying_raw.sort_values("raceId").copy()
qual12 = clean_numeric(qual12, ["position"])
qual12 = qual12.rename(columns={"position": "qual_position"})
# ---------- Prepare pit stops (total duration per driver-race) ----------
pit12 = pit_stops_raw.sort_values("raceId").copy()
pit12 = clean_numeric(pit12, ["milliseconds"])
pit12_agg = pit12.groupby(["raceId", "driverId"])["milliseconds"].sum().reset_index(
    name="total_pit_ms"
)
# ---------- Prepare results ----------
res12 = results_raw.sort_values("raceId").copy()
res12 = clean_numeric(res12, ["points"])
# ---------- Inner join all three ----------
m12 = pd.merge(
    qual12[["raceId", "driverId", "qual_position"]],
    res12[["raceId", "driverId", "points"]],
    on=["raceId", "driverId"],
)
m12 = pd.merge(m12, pit12_agg, on=["raceId", "driverId"])
m12 = clean_numeric(m12, ["qual_position", "total_pit_ms", "points"])
# ---------- Standardize predictors (z-score) ----------
m12["z_qual"] = (m12["qual_position"] - m12["qual_position"].mean()) / m12[
    "qual_position"
].std()
m12["z_pit"] = (m12["total_pit_ms"] - m12["total_pit_ms"].mean()) / m12[
    "total_pit_ms"
].std()
# ---------- Fit OLS ----------
lr12 = LinearRegression().fit(m12[["z_qual", "z_pit"]], m12["points"])
abs_coef_qual = abs(lr12.coef_[0])
abs_coef_pit = abs(lr12.coef_[1])
# ---------- Report predictor (ties favor qualifying) ----------
predictor12 = (
    "qualifying position" if abs_coef_qual >= abs_coef_pit
    else "total pit stop duration"
)
print(f"Task 12 – Predictor with larger |standardized β|: {predictor12}")

# =====================================================
# Task 13: Binary Indicator – |β_qual| > |β_pit|
# =====================================================
indicator13 = 1 if abs_coef_qual > abs_coef_pit else 0
print(f"Task 13 – |β_qual| > |β_pit| indicator: {indicator13}")

# =====================================================
# Plot 1: Line Plot – Race-Level Scoring Trends
# (results.xlsx)
# =====================================================
# ---------- Sort and clean ----------
res_p1 = results_raw.sort_values("raceId").copy()
res_p1 = clean_numeric(res_p1, ["points"])
# ---------- Mean points per race ----------
race_mean_pts = res_p1.groupby("raceId")["points"].mean()
# ---------- Draw line plot ----------
plt.figure(figsize=(12, 5))
plt.plot(race_mean_pts.index, race_mean_pts.values, linewidth=0.8)
plt.xlabel("raceId")
plt.ylabel("Mean Points")
plt.title("Race-Level Scoring Trends")
plt.tight_layout()
plt.savefig("race_scoring_trends.png", dpi=150)
plt.close()
print("\nPlot saved: race_scoring_trends.png")

# =====================================================
# Plot 2: Scatter Plot – Qualifying vs Race Outcome
# (qualifying.xlsx + results.xlsx)
# =====================================================
# ---------- Prepare qualifying ----------
qual_p2 = qualifying_raw.sort_values("raceId").copy()
qual_p2 = clean_numeric(qual_p2, ["position"])
qual_p2 = qual_p2.rename(columns={"position": "qual_position"})
# ---------- Prepare results ----------
res_p2 = results_raw.sort_values("raceId").copy()
res_p2 = clean_numeric(res_p2, ["position"])
res_p2 = res_p2.rename(columns={"position": "fin_position"})
# ---------- Inner join ----------
m_p2 = pd.merge(
    qual_p2[["raceId", "driverId", "qual_position"]],
    res_p2[["raceId", "driverId", "fin_position"]],
    on=["raceId", "driverId"],
)
# ---------- Draw scatter plot ----------
plt.figure(figsize=(8, 8))
plt.scatter(m_p2["qual_position"], m_p2["fin_position"], alpha=0.3, s=10)
plt.xlabel("Qualifying Position")
plt.ylabel("Finishing Position")
plt.title("Qualifying Performance vs Race Outcome")
plt.tight_layout()
plt.savefig("qualifying_vs_race_outcome.png", dpi=150)
plt.close()
print("Plot saved: qualifying_vs_race_outcome.png")

# =====================================================
# Plot 3: Histogram – Pit Stop Duration Distribution
# (pit_stops.xlsx)
# =====================================================
# ---------- Sort and clean ----------
pit_p3 = pit_stops_raw.sort_values("raceId").copy()
pit_p3 = clean_numeric(pit_p3, ["milliseconds"])
# ---------- Draw histogram ----------
plt.figure(figsize=(10, 5))
plt.hist(pit_p3["milliseconds"], bins=50, edgecolor="black", alpha=0.7)
plt.xlabel("Pit Stop Duration (ms)")
plt.ylabel("Frequency")
plt.title("Pit Stop Duration Distribution")
plt.tight_layout()
plt.savefig("pit_stop_duration_histogram.png", dpi=150)
plt.close()
print("Plot saved: pit_stop_duration_histogram.png")

# =====================================================
# Plot 4: Box Plot – Pit Stop Duration by Driver
# (pit_stops.xlsx)
# =====================================================
# ---------- Sort and clean ----------
pit_p4 = pit_stops_raw.sort_values("raceId").copy()
pit_p4 = clean_numeric(pit_p4, ["milliseconds"])
# ---------- Prepare box data per driverId ----------
sorted_drivers = sorted(pit_p4["driverId"].unique())
box_data = [
    pit_p4.loc[pit_p4["driverId"] == d, "milliseconds"].values
    for d in sorted_drivers
]
# ---------- Draw box plot ----------
fig, ax = plt.subplots(figsize=(max(14, len(sorted_drivers) * 0.25), 6))
ax.boxplot(box_data, tick_labels=[str(d) for d in sorted_drivers])
ax.set_xlabel("driverId")
ax.set_ylabel("Pit Stop Duration (ms)")
ax.set_title("Pit Stop Duration by Driver")
plt.xticks(rotation=90, fontsize=5)
plt.tight_layout()
plt.savefig("pit_stop_duration_boxplot.png", dpi=150)
plt.close()
print("Plot saved: pit_stop_duration_boxplot.png")

print("\nAnalysis complete.")
