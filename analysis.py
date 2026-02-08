# ==========================================
# Integrated Employment, Groupon & Smoker Analysis Script
# Requirements: pandas, numpy, matplotlib, scipy, scikit-learn, statsmodels
# Input files: employment.xlsx, groupon.xlsx, smoker.xlsx
# Output files: employment_density.png, employment_clustering.png,
#               revenue_hexbin.png, sales_volume.png, lorenz_curve.png,
#               treatment_boxplot.png, entropy_barchart.png
# ==========================================

import warnings
warnings.filterwarnings("ignore")

import numpy as np
import pandas as pd
import matplotlib
matplotlib.use("Agg")
import matplotlib.pyplot as plt
from scipy import stats
from scipy.stats import gaussian_kde, mannwhitneyu
from sklearn.cluster import KMeans
from sklearn.metrics import silhouette_score
import statsmodels.api as sm
from statsmodels.regression.quantile_regression import QuantReg

# ==========================================================================
# SECTION 1 — Employment: Kernel Density Estimation of emp_change_ratio
# ==========================================================================

# ---------- Load Employment Data ----------
emp = pd.read_excel("employment.xlsx")
emp = emp[["state", "total_emp_feb", "total_emp_nov"]].dropna()

# ---------- Compute emp_change_ratio ----------
emp["emp_change_ratio"] = (emp["total_emp_nov"] - emp["total_emp_feb"]) / emp["total_emp_feb"]

# ---------- Kernel Density Estimation (Gaussian, Silverman bandwidth) ----------
kde = gaussian_kde(emp["emp_change_ratio"], bw_method="silverman")
x_grid = np.linspace(emp["emp_change_ratio"].min() - 0.5, emp["emp_change_ratio"].max() + 0.5, 2000)
density_vals = kde(x_grid)
peak_value = x_grid[np.argmax(density_vals)]

print("=" * 60)
print("SECTION 1: Employment Kernel Density")
print("=" * 60)
print(f"emp_change_ratio at peak density: {peak_value:.4f}")

# ---------- Density Plot ----------
fig, ax = plt.subplots(figsize=(8, 5))
ax.plot(x_grid, density_vals, color="steelblue", linewidth=2)
ax.fill_between(x_grid, density_vals, alpha=0.3, color="steelblue")
ax.axvline(peak_value, color="red", linestyle="--", label=f"Peak = {peak_value:.4f}")
ax.set_xlabel("emp_change_ratio")
ax.set_ylabel("Density")
ax.set_title("Density Plot for Employment Change")
ax.legend()
plt.tight_layout()
plt.savefig("employment_density.png", dpi=150)
plt.close()

# ==========================================================================
# SECTION 2 — Employment: K-Means Clustering on Standardised emp_change_ratio
# ==========================================================================

# ---------- Standardise using population mean & std ----------
pop_mean = emp["emp_change_ratio"].mean()
pop_std = emp["emp_change_ratio"].std(ddof=0)
emp["emp_change_std"] = (emp["emp_change_ratio"] - pop_mean) / pop_std

# ---------- K-Means (k=3, k-means++, random_state=42) ----------
X_cluster = emp[["emp_change_std"]].values
km = KMeans(n_clusters=3, init="k-means++", random_state=42, n_init=10)
emp["cluster"] = km.fit_predict(X_cluster)
sil_score = silhouette_score(X_cluster, emp["cluster"])

print()
print("=" * 60)
print("SECTION 2: Employment K-Means Clustering")
print("=" * 60)
print(f"Silhouette Score (3 clusters): {sil_score:.4f}")

# ---------- Strip Plot ----------
fig, ax = plt.subplots(figsize=(8, 5))
for cl in sorted(emp["cluster"].unique()):
    subset = emp[emp["cluster"] == cl]
    ax.scatter(
        np.full(len(subset), cl) + np.random.uniform(-0.15, 0.15, len(subset)),
        subset["emp_change_std"],
        alpha=0.5, s=20, label=f"Cluster {cl}"
    )
ax.set_xlabel("Cluster Label")
ax.set_ylabel("Standardized emp_change_ratio")
ax.set_title("Strip Plot for Employment Clustering")
ax.set_xticks(sorted(emp["cluster"].unique()))
ax.legend()
plt.tight_layout()
plt.savefig("employment_clustering.png", dpi=150)
plt.close()

# ==========================================================================
# SECTION 3 — Groupon: Quantile Regression (75th percentile) on log_revenue
# ==========================================================================

# ---------- Load Groupon Data ----------
grp = pd.read_excel("groupon.xlsx")
grp = grp[["deal_id", "treatment", "price", "discount_pct", "featured", "quantity_sold", "revenue"]].dropna()

# ---------- Compute log_revenue ----------
grp["log_revenue"] = np.log1p(grp["revenue"])

# ---------- Quantile Regression at tau=0.75 ----------
X_qr = sm.add_constant(grp[["discount_pct", "price", "featured"]])
qr_model = QuantReg(grp["log_revenue"], X_qr).fit(q=0.75)
coef_discount = qr_model.params["discount_pct"]

print()
print("=" * 60)
print("SECTION 3: Groupon Quantile Regression (75th percentile)")
print("=" * 60)
print(f"Coefficient for discount_pct: {coef_discount:.4f}")

# ---------- Hexbin Plot ----------
fig, ax = plt.subplots(figsize=(8, 5))
hb = ax.hexbin(grp["discount_pct"], grp["log_revenue"], gridsize=25, cmap="YlGnBu", mincnt=1)
plt.colorbar(hb, ax=ax, label="Count")
ax.set_xlabel("discount_pct")
ax.set_ylabel("log_revenue")
ax.set_title("Hexbin Plot for Revenue Response")
plt.tight_layout()
plt.savefig("revenue_hexbin.png", dpi=150)
plt.close()

# ==========================================================================
# SECTION 4 — Groupon: Poisson GLM for quantity_sold
# ==========================================================================

# ---------- Poisson GLM (log link) ----------
X_glm = sm.add_constant(grp[["discount_pct", "featured", "treatment"]])
poisson_model = sm.GLM(grp["quantity_sold"], X_glm, family=sm.families.Poisson(link=sm.families.links.Log())).fit()
model_deviance = poisson_model.deviance

print()
print("=" * 60)
print("SECTION 4: Groupon Poisson GLM")
print("=" * 60)
print(f"Model Deviance: {model_deviance:.4f}")

# ---------- Line Plot: mean quantity_sold by discount_pct bins ----------
grp["discount_bin"] = pd.cut(grp["discount_pct"], bins=10)
bin_means = grp.groupby("discount_bin", observed=True)["quantity_sold"].mean()
bin_midpoints = [interval.mid for interval in bin_means.index]

fig, ax = plt.subplots(figsize=(8, 5))
ax.plot(bin_midpoints, bin_means.values, marker="o", color="darkorange", linewidth=2)
ax.set_xlabel("discount_pct")
ax.set_ylabel("Mean quantity_sold")
ax.set_title("Line Plot for Sales Volume")
plt.tight_layout()
plt.savefig("sales_volume.png", dpi=150)
plt.close()

# ==========================================================================
# SECTION 5 — Groupon: Gini Coefficient of log_revenue
# ==========================================================================

# ---------- Gini (Brown–Gastwirth formulation) ----------
sorted_rev = np.sort(grp["log_revenue"].values)
n = len(sorted_rev)
index = np.arange(1, n + 1)
gini = (2 * np.sum(index * sorted_rev) - (n + 1) * np.sum(sorted_rev)) / (n * np.sum(sorted_rev))

print()
print("=" * 60)
print("SECTION 5: Groupon Gini Coefficient")
print("=" * 60)
print(f"Gini Coefficient of log_revenue: {gini:.4f}")

# ---------- Lorenz Curve ----------
cum_share_deals = np.concatenate(([0], np.cumsum(np.ones(n)) / n))
cum_share_revenue = np.concatenate(([0], np.cumsum(sorted_rev) / np.sum(sorted_rev)))

fig, ax = plt.subplots(figsize=(8, 5))
ax.plot(cum_share_deals, cum_share_revenue, color="darkgreen", linewidth=2, label="Lorenz Curve")
ax.plot([0, 1], [0, 1], color="gray", linestyle="--", label="Line of Equality")
ax.set_xlabel("Cumulative Share of Deals")
ax.set_ylabel("Cumulative Share of log_revenue")
ax.set_title("Lorenz Curve Plot for Revenue Inequality")
ax.legend()
plt.tight_layout()
plt.savefig("lorenz_curve.png", dpi=150)
plt.close()

# ==========================================================================
# SECTION 6 — Smoker: Mann–Whitney U Test (treatment vs control)
# ==========================================================================

# ---------- Load Smoker Data ----------
smk = pd.read_excel("smoker.xlsx")
smk = smk[["smoker", "treatment", "outcome"]].dropna()

# ---------- Mann–Whitney U ----------
treatment_outcomes = smk.loc[smk["treatment"] == 1, "outcome"]
control_outcomes = smk.loc[smk["treatment"] == 0, "outcome"]
u_stat, u_pval = mannwhitneyu(treatment_outcomes, control_outcomes, alternative="two-sided")

print()
print("=" * 60)
print("SECTION 6: Smoker Mann–Whitney U Test")
print("=" * 60)
print(f"U Statistic: {u_stat:.4f}")

# ---------- Boxplot ----------
fig, ax = plt.subplots(figsize=(8, 5))
data_box = [control_outcomes.values, treatment_outcomes.values]
bp = ax.boxplot(data_box, labels=["Control (0)", "Treatment (1)"], patch_artist=True)
bp["boxes"][0].set_facecolor("lightcoral")
bp["boxes"][1].set_facecolor("lightblue")
ax.set_xlabel("Treatment Group")
ax.set_ylabel("Outcome")
ax.set_title("Boxplot for Treatment Comparison")
plt.tight_layout()
plt.savefig("treatment_boxplot.png", dpi=150)
plt.close()

# ==========================================================================
# SECTION 7 — Smoker: Shannon Entropy of Outcome by Smoker Status
# ==========================================================================

# ---------- Shannon Entropy ----------
def shannon_entropy(series):
    """Compute Shannon entropy (base-2) from a discrete series."""
    counts = series.value_counts(normalize=True)
    return -np.sum(counts * np.log2(counts))

entropy_smoker_1 = shannon_entropy(smk.loc[smk["smoker"] == 1, "outcome"])
entropy_smoker_0 = shannon_entropy(smk.loc[smk["smoker"] == 0, "outcome"])
entropy_diff = entropy_smoker_1 - entropy_smoker_0

print()
print("=" * 60)
print("SECTION 7: Smoker Shannon Entropy")
print("=" * 60)
print(f"Entropy (smoker=1): {entropy_smoker_1:.4f}")
print(f"Entropy (smoker=0): {entropy_smoker_0:.4f}")
print(f"Entropy Difference (smoker=1 minus smoker=0): {entropy_diff:.4f}")

# ---------- Bar Chart ----------
fig, ax = plt.subplots(figsize=(8, 5))
ax.bar(["smoker=0", "smoker=1"], [entropy_smoker_0, entropy_smoker_1],
       color=["salmon", "skyblue"], edgecolor="black")
ax.set_xlabel("Smoker Status")
ax.set_ylabel("Shannon Entropy")
ax.set_title("Bar Chart for Outcome Uncertainty")
plt.tight_layout()
plt.savefig("entropy_barchart.png", dpi=150)
plt.close()

# ==========================================================================
# SECTION 8 — Smoker: Which Treatment Group Has Higher Median Outcome?
# ==========================================================================

# ---------- Median Outcome by Treatment Group ----------
median_treatment = smk.groupby("treatment")["outcome"].median()
higher_group = median_treatment.idxmax()

print()
print("=" * 60)
print("SECTION 8: Higher Median Outcome Group")
print("=" * 60)
print(f"Median outcome (control=0): {median_treatment[0]:.4f}")
print(f"Median outcome (treatment=1): {median_treatment[1]:.4f}")
print(f"Group with higher median outcome: {higher_group}")

print()
print("=" * 60)
print("ALL PLOTS SAVED SUCCESSFULLY")
print("=" * 60)
