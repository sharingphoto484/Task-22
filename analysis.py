# ==========================================
# Integrated Employment, Groupon & Smoker Analysis Script
# Requirements: pandas, numpy, matplotlib, scipy, scikit-learn
# Input files: employment.xlsx, groupon.xlsx, smoker.xlsx
# Output files: qq_plot.png, ecdf_plot.png, rank_rank_plot.png,
#               density_plot.png, lorenz_curve.png, mosaic_plot.png,
#               calibration_plot.png
# ==========================================

import pandas as pd
import numpy as np
import matplotlib
matplotlib.use("Agg")
import matplotlib.pyplot as plt
from scipy import stats
from scipy.stats import shapiro, kstest, zscore, spearmanr, chi2_contingency
from sklearn.linear_model import LogisticRegression

# ---------- 1. Load Employment Data ----------
emp = pd.read_excel("employment.xlsx")
emp = emp[["state", "total_emp_feb", "total_emp_nov"]].dropna()
emp["emp_change_ratio"] = (emp["total_emp_nov"] - emp["total_emp_feb"]) / emp["total_emp_feb"]

# ---------- 2. Shapiro–Wilk Normality Test on emp_change_ratio ----------
sw_stat, sw_p = shapiro(emp["emp_change_ratio"])
print(f"Shapiro-Wilk test statistic: {sw_stat:.6f}")

# ---------- 3. Q–Q Plot for Employment Change ----------
fig, ax = plt.subplots(figsize=(7, 5))
stats.probplot(emp["emp_change_ratio"], dist="norm", plot=ax)
ax.set_title("Q-Q Plot for Employment Change")
ax.set_xlabel("Theoretical Quantiles")
ax.set_ylabel("emp_change_ratio Quantiles")
plt.tight_layout()
plt.savefig("qq_plot.png", dpi=150)
plt.close()
print("Saved: qq_plot.png")

# ---------- 4. One-Sample KS Test (Standard Normal) on z-scored emp_change_ratio ----------
emp_z = zscore(emp["emp_change_ratio"])
ks_stat, ks_p = kstest(emp_z, "norm")
print(f"KS test D statistic: {ks_stat:.6f}")

# ---------- 5. Empirical CDF Plot for Standardized emp_change_ratio ----------
sorted_z = np.sort(emp_z)
ecdf_y = np.arange(1, len(sorted_z) + 1) / len(sorted_z)

fig, ax = plt.subplots(figsize=(7, 5))
ax.step(sorted_z, ecdf_y, where="post", linewidth=1.5, label="Empirical CDF")
x_norm = np.linspace(sorted_z.min() - 0.5, sorted_z.max() + 0.5, 300)
ax.plot(x_norm, stats.norm.cdf(x_norm), "r--", linewidth=1.2, label="Standard Normal CDF")
ax.set_title("Empirical CDF for Standardized emp_change_ratio")
ax.set_xlabel("Standardized emp_change_ratio")
ax.set_ylabel("Cumulative Probability")
ax.legend()
plt.tight_layout()
plt.savefig("ecdf_plot.png", dpi=150)
plt.close()
print("Saved: ecdf_plot.png")

# ---------- 6. Load Groupon Data ----------
grp = pd.read_excel("groupon.xlsx")
grp = grp[["deal_id", "price", "discount_pct", "featured", "quantity_sold", "revenue"]].dropna()
grp["log_revenue"] = np.log1p(grp["revenue"])

# ---------- 7. Spearman Rank Correlation: discount_pct vs log_revenue ----------
spear_corr, spear_p = spearmanr(grp["discount_pct"], grp["log_revenue"])
print(f"Spearman correlation coefficient: {spear_corr:.6f}")

# ---------- 8. Rank–Rank Plot for Pricing Response ----------
disc_rank = grp["discount_pct"].rank()
rev_rank = grp["log_revenue"].rank()

fig, ax = plt.subplots(figsize=(7, 5))
ax.scatter(disc_rank, rev_rank, alpha=0.4, s=15, edgecolors="none")
ax.set_title("Rank-Rank Plot for Pricing Response")
ax.set_xlabel("discount_pct Rank")
ax.set_ylabel("log_revenue Rank")
plt.tight_layout()
plt.savefig("rank_rank_plot.png", dpi=150)
plt.close()
print("Saved: rank_rank_plot.png")

# ---------- 9. Two-Sample KS Test: log_revenue by featured status ----------
feat = grp[grp["featured"] == 1]["log_revenue"]
nofeat = grp[grp["featured"] == 0]["log_revenue"]
ks2_stat, ks2_p = stats.ks_2samp(feat, nofeat)
print(f"Two-sample KS D statistic (featured vs non-featured): {ks2_stat:.6f}")

# ---------- 10. Overlaid Density Plot for Deal Revenue ----------
fig, ax = plt.subplots(figsize=(7, 5))
feat_vals = feat.values
nofeat_vals = nofeat.values
x_min = min(feat_vals.min(), nofeat_vals.min())
x_max = max(feat_vals.max(), nofeat_vals.max())
xs = np.linspace(x_min - 0.5, x_max + 0.5, 500)

kde_feat = stats.gaussian_kde(feat_vals)
kde_nofeat = stats.gaussian_kde(nofeat_vals)
ax.plot(xs, kde_feat(xs), label="Featured", linewidth=1.5)
ax.fill_between(xs, kde_feat(xs), alpha=0.25)
ax.plot(xs, kde_nofeat(xs), label="Non-Featured", linewidth=1.5)
ax.fill_between(xs, kde_nofeat(xs), alpha=0.25)
ax.set_title("Overlaid Density Plot for Deal Revenue")
ax.set_xlabel("log_revenue")
ax.set_ylabel("Density")
ax.legend()
plt.tight_layout()
plt.savefig("density_plot.png", dpi=150)
plt.close()
print("Saved: density_plot.png")

# ---------- 11. Theil Index of Inequality on quantity_sold ----------
q = grp["quantity_sold"].values.astype(float)
q = q[q > 0]
n = len(q)
mu = q.mean()
theil = np.sum((q / mu) * np.log(q / mu)) / n
print(f"Theil index: {theil:.6f}")

# ---------- 12. Lorenz Curve for Sales Inequality ----------
q_sorted = np.sort(q)
cum_share_deals = np.arange(1, n + 1) / n
cum_share_qty = np.cumsum(q_sorted) / q_sorted.sum()

fig, ax = plt.subplots(figsize=(7, 5))
ax.plot(cum_share_deals, cum_share_qty, linewidth=1.5, label="Lorenz Curve")
ax.plot([0, 1], [0, 1], "r--", linewidth=1, label="Line of Equality")
ax.set_title("Lorenz Curve for Sales Inequality")
ax.set_xlabel("Cumulative Share of Deals")
ax.set_ylabel("Cumulative Share of quantity_sold")
ax.legend()
plt.tight_layout()
plt.savefig("lorenz_curve.png", dpi=150)
plt.close()
print("Saved: lorenz_curve.png")

# ---------- 13. Load Smoker Data ----------
smk = pd.read_excel("smoker.xlsx")
smk = smk[["smoker", "treatment", "outcome"]].dropna()

# ---------- 14. Chi-Square Test of Independence: smoker vs treatment ----------
ct = pd.crosstab(smk["smoker"], smk["treatment"])
chi2, chi2_p, dof, expected = chi2_contingency(ct)
print(f"Chi-square statistic: {chi2:.6f}")

# ---------- 15. Mosaic Plot for Treatment Assignment ----------
from matplotlib.patches import Rectangle

fig, ax = plt.subplots(figsize=(7, 5))
ct_norm = ct.copy().astype(float)
totals = ct_norm.sum(axis=1)
smoker_labels = ct_norm.index.tolist()
treatment_labels = ct_norm.columns.tolist()

widths = (totals / totals.sum()).values
colors = plt.cm.Set2.colors

x_start = 0.0
for i, sl in enumerate(smoker_labels):
    w = widths[i]
    col_total = ct_norm.loc[sl].sum()
    y_start = 0.0
    for j, tl in enumerate(treatment_labels):
        h = ct_norm.loc[sl, tl] / col_total
        rect = Rectangle((x_start, y_start), w, h, linewidth=1,
                          edgecolor="black", facecolor=colors[j % len(colors)])
        ax.add_patch(rect)
        ax.text(x_start + w / 2, y_start + h / 2,
                f"T={tl}\n{int(ct_norm.loc[sl, tl])}",
                ha="center", va="center", fontsize=8)
        y_start += h
    ax.text(x_start + w / 2, -0.05, f"Smoker={sl}", ha="center", va="top", fontsize=9)
    x_start += w

ax.set_xlim(0, 1)
ax.set_ylim(-0.1, 1)
ax.set_title("Mosaic Plot for Treatment Assignment")
ax.set_xlabel("Smoker Status")
ax.set_ylabel("Treatment Group")
ax.set_xticks([])
ax.set_yticks([0, 0.5, 1])
plt.tight_layout()
plt.savefig("mosaic_plot.png", dpi=150)
plt.close()
print("Saved: mosaic_plot.png")

# ---------- 16. Logistic Regression: treatment ~ smoker ----------
X = smk[["smoker"]].values
y = smk["treatment"].values
lr = LogisticRegression(solver="lbfgs", max_iter=1000)
lr.fit(X, y)
coef_smoker = lr.coef_[0][0]
intercept = lr.intercept_[0]
print(f"Logistic regression coefficient for smoker: {coef_smoker:.6f}")

# ---------- 17. Calibration Plot for Treatment Probability ----------
probs = lr.predict_proba(X)[:, 1]
smk_temp = smk.copy()
smk_temp["pred_prob"] = probs

unique_probs = sorted(smk_temp["pred_prob"].unique())
obs_freq = []
for p in unique_probs:
    subset = smk_temp[smk_temp["pred_prob"] == p]
    obs_freq.append(subset["treatment"].mean())

fig, ax = plt.subplots(figsize=(7, 5))
ax.scatter(unique_probs, obs_freq, s=80, zorder=5, edgecolors="black")
ax.plot([0, 1], [0, 1], "r--", linewidth=1, label="Perfect Calibration")
ax.set_title("Calibration Plot for Treatment Probability")
ax.set_xlabel("Predicted Probability")
ax.set_ylabel("Observed Treatment Frequency")
ax.legend()
plt.tight_layout()
plt.savefig("calibration_plot.png", dpi=150)
plt.close()
print("Saved: calibration_plot.png")

# ---------- 18. Interpret Association Direction ----------
prob_smoker = 1 / (1 + np.exp(-(intercept + coef_smoker * 1)))
prob_nonsmoker = 1 / (1 + np.exp(-(intercept + coef_smoker * 0)))
if prob_smoker > prob_nonsmoker:
    print("Smoker status is associated with a HIGHER predicted probability of treatment.")
else:
    print("Non-smoker status is associated with a HIGHER predicted probability of treatment.")
