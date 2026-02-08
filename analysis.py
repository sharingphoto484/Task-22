# ==========================================
# Integrated Employment, Deal & Smoking Analysis Script
# Requirements: pandas, numpy, matplotlib, scipy, scikit-learn
# Input files: employment.xlsx, groupon.xlsx, smoker.xlsx
# Output files: mean_comparison.png, rank_correlation_heatmap.png,
#               groupon_cluster_scatter.png, employment_pca_scatter.png,
#               lorenz_curve.png
# ==========================================

import pandas as pd
import numpy as np
import matplotlib
matplotlib.use("Agg")
import matplotlib.pyplot as plt
from scipy.stats import spearmanr, mannwhitneyu
from sklearn.preprocessing import StandardScaler
from sklearn.cluster import KMeans
from sklearn.metrics import silhouette_score
from sklearn.decomposition import PCA

# ---------- Load Data ----------
emp_raw = pd.read_excel("employment.xlsx")
grp_raw = pd.read_excel("groupon.xlsx")
smk_raw = pd.read_excel("smoker.xlsx")

# ================================================================
# OPERATION 1 – Mean Comparison across datasets
# ================================================================
print("=" * 60)
print("OPERATION 1: Mean Comparison")
print("=" * 60)

# --- Employment: mean emp_change_ratio ---
op1_emp = emp_raw[["state", "total_emp_feb", "total_emp_nov"]].copy()
op1_emp = op1_emp.dropna(subset=["state", "total_emp_feb", "total_emp_nov"], how="any")
op1_emp["emp_change_ratio"] = (op1_emp["total_emp_nov"] - op1_emp["total_emp_feb"]) / op1_emp["total_emp_feb"]
mean_emp = op1_emp["emp_change_ratio"].mean()

# --- Groupon: mean log_revenue ---
op1_grp = grp_raw[["deal_id", "treatment", "price", "discount_pct", "featured",
                    "quantity_sold", "revenue"]].copy()
op1_grp = op1_grp.dropna(subset=["deal_id", "treatment", "price", "discount_pct",
                                  "featured", "quantity_sold", "revenue"], how="any")
op1_grp["log_revenue"] = np.log1p(op1_grp["revenue"])
mean_grp = op1_grp["log_revenue"].mean()

# --- Smoker: mean outcome ---
op1_smk = smk_raw[["smoker", "treatment", "outcome"]].copy()
op1_smk = op1_smk.dropna(subset=["smoker", "treatment", "outcome"], how="any")
mean_smk = op1_smk["outcome"].mean()

means = {"Employment\n(emp_change_ratio)": mean_emp,
         "Groupon\n(log_revenue)": mean_grp,
         "Smoker\n(outcome)": mean_smk}
largest_mean_label = max(means, key=means.get)
largest_mean_val = means[largest_mean_label]

print(f"  Mean emp_change_ratio : {mean_emp:.6f}")
print(f"  Mean log_revenue      : {mean_grp:.6f}")
print(f"  Mean outcome          : {mean_smk:.6f}")
print(f"  >> Largest mean value : {largest_mean_val:.6f} ({largest_mean_label.replace(chr(10), ' ')})")

# --- Bar plot ---
fig, ax = plt.subplots(figsize=(7, 5))
labels = list(means.keys())
values = list(means.values())
bars = ax.bar(labels, values, color=["#4C72B0", "#55A868", "#C44E52"])
ax.set_ylabel("Mean Value")
ax.set_title("Operation 1: Mean Comparison Across Datasets")
for b, v in zip(bars, values):
    ax.text(b.get_x() + b.get_width() / 2, v + 0.02 * max(values),
            f"{v:.4f}", ha="center", va="bottom", fontsize=9)
plt.tight_layout()
plt.savefig("mean_comparison.png", dpi=150)
plt.close()

# ================================================================
# OPERATION 2 – Spearman Rank Correlation
# ================================================================
print("\n" + "=" * 60)
print("OPERATION 2: Spearman Rank Correlation")
print("=" * 60)

# --- Employment: total_emp_feb vs total_emp_nov ---
op2_emp = emp_raw[["total_emp_feb", "total_emp_nov"]].dropna(how="any")
rho_emp, _ = spearmanr(op2_emp["total_emp_feb"], op2_emp["total_emp_nov"])

# --- Groupon: discount_pct vs log_revenue ---
op2_grp = grp_raw[["discount_pct", "revenue"]].dropna(how="any")
op2_grp["log_revenue"] = np.log1p(op2_grp["revenue"])
rho_grp, _ = spearmanr(op2_grp["discount_pct"], op2_grp["log_revenue"])

# --- Smoker: treatment vs outcome ---
op2_smk = smk_raw[["treatment", "outcome"]].dropna(how="any")
rho_smk, _ = spearmanr(op2_smk["treatment"], op2_smk["outcome"])

abs_corrs = {"Employment": abs(rho_emp),
             "Groupon": abs(rho_grp),
             "Smoker": abs(rho_smk)}
max_corr_label = max(abs_corrs, key=abs_corrs.get)
max_corr_val = abs_corrs[max_corr_label]

print(f"  |rho| Employment (feb vs nov)      : {abs(rho_emp):.6f}")
print(f"  |rho| Groupon (disc vs log_rev)     : {abs(rho_grp):.6f}")
print(f"  |rho| Smoker (treatment vs outcome)  : {abs(rho_smk):.6f}")
print(f"  >> Max absolute correlation          : {max_corr_val:.6f} ({max_corr_label})")

# --- Heatmap ---
fig, ax = plt.subplots(figsize=(6, 4))
corr_matrix = np.array([[abs(rho_emp)], [abs(rho_grp)], [abs(rho_smk)]])
dataset_labels = ["Employment", "Groupon", "Smoker"]
im = ax.imshow(corr_matrix, cmap="YlOrRd", aspect="auto", vmin=0, vmax=1)
ax.set_yticks(range(3))
ax.set_yticklabels(dataset_labels)
ax.set_xticks([0])
ax.set_xticklabels(["Spearman |rho|"])
for i in range(3):
    ax.text(0, i, f"{corr_matrix[i, 0]:.4f}", ha="center", va="center",
            color="black", fontsize=11)
ax.set_xlabel("Correlation Label")
ax.set_ylabel("Dataset Label")
ax.set_title("Operation 2: Rank Correlation Heatmap")
plt.colorbar(im, ax=ax)
plt.tight_layout()
plt.savefig("rank_correlation_heatmap.png", dpi=150)
plt.close()

# ================================================================
# OPERATION 3 – K-Means Clustering on Groupon
# ================================================================
print("\n" + "=" * 60)
print("OPERATION 3: K-Means Clustering (Groupon)")
print("=" * 60)

op3_grp = grp_raw[["price", "discount_pct"]].dropna(how="any").copy()
scaler3 = StandardScaler()
X3 = scaler3.fit_transform(op3_grp[["price", "discount_pct"]])

km = KMeans(n_clusters=3, random_state=42, n_init=10, max_iter=300)
labels3 = km.fit_predict(X3)
sil = silhouette_score(X3, labels3, metric="euclidean")

print(f"  Silhouette coefficient: {sil:.6f}")

# --- Scatter plot ---
fig, ax = plt.subplots(figsize=(7, 5))
scatter = ax.scatter(op3_grp["price"], op3_grp["discount_pct"],
                     c=labels3, cmap="viridis", alpha=0.7, edgecolors="k", linewidths=0.3)
ax.set_xlabel("Price")
ax.set_ylabel("Discount %")
ax.set_title("Operation 3: Groupon Cluster Structure (k=3)")
plt.colorbar(scatter, ax=ax, label="Cluster")
plt.tight_layout()
plt.savefig("groupon_cluster_scatter.png", dpi=150)
plt.close()

# ================================================================
# OPERATION 4 – PCA on Employment
# ================================================================
print("\n" + "=" * 60)
print("OPERATION 4: PCA (Employment)")
print("=" * 60)

op4_emp = emp_raw[["total_emp_feb", "total_emp_nov"]].dropna(how="any").copy()
op4_emp["emp_change_ratio"] = (op4_emp["total_emp_nov"] - op4_emp["total_emp_feb"]) / op4_emp["total_emp_feb"]

scaler4 = StandardScaler()
X4 = scaler4.fit_transform(op4_emp[["total_emp_feb", "total_emp_nov", "emp_change_ratio"]])

pca = PCA(n_components=2)
X4_pca = pca.fit_transform(X4)
var_pc1 = pca.explained_variance_ratio_[0]

print(f"  Variance explained by PC1: {var_pc1:.6f}")

# --- Scatter plot ---
fig, ax = plt.subplots(figsize=(7, 5))
ax.scatter(X4_pca[:, 0], X4_pca[:, 1], alpha=0.6, edgecolors="k", linewidths=0.3)
ax.set_xlabel(f"PC1 ({var_pc1 * 100:.1f}% variance)")
ax.set_ylabel(f"PC2 ({pca.explained_variance_ratio_[1] * 100:.1f}% variance)")
ax.set_title("Operation 4: Employment PCA Space")
plt.tight_layout()
plt.savefig("employment_pca_scatter.png", dpi=150)
plt.close()

# ================================================================
# OPERATION 5 – Gini Coefficient & Lorenz Curve (Groupon Revenue)
# ================================================================
print("\n" + "=" * 60)
print("OPERATION 5: Gini Coefficient (Groupon Revenue)")
print("=" * 60)

op5_grp = grp_raw[["revenue"]].dropna(how="any").copy()
rev = op5_grp["revenue"].values.astype(float)

# Shift if any negative values
if rev.min() < 0:
    rev = rev - rev.min()

rev_sorted = np.sort(rev)
n = len(rev_sorted)
cumulative = np.cumsum(rev_sorted)
total = cumulative[-1]
# Gini = 1 - 2*B where B = area under Lorenz curve
# Using the standard formula: G = (2 * sum(i * y_i) / (n * sum(y_i))) - (n + 1) / n
index = np.arange(1, n + 1)
gini = (2.0 * np.sum(index * rev_sorted)) / (n * np.sum(rev_sorted)) - (n + 1) / n

print(f"  Gini coefficient: {gini:.6f}")

# --- Lorenz curve ---
lorenz_y = np.concatenate(([0], cumulative / total))
lorenz_x = np.linspace(0, 1, n + 1)

fig, ax = plt.subplots(figsize=(7, 5))
ax.plot(lorenz_x, lorenz_y, color="#C44E52", linewidth=2, label=f"Lorenz (Gini={gini:.4f})")
ax.plot([0, 1], [0, 1], "k--", linewidth=1, label="Perfect Equality")
ax.fill_between(lorenz_x, lorenz_y, lorenz_x, alpha=0.15, color="#C44E52")
ax.set_xlabel("Cumulative Share of Deals")
ax.set_ylabel("Cumulative Share of Revenue")
ax.set_title("Operation 5: Lorenz Curve – Revenue Inequality")
ax.legend()
plt.tight_layout()
plt.savefig("lorenz_curve.png", dpi=150)
plt.close()

# ================================================================
# OPERATION 6 – Mann-Whitney U Test (Groupon: featured vs quantity_sold)
# ================================================================
print("\n" + "=" * 60)
print("OPERATION 6: Mann-Whitney U Test (Groupon)")
print("=" * 60)

op6_grp = grp_raw[["featured", "quantity_sold"]].dropna(how="any").copy()

# Check binary: exactly two unique non-missing values
unique_feat = op6_grp["featured"].dropna().unique()
if len(unique_feat) == 2:
    group_1 = op6_grp.loc[op6_grp["featured"] == 1, "quantity_sold"]
    group_0 = op6_grp.loc[op6_grp["featured"] == 0, "quantity_sold"]
    if len(group_1) >= 2 and len(group_0) >= 2:
        u_stat, p_val = mannwhitneyu(group_1, group_0, alternative="two-sided")
        print(f"  U statistic: {u_stat:.6f}")
    else:
        print("  Skipped: one group has fewer than 2 observations.")
else:
    print("  Skipped: 'featured' does not have exactly 2 unique values.")

# ================================================================
# OPERATION 7 – Employment Up Proportion
# ================================================================
print("\n" + "=" * 60)
print("OPERATION 7: Employment Up Proportion")
print("=" * 60)

op7_emp = emp_raw[["total_emp_feb", "total_emp_nov"]].dropna(how="any").copy()
op7_emp["emp_up"] = (op7_emp["total_emp_nov"] > op7_emp["total_emp_feb"]).astype(int)
prop_up = op7_emp["emp_up"].mean()

print(f"  Proportion emp_up=1: {prop_up:.6f}")

# ================================================================
# OPERATION 8 – Efficiency Score by Treatment (Groupon)
# ================================================================
print("\n" + "=" * 60)
print("OPERATION 8: Efficiency Score by Treatment (Groupon)")
print("=" * 60)

op8_grp = grp_raw[["treatment", "revenue", "quantity_sold"]].dropna(how="any").copy()
op8_grp["efficiency"] = op8_grp["revenue"] / op8_grp["quantity_sold"]

median_eff = op8_grp.groupby("treatment")["efficiency"].median()
print(f"  Median efficiency by treatment:\n{median_eff.to_string()}")

# Identify treatment with higher median; tie-break: treatment=0
if median_eff.nunique() == 1:
    best_treatment = 0
else:
    best_treatment = median_eff.idxmax()

print(f"  >> Treatment with higher median efficiency: {best_treatment}")

# ---------- Done ----------
print("\n" + "=" * 60)
print("All 8 operations completed. Plots saved.")
print("=" * 60)
