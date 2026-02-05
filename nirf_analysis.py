# ==========================================
# Integrated NIRF Rankings Analysis Script
# Requirements: pandas, numpy, matplotlib, scipy, scikit-learn, statsmodels
# Input files: NIRF Ranking 2023.xlsx, NIRF Ranking 2024.xlsx,
#              NIRF Ranking 2025.xlsx
# Output files: quantile_regression_residuals.png, cca_biplot.png,
#               nmf_heatmap.png, pls_component_contribution.png,
#               cdf_score_comparison.png, bootstrap_histogram.png
# ==========================================

import warnings
warnings.filterwarnings("ignore")

import pandas as pd
import numpy as np
import matplotlib.pyplot as plt
from sklearn.preprocessing import StandardScaler, MinMaxScaler
from sklearn.cross_decomposition import CCA, PLSRegression
from sklearn.decomposition import NMF
from sklearn.model_selection import train_test_split
from scipy.stats import wasserstein_distance, pearsonr
import statsmodels.api as sm
from statsmodels.regression.quantile_regression import QuantReg

# ---------- Configuration ----------
DATA_DIR = "/home/user/Task-22"
np.random.seed(42)

# ---------- Load Excel Files ----------
df_2023 = pd.read_excel(f"{DATA_DIR}/NIRF Ranking 2023.xlsx")
df_2024 = pd.read_excel(f"{DATA_DIR}/NIRF Ranking 2024.xlsx")
df_2025 = pd.read_excel(f"{DATA_DIR}/NIRF Ranking 2025.xlsx")

# ---------- Strip Unnamed Columns ----------
df_2023 = df_2023.loc[:, ~df_2023.columns.str.contains("Unnamed")]
df_2024 = df_2024.loc[:, ~df_2024.columns.str.contains("Unnamed")]
df_2025 = df_2025.loc[:, ~df_2025.columns.str.contains("Unnamed")]


# ============================================================
# Protocol 1: Quantile Regression (75th Percentile)
# Source: NIRF_Ranking_2025  |  Target: Score
# ============================================================

# ---------- Isolate and Drop Nulls on Referenced Variables ----------
qr_vars = ["Score", "TLR", "RPC", "GO", "OI"]
df_qr = df_2025[qr_vars].dropna(subset=qr_vars).copy()

# ---------- Build Exogenous Matrix with Intercept ----------
X_qr = sm.add_constant(df_qr[["TLR", "RPC", "GO", "OI"]])
y_qr = df_qr["Score"]

# ---------- Fit QuantReg at q = 0.75 ----------
qr_model = QuantReg(y_qr, X_qr)
qr_result = qr_model.fit(q=0.75)

quantile_regression_tlr_coefficient = round(float(qr_result.params["TLR"]), 4)
print(f"quantile_regression_tlr_coefficient : {quantile_regression_tlr_coefficient}")

# ---------- Connected Scatter — Fitted vs Residuals ----------
fitted_vals = qr_result.fittedvalues.values
residuals_qr = y_qr.values - fitted_vals
sort_idx = np.argsort(fitted_vals)

fig, ax = plt.subplots(figsize=(10, 6))
ax.plot(fitted_vals[sort_idx], residuals_qr[sort_idx],
        color="steelblue", linewidth=1.2, alpha=0.5)
ax.scatter(fitted_vals, residuals_qr, color="steelblue",
           edgecolors="navy", s=50, zorder=5, alpha=0.8)
ax.axhline(0, color="crimson", linestyle="--", linewidth=1.2)
ax.set_xlabel("Fitted Values", fontsize=12)
ax.set_ylabel("Residuals", fontsize=12)
ax.set_title("Quantile Regression Diagnostic: Fitted Values vs Residuals", fontsize=13)
ax.grid(True, alpha=0.3)
plt.tight_layout()
plt.savefig(f"{DATA_DIR}/quantile_regression_residuals.png", dpi=150, bbox_inches="tight")
plt.close()


# ============================================================
# Protocol 2: Canonical Correlation Analysis
# Source: NIRF_Ranking_2024  |  X = [TLR, GO]  Y = [RPC, PERCEPTION]
# ============================================================

# ---------- Isolate and Drop Nulls on Referenced Variables ----------
cca_vars = ["TLR", "GO", "RPC", "PERCEPTION"]
df_cca = df_2024[cca_vars].dropna(subset=cca_vars).copy()

X_cca_raw = df_cca[["TLR", "GO"]].values
Y_cca_raw = df_cca[["RPC", "PERCEPTION"]].values

# ---------- Independent Standardisation ----------
X_cca_std = StandardScaler().fit_transform(X_cca_raw)
Y_cca_std = StandardScaler().fit_transform(Y_cca_raw)

# ---------- Fit CCA (n_components = 2) ----------
cca_model = CCA(n_components=2)
cca_model.fit(X_cca_std, Y_cca_std)
X_cca_t, Y_cca_t = cca_model.transform(X_cca_std, Y_cca_std)

# ---------- First Canonical Correlation ----------
first_canonical_correlation = round(float(pearsonr(X_cca_t[:, 0], Y_cca_t[:, 0])[0]), 4)
print(f"first_canonical_correlation          : {first_canonical_correlation}")

# ---------- Biplot — 1st Canonical Variates ----------
fig, ax = plt.subplots(figsize=(10, 6))
sc = ax.scatter(X_cca_t[:, 0], Y_cca_t[:, 0],
                c=np.arange(len(X_cca_t)), cmap="viridis",
                edgecolors="black", s=55, alpha=0.85)
plt.colorbar(sc, ax=ax, label="Institution Index")
ax.set_xlabel("1st Canonical Variate — Academic Delivery (TLR, GO)", fontsize=12)
ax.set_ylabel("1st Canonical Variate — Research Reputation (RPC, PERCEPTION)", fontsize=12)
ax.set_title("CCA Biplot: Academic Delivery vs Research Reputation", fontsize=13)
ax.grid(True, alpha=0.3)
plt.tight_layout()
plt.savefig(f"{DATA_DIR}/cca_biplot.png", dpi=150, bbox_inches="tight")
plt.close()


# ============================================================
# Protocol 3: Non-negative Matrix Factorisation
# Source: NIRF_Ranking_2025  |  Features: TLR, RPC, GO, OI, PERCEPTION
# ============================================================

# ---------- Isolate and Drop Nulls on Referenced Variables ----------
nmf_features = ["TLR", "RPC", "GO", "OI", "PERCEPTION"]
df_nmf = df_2025[nmf_features].dropna(subset=nmf_features).copy()

# ---------- MinMax Scale to [0, 1] ----------
V_scaled = MinMaxScaler().fit_transform(df_nmf[nmf_features].values)

# ---------- NMF Decomposition (n_components = 2) ----------
nmf_model = NMF(n_components=2, init="nndsvd", max_iter=500, random_state=42)
W = nmf_model.fit_transform(V_scaled)
H = nmf_model.components_          # shape (2, 5)

# ---------- Relative Reconstruction Error ----------
V_recon = W @ H
nmf_relative_reconstruction_error = round(
    float(np.linalg.norm(V_scaled - V_recon, "fro") /
          np.linalg.norm(V_scaled, "fro")), 4)
print(f"nmf_relative_reconstruction_error    : {nmf_relative_reconstruction_error}")

# ---------- Heatmap — Latent Factor Loadings (H transposed) ----------
H_T = H.T          # shape (n_features=5, n_components=2)
fig, ax = plt.subplots(figsize=(8, 6))
im = ax.imshow(H_T, aspect="auto", cmap="YlOrRd")
plt.colorbar(im, ax=ax, label="Loading Value")

ax.set_xticks(range(2))
ax.set_xticklabels(["Latent Factor 0", "Latent Factor 1"], fontsize=11)
ax.set_yticks(range(len(nmf_features)))
ax.set_yticklabels(nmf_features, fontsize=11)
ax.set_xlabel("Latent Factor Index", fontsize=12)
ax.set_ylabel("Original Feature", fontsize=12)
ax.set_title("NMF Latent Factor Loadings", fontsize=13)

# ---------- Cell Annotations ----------
mid_val = (H_T.min() + H_T.max()) / 2
for i in range(H_T.shape[0]):
    for j in range(H_T.shape[1]):
        txt_color = "white" if H_T[i, j] > mid_val else "black"
        ax.text(j, i, f"{H_T[i, j]:.4f}", ha="center", va="center",
                fontsize=10, color=txt_color, fontweight="bold")
plt.tight_layout()
plt.savefig(f"{DATA_DIR}/nmf_heatmap.png", dpi=150, bbox_inches="tight")
plt.close()


# ============================================================
# Protocol 4: Partial Least Squares Regression
# Source: Combined 2023 + 2024  |  Predictors → Score
# ============================================================

# ---------- Isolate and Drop Nulls Independently per Year ----------
pls_feats = ["TLR", "RPC", "GO", "OI", "PERCEPTION"]
pls_all   = pls_feats + ["Score"]

df_2023_pls = df_2023[pls_all].dropna(subset=pls_all).copy()
df_2024_pls = df_2024[pls_all].dropna(subset=pls_all).copy()

# ---------- Vertical Concatenation (2023 first) ----------
df_pls = pd.concat([df_2023_pls, df_2024_pls], ignore_index=True)
X_pls = df_pls[pls_feats].values
y_pls = df_pls["Score"].values

# ---------- Train / Validation Split (70 / 30) ----------
X_train, X_val, y_train, y_val = train_test_split(
    X_pls, y_pls, test_size=0.3, random_state=42)

# ---------- Standardise on Training Set Only ----------
scaler_pls = StandardScaler()
X_train_std = scaler_pls.fit_transform(X_train)
X_val_std   = scaler_pls.transform(X_val)

# ---------- PLS Model (n_components = 3, scale = False) ----------
pls_model = PLSRegression(n_components=3, scale=False)
pls_model.fit(X_train_std, y_train)

pls_validation_r_squared = round(float(pls_model.score(X_val_std, y_val)), 4)
print(f"pls_validation_r_squared             : {pls_validation_r_squared}")

# ---------- Incremental Explained-Variance Ratio (training) ----------
r2_incremental = []
r2_prev = 0.0
for n in range(1, 4):
    pls_tmp = PLSRegression(n_components=n, scale=False)
    pls_tmp.fit(X_train_std, y_train)
    r2_curr = pls_tmp.score(X_train_std, y_train)
    r2_incremental.append(r2_curr - r2_prev)
    r2_prev = r2_curr

# ---------- Bar Plot — Component Contribution ----------
colors_pls = ["#4c72b0", "#55a868", "#c44e52"]
fig, ax = plt.subplots(figsize=(8, 5))
bars = ax.bar(["Component 1", "Component 2", "Component 3"],
              r2_incremental, color=colors_pls, edgecolor="black", width=0.45)
for bar, val in zip(bars, r2_incremental):
    ax.text(bar.get_x() + bar.get_width() / 2,
            bar.get_height() + 0.004,
            f"{val:.4f}", ha="center", va="bottom", fontsize=11)
ax.set_xlabel("PLS Component Index", fontsize=12)
ax.set_ylabel("Explained Variance Ratio", fontsize=12)
ax.set_title("PLS Component Contribution Analysis", fontsize=13)
ax.grid(True, axis="y", alpha=0.3)
plt.tight_layout()
plt.savefig(f"{DATA_DIR}/pls_component_contribution.png", dpi=150, bbox_inches="tight")
plt.close()


# ============================================================
# Protocol 5: Wasserstein Distance (2023 vs 2025)
# ============================================================

# ---------- Extract Scores, Drop Nulls Independently ----------
scores_2023 = df_2023["Score"].dropna().values
scores_2025 = df_2025["Score"].dropna().values

# ---------- Compute Wasserstein Distance ----------
wasserstein_distance_2023_2025 = round(
    float(wasserstein_distance(scores_2023, scores_2025)), 4)
print(f"wasserstein_distance_2023_2025       : {wasserstein_distance_2023_2025}")

# ---------- CDF Plot — 2023 vs 2025 ----------
fig, ax = plt.subplots(figsize=(10, 6))
plot_spec = [
    (scores_2023, "2023", "#2166ac", "-"),
    (scores_2025, "2025", "#d6604d", "--"),
]
for scores, year, color, ls in plot_spec:
    s = np.sort(scores)
    cdf = np.arange(1, len(s) + 1) / len(s)
    ax.plot(s, cdf, color=color, linestyle=ls, linewidth=2.5, label=f"NIRF {year}")

ax.set_xlabel("Score", fontsize=12)
ax.set_ylabel("Cumulative Probability", fontsize=12)
ax.set_title("CDF: Score Distribution Comparison (2023 vs 2025)", fontsize=13)
ax.legend(fontsize=11)
ax.grid(True, alpha=0.3)
plt.tight_layout()
plt.savefig(f"{DATA_DIR}/cdf_score_comparison.png", dpi=150, bbox_inches="tight")
plt.close()


# ============================================================
# Protocol 6: Bootstrap Simulation — Mean Score Improvement
# Matched institutions across all three years
# ============================================================

# ---------- Identify Common Institute IDs ----------
common_ids = (set(df_2023["Institute ID"]) &
              set(df_2024["Institute ID"]) &
              set(df_2025["Institute ID"]))

df23_m = df_2023[df_2023["Institute ID"].isin(common_ids)].set_index("Institute ID")
df25_m = df_2025[df_2025["Institute ID"].isin(common_ids)].set_index("Institute ID")

aligned = df23_m.index.intersection(df25_m.index)
score_diff = df25_m.loc[aligned, "Score"].values - df23_m.loc[aligned, "Score"].values

# ---------- Bootstrap Loop (10 000 iterations, seed = 42) ----------
np.random.seed(42)
n_boot = 10000
boot_means = np.zeros(n_boot)
for i in range(n_boot):
    sample = np.random.choice(score_diff, size=len(score_diff), replace=True)
    boot_means[i] = np.mean(sample)

bootstrap_ci_lower_bound = round(float(np.percentile(boot_means, 2.5)), 4)
bootstrap_ci_upper_bound = round(float(np.percentile(boot_means, 97.5)), 4)
print(f"bootstrap_ci_lower_bound             : {bootstrap_ci_lower_bound}")
print(f"bootstrap_ci_upper_bound             : {bootstrap_ci_upper_bound}")

# ---------- Histogram — Bootstrap Sampling Distribution ----------
fig, ax = plt.subplots(figsize=(10, 6))
ax.hist(boot_means, bins=60, color="mediumpurple", edgecolor="black", alpha=0.85)
ax.axvline(bootstrap_ci_lower_bound, color="crimson", linestyle="--", linewidth=2,
           label=f"2.5th pctile : {bootstrap_ci_lower_bound}")
ax.axvline(bootstrap_ci_upper_bound, color="crimson", linestyle="--", linewidth=2,
           label=f"97.5th pctile : {bootstrap_ci_upper_bound}")
ax.axvline(np.mean(boot_means), color="darkgreen", linestyle="-", linewidth=2,
           label=f"Mean : {np.mean(boot_means):.4f}")
ax.set_xlabel("Bootstrap Sample Means", fontsize=12)
ax.set_ylabel("Frequency Count", fontsize=12)
ax.set_title("Bootstrap Distribution: Mean Score Improvement (2025 − 2023)", fontsize=13)
ax.legend(fontsize=10)
ax.grid(True, alpha=0.3)
plt.tight_layout()
plt.savefig(f"{DATA_DIR}/bootstrap_histogram.png", dpi=150, bbox_inches="tight")
plt.close()


# ============================================================
# Key Output
# ============================================================
print("\n" + "=" * 58)
print(" KEY METRICS")
print("=" * 58)
print(f"  quantile_regression_tlr_coefficient  : {quantile_regression_tlr_coefficient}")
print(f"  first_canonical_correlation          : {first_canonical_correlation}")
print(f"  nmf_relative_reconstruction_error    : {nmf_relative_reconstruction_error}")
print(f"  pls_validation_r_squared             : {pls_validation_r_squared}")
print(f"  wasserstein_distance_2023_2025       : {wasserstein_distance_2023_2025}")
print(f"  bootstrap_ci_lower_bound             : {bootstrap_ci_lower_bound}")
print(f"  bootstrap_ci_upper_bound             : {bootstrap_ci_upper_bound}")
print("=" * 58)
