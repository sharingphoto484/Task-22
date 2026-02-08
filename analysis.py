# ==========================================
# Integrated Promotional Effectiveness & Behavioral Intervention Analysis Script
# Requirements: pandas, numpy, matplotlib, scipy, scikit-learn, statsmodels, openpyxl
# Input files: employment.xlsx, groupon.xlsx, smoker.xlsx
# Output files: quantile_scatter.png, poisson_boxplot.png,
#               polynomial_line.png, ridge_histogram.png,
#               employment_scatter.png
# ==========================================

import pandas as pd
import numpy as np
import matplotlib
matplotlib.use('Agg')
import matplotlib.pyplot as plt
from scipy import stats
from sklearn.model_selection import train_test_split
from sklearn.linear_model import Ridge
from sklearn.preprocessing import PolynomialFeatures
from sklearn.metrics import mean_squared_error, r2_score
import statsmodels.formula.api as smf
import statsmodels.api as sm

# ---------- Load Excel Files Robustly ----------

groupon = pd.read_excel("groupon.xlsx")
employment = pd.read_excel("employment.xlsx")
smoker = pd.read_excel("smoker.xlsx")

# =====================================================================
# 1. QUANTILE REGRESSION: median relationship quantity_sold ~ price
# =====================================================================

# ---------- Clean Data for Quantile Regression ----------
qr_data = groupon.dropna(subset=["quantity_sold", "price"]).copy()
qr_data = qr_data.sort_values("deal_id", ascending=True).reset_index(drop=True)

# ---------- Fit Quantile Regression at 50th Percentile ----------
qr_model = smf.quantreg("quantity_sold ~ price", data=qr_data).fit(q=0.5)

print("=" * 60)
print("1. QUANTILE REGRESSION (50th percentile)")
print("=" * 60)
print(f"   Price Coefficient: {qr_model.params['price']:.4f}")
print()

# =====================================================================
# 2. POISSON REGRESSION: quantity_sold ~ featured + discount_pct
# =====================================================================

# ---------- Clean Data for Poisson Regression ----------
pois_data = groupon.dropna(subset=["quantity_sold", "featured", "discount_pct"]).copy()
pois_data = pois_data.sort_values("deal_id", ascending=True).reset_index(drop=True)

# ---------- Fit Poisson Regression ----------
pois_model = smf.glm(
    "quantity_sold ~ featured + discount_pct",
    data=pois_data,
    family=sm.families.Poisson()
).fit()

print("=" * 60)
print("2. POISSON REGRESSION")
print("=" * 60)
print(f"   Featured Coefficient: {pois_model.params['featured']:.4f}")
print()

# =====================================================================
# 3. POLYNOMIAL REGRESSION: revenue ~ price + price^2
# =====================================================================

# ---------- Clean Data for Polynomial Regression ----------
poly_data = groupon.dropna(subset=["revenue", "price"]).copy()
poly_data = poly_data.sort_values("deal_id", ascending=True).reset_index(drop=True)

# ---------- Create Polynomial Features (degree 2) ----------
poly_features = PolynomialFeatures(degree=2, include_bias=False)
X_poly = poly_features.fit_transform(poly_data[["price"]])
y_poly = poly_data["revenue"]

# ---------- Fit OLS with Polynomial Features ----------
X_poly_with_const = sm.add_constant(X_poly)
poly_model = sm.OLS(y_poly, X_poly_with_const).fit()
r_squared = poly_model.rsquared

print("=" * 60)
print("3. POLYNOMIAL REGRESSION (degree 2)")
print("=" * 60)
print(f"   R-squared: {r_squared:.4f}")
print()

# =====================================================================
# 4. RIDGE REGRESSION: revenue ~ price + discount_pct + fb_likes
# =====================================================================

# ---------- Clean Data for Ridge Regression ----------
ridge_data = groupon.dropna(subset=["revenue", "price", "discount_pct", "fb_likes"]).copy()
ridge_data = ridge_data.sort_values("deal_id", ascending=True).reset_index(drop=True)

# ---------- Train-Test Split ----------
X_ridge = ridge_data[["price", "discount_pct", "fb_likes"]]
y_ridge = ridge_data["revenue"]

X_train, X_test, y_train, y_test = train_test_split(
    X_ridge, y_ridge, test_size=0.3, random_state=42, shuffle=True
)

# ---------- Fit Ridge Regression ----------
ridge_model = Ridge(alpha=1.0, random_state=42)
ridge_model.fit(X_train, y_train)

y_pred_test = ridge_model.predict(X_test)
test_mse = mean_squared_error(y_test, y_pred_test)

print("=" * 60)
print("4. RIDGE REGRESSION (alpha=1.0)")
print("=" * 60)
print(f"   Testing Set MSE: {test_mse:.4f}")
print()

# =====================================================================
# 5. SPEARMAN RANK CORRELATION: total_emp_feb vs total_emp_nov
# =====================================================================

# ---------- Clean Data for Spearman Correlation ----------
emp_data = employment.dropna(subset=["total_emp_feb", "total_emp_nov"]).copy()
emp_data = emp_data.sort_values(
    ["state", "total_emp_feb"], ascending=[True, True]
).reset_index(drop=True)

# ---------- Compute Spearman Rank Correlation ----------
spearman_rho, spearman_p = stats.spearmanr(emp_data["total_emp_feb"], emp_data["total_emp_nov"])

print("=" * 60)
print("5. SPEARMAN RANK CORRELATION")
print("=" * 60)
print(f"   Spearman rho: {spearman_rho:.4f}")
print()

# =====================================================================
# 6. MANN-WHITNEY U TEST: outcome by treatment group
# =====================================================================

# ---------- Clean Data for Mann-Whitney U Test ----------
mw_data = smoker.dropna(subset=["outcome", "treatment"]).copy()

# ---------- Extract Control and Intervention Groups ----------
control = mw_data.loc[mw_data["treatment"] == 0, "outcome"].values
intervention = mw_data.loc[mw_data["treatment"] == 1, "outcome"].values

# ---------- Compute Mann-Whitney U Statistic ----------
u_stat, u_pvalue = stats.mannwhitneyu(control, intervention, alternative="two-sided")

print("=" * 60)
print("6. MANN-WHITNEY U TEST")
print("=" * 60)
print(f"   U Statistic: {u_stat:.4f}")
print()

# =====================================================================
# 7. CHI-SQUARE INDEPENDENCE TEST: smoker vs treatment
# =====================================================================

# ---------- Clean Data for Chi-Square Test ----------
chi_data = smoker.dropna(subset=["smoker", "treatment"]).copy()

# ---------- Construct Contingency Table ----------
contingency_table = pd.crosstab(chi_data["smoker"], chi_data["treatment"])

# ---------- Compute Chi-Square Test ----------
chi2_stat, chi2_p, chi2_dof, chi2_expected = stats.chi2_contingency(contingency_table)

print("=" * 60)
print("7. CHI-SQUARE INDEPENDENCE TEST")
print("=" * 60)
print(f"   Chi-square Statistic: {chi2_stat:.4f}")
print()

# =====================================================================
# 8. BOOTSTRAP RESAMPLING: mean revenue confidence interval
# =====================================================================

# ---------- Clean Data for Bootstrap ----------
boot_data = groupon.dropna(subset=["revenue"]).copy()
boot_data = boot_data.sort_values("deal_id", ascending=True).reset_index(drop=True)

# ---------- Generate Bootstrap Samples ----------
np.random.seed(42)
n_bootstrap = 1000
n_obs = len(boot_data)
revenue_values = boot_data["revenue"].values

bootstrap_means = np.array([
    np.mean(np.random.choice(revenue_values, size=n_obs, replace=True))
    for _ in range(n_bootstrap)
])

# ---------- Compute 95% Confidence Interval ----------
ci_lower = np.percentile(bootstrap_means, 2.5)
ci_upper = np.percentile(bootstrap_means, 97.5)
ci_width = ci_upper - ci_lower

print("=" * 60)
print("8. BOOTSTRAP RESAMPLING (1000 samples)")
print("=" * 60)
print(f"   95% CI Width: {ci_width:.4f}")
print()

# =====================================================================
# VISUALIZATIONS
# =====================================================================

# ---------- Plot 1: Scatter Plot for Quantile Regression ----------
fig, ax = plt.subplots(figsize=(8, 6))
ax.scatter(qr_data["price"], qr_data["quantity_sold"], alpha=0.5, edgecolors="k", linewidth=0.3)
price_range = np.linspace(qr_data["price"].min(), qr_data["price"].max(), 200)
fitted_line = qr_model.params["Intercept"] + qr_model.params["price"] * price_range
ax.plot(price_range, fitted_line, color="red", linewidth=2, label="Median (Q50) fit")
ax.set_xlabel("Price")
ax.set_ylabel("Quantity Sold")
ax.set_title("Quantile Regression: Quantity Sold vs Price")
ax.legend()
plt.tight_layout()
plt.savefig("quantile_scatter.png", dpi=150)
plt.close()

# ---------- Plot 2: Box Plot for Poisson Model Residuals ----------
pois_pred = pois_model.predict(pois_data)
pearson_resid = (pois_data["quantity_sold"].values - pois_pred.values) / np.sqrt(pois_pred.values)
pois_data = pois_data.copy()
pois_data["pearson_residual"] = pearson_resid

fig, ax = plt.subplots(figsize=(8, 6))
groups = [
    pois_data.loc[pois_data["featured"] == 0, "pearson_residual"].values,
    pois_data.loc[pois_data["featured"] == 1, "pearson_residual"].values,
]
ax.boxplot(groups, tick_labels=["Not Featured (0)", "Featured (1)"])
ax.set_xlabel("Featured")
ax.set_ylabel("Pearson Residuals")
ax.set_title("Poisson Model: Residual Distribution by Featured Status")
plt.tight_layout()
plt.savefig("poisson_boxplot.png", dpi=150)
plt.close()

# ---------- Plot 3: Line Plot for Polynomial Regression Curve ----------
price_sorted_idx = poly_data["price"].argsort()
price_sorted = poly_data["price"].values[price_sorted_idx]
fitted_revenue = poly_model.fittedvalues.values[price_sorted_idx]

fig, ax = plt.subplots(figsize=(8, 6))
ax.scatter(poly_data["price"], poly_data["revenue"], alpha=0.3, label="Observed", edgecolors="k", linewidth=0.3)
ax.plot(price_sorted, fitted_revenue, color="red", linewidth=2, label="Polynomial fit (degree 2)")
ax.set_xlabel("Price")
ax.set_ylabel("Revenue")
ax.set_title("Polynomial Regression: Revenue vs Price")
ax.legend()
plt.tight_layout()
plt.savefig("polynomial_line.png", dpi=150)
plt.close()

# ---------- Plot 4: Histogram for Ridge Regression Prediction Errors ----------
ridge_residuals = np.abs(y_test.values - y_pred_test)

fig, ax = plt.subplots(figsize=(8, 6))
ax.hist(ridge_residuals, bins=30, edgecolor="black", alpha=0.7)
ax.set_xlabel("Absolute Residuals")
ax.set_ylabel("Frequency")
ax.set_title("Ridge Regression: Prediction Error Distribution")
plt.tight_layout()
plt.savefig("ridge_histogram.png", dpi=150)
plt.close()

# ---------- Plot 5: Scatter Plot for Employment Rank Correlation ----------
fig, ax = plt.subplots(figsize=(8, 6))
ax.scatter(emp_data["total_emp_feb"], emp_data["total_emp_nov"], alpha=0.5, edgecolors="k", linewidth=0.3)
ax.set_xlabel("Total Employment (February)")
ax.set_ylabel("Total Employment (November)")
ax.set_title(f"Employment Rank Correlation (Spearman rho = {spearman_rho:.4f})")
plt.tight_layout()
plt.savefig("employment_scatter.png", dpi=150)
plt.close()

print("=" * 60)
print("All plots saved successfully.")
print("=" * 60)
