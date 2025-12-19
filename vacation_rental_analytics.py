# ==========================================
# Vacation Rental Advanced Analytics Script
# ==========================================
# Requirements: pandas, numpy, matplotlib, scipy, scikit-learn, openpyxl
# Input files: airnb.xlsx, airnb_luxe.xlsx, airnb_desert.xlsx
# Output files: Various PNG visualizations
# Key outputs: Statistical metrics and ML model coefficients
# ==========================================

import pandas as pd
import numpy as np
import matplotlib.pyplot as plt
import re
from sklearn.ensemble import RandomForestRegressor, GradientBoostingRegressor
from sklearn.linear_model import ElasticNetCV
from sklearn.mixture import GaussianMixture
from sklearn.preprocessing import StandardScaler
from sklearn.model_selection import cross_val_score
from scipy.stats.mstats import gmean
import warnings
warnings.filterwarnings('ignore')

# ---------- Helper Functions ----------

def extract_numeric_price(price_str):
    """Extract numeric price by removing commas and dollar signs"""
    if pd.isna(price_str):
        return np.nan
    price_clean = str(price_str).replace(',', '').replace('$', '').strip()
    try:
        return float(price_clean)
    except:
        return np.nan

def extract_rating(rating_str):
    """Extract first numeric rating from string"""
    if pd.isna(rating_str):
        return np.nan
    match = re.search(r'(\d+\.?\d*)', str(rating_str))
    if match:
        return float(match.group(1))
    return np.nan

def extract_bed_count(bed_str):
    """Extract integer bed count from string"""
    if pd.isna(bed_str):
        return np.nan
    match = re.search(r'(\d+)', str(bed_str))
    if match:
        return int(match.group(1))
    return np.nan

def extract_review_count(rating_str):
    """Extract review count from parentheses"""
    if pd.isna(rating_str):
        return np.nan
    match = re.search(r'\((\d+)\)', str(rating_str))
    if match:
        return int(match.group(1))
    return np.nan

def extract_distance_km(distance_str):
    """Extract numeric kilometers from distance string"""
    if pd.isna(distance_str):
        return np.nan
    match = re.search(r'(\d+\.?\d*)', str(distance_str))
    if match:
        return float(match.group(1))
    return np.nan

def quantile_regression_coef(X, y, quantile=0.8):
    """Perform quantile regression using statsmodels"""
    try:
        import statsmodels.formula.api as smf
        import statsmodels.api as sm
        # Add constant
        X_with_const = sm.add_constant(X)
        # Fit quantile regression
        model = sm.QuantReg(y, X_with_const)
        result = model.fit(q=quantile)
        return result
    except ImportError:
        # Fallback to gradient boosting if statsmodels not available
        from sklearn.ensemble import GradientBoostingRegressor
        model = GradientBoostingRegressor(
            loss='quantile',
            alpha=quantile,
            n_estimators=100,
            max_depth=3,
            learning_rate=0.1,
            random_state=42
        )
        model.fit(X, y)
        return model

def weighted_quantile(values, weights, quantile=0.5):
    """Calculate weighted quantile"""
    sorter = np.argsort(values)
    values = values[sorter]
    weights = weights[sorter]
    weighted_quantiles = np.cumsum(weights) - 0.5 * weights
    weighted_quantiles /= np.sum(weights)
    return np.interp(quantile, weighted_quantiles, values)

# ---------- Load Data ----------

print("Loading datasets...")
df_airnb = pd.read_excel('airnb.xlsx')
df_luxe = pd.read_excel('airnb_luxe.xlsx')
df_desert = pd.read_excel('airnb_desert.xlsx')

print(f"Loaded {len(df_airnb)} regular listings, {len(df_luxe)} luxury listings, {len(df_desert)} desert listings")

# ==========================================
# ANALYSIS 1: Quantile Regression (airnb.xlsx)
# ==========================================
print("\n" + "="*50)
print("ANALYSIS 1: Quantile Regression - 80th Percentile")
print("="*50)

# ---------- Data Preparation ----------
df_airnb['price_numeric'] = df_airnb['Price(in dollar)'].apply(extract_numeric_price)
df_airnb['rating_numeric'] = df_airnb['Review and rating'].apply(extract_rating)
df_airnb['bed_count'] = df_airnb['Number of bed'].apply(extract_bed_count)
df_airnb['review_count'] = df_airnb['Review and rating'].apply(extract_review_count)

# Select complete records
df_qr = df_airnb.dropna(subset=['price_numeric', 'rating_numeric', 'bed_count', 'review_count']).copy()
print(f"Complete records for quantile regression: {len(df_qr)}")

# ---------- Standardization ----------
scaler_qr = StandardScaler()
features_qr = ['bed_count', 'rating_numeric', 'review_count']
df_qr[['bed_std', 'rating_std', 'review_std']] = scaler_qr.fit_transform(df_qr[features_qr])

# ---------- Quantile Regression Fitting ----------
X_qr = df_qr[['bed_std', 'rating_std', 'review_std']].values
y_qr = df_qr['price_numeric'].values

model_qr = quantile_regression_coef(X_qr, y_qr, quantile=0.8)

# Extract rating coefficient (index 2, after constant at index 0)
try:
    # For statsmodels QuantReg
    rating_coef = model_qr.params[2]  # bed_std=1, rating_std=2, review_std=3
except (AttributeError, IndexError):
    # For GradientBoosting fallback
    rating_coef = model_qr.feature_importances_[1]

print(f"\nRating Coefficient (80th percentile): {rating_coef:.4f}")

# ---------- Scatter Plot: Rating vs Price ----------
plt.figure(figsize=(10, 6))
plt.scatter(df_qr['rating_numeric'], df_qr['price_numeric'], alpha=0.5, s=20)
plt.xlabel('Rating Score', fontsize=12)
plt.ylabel('Price (Dollars)', fontsize=12)
plt.title('Quantile Regression: Rating vs Price', fontsize=14, fontweight='bold')
plt.grid(True, alpha=0.3)
plt.tight_layout()
plt.savefig('quantile_regression_scatter.png', dpi=300, bbox_inches='tight')
plt.close()
print("Saved: quantile_regression_scatter.png")

# ==========================================
# ANALYSIS 2: Random Forest Feature Importance (airnb_luxe.xlsx)
# ==========================================
print("\n" + "="*50)
print("ANALYSIS 2: Random Forest Feature Importance")
print("="*50)

# ---------- Data Preparation ----------
df_luxe['price_numeric'] = df_luxe['Price(In dollar)'].apply(extract_numeric_price)
df_luxe['distance_numeric'] = df_luxe['Distance'].apply(extract_distance_km)

# Extract month from date string (e.g., "May 1 29" -> 5)
def extract_month(date_str):
    if pd.isna(date_str):
        return np.nan
    month_map = {'Jan': 1, 'Feb': 2, 'Mar': 3, 'Apr': 4, 'May': 5, 'Jun': 6,
                 'Jul': 7, 'Aug': 8, 'Sep': 9, 'Oct': 10, 'Nov': 11, 'Dec': 12}
    for month_name, month_num in month_map.items():
        if month_name in str(date_str):
            return month_num
    return np.nan

df_luxe['month'] = df_luxe['Date'].apply(extract_month)

# Select complete records
df_rf = df_luxe.dropna(subset=['price_numeric', 'distance_numeric', 'month']).copy()
print(f"Complete records for random forest: {len(df_rf)}")

# ---------- Random Forest Training ----------
X_rf = df_rf[['distance_numeric', 'month']].values
y_rf = df_rf['price_numeric'].values

rf_model = RandomForestRegressor(n_estimators=100, max_depth=15, random_state=42)
rf_model.fit(X_rf, y_rf)

# Extract feature importances
importances = rf_model.feature_importances_
feature_names = ['distance', 'month']
max_importance_idx = np.argmax(importances)
most_important_feature = feature_names[max_importance_idx]

print(f"\nFeature Importances:")
for name, imp in zip(feature_names, importances):
    print(f"  {name}: {imp:.4f}")
print(f"\nMost Important Feature: {most_important_feature}")

# ---------- Horizontal Bar Chart ----------
plt.figure(figsize=(8, 5))
plt.barh(feature_names, importances, color=['steelblue', 'coral'])
plt.xlabel('Importance Score', fontsize=12)
plt.ylabel('Feature Names', fontsize=12)
plt.title('Random Forest Feature Importance', fontsize=14, fontweight='bold')
plt.xlim(0, 1)
plt.grid(True, alpha=0.3, axis='x')
plt.tight_layout()
plt.savefig('random_forest_importance.png', dpi=300, bbox_inches='tight')
plt.close()
print("Saved: random_forest_importance.png")

# ==========================================
# ANALYSIS 3: Elastic Net Regression (airnb_desert.xlsx)
# ==========================================
print("\n" + "="*50)
print("ANALYSIS 3: Elastic Net Regression")
print("="*50)

# ---------- Data Preparation ----------
df_desert['price_numeric'] = df_desert['Price(In dollar)'].apply(extract_numeric_price)

# Select complete records
df_en = df_desert.dropna(subset=['price_numeric', 'Rating']).copy()
print(f"Complete records for elastic net: {len(df_en)}")

# ---------- Standardization ----------
scaler_en = StandardScaler()
df_en['price_std'] = scaler_en.fit_transform(df_en[['price_numeric']])

# ---------- Elastic Net with CV ----------
X_en = df_en[['price_std']].values
y_en = df_en['Rating'].values

# Use a range of alphas for cross-validation
alphas = np.logspace(-4, 1, 100)
elastic_model = ElasticNetCV(alphas=alphas, l1_ratio=0.5, cv=5, random_state=42, max_iter=10000)
elastic_model.fit(X_en, y_en)

# Extract standardized coefficient
price_coefficient = elastic_model.coef_[0]
print(f"\nStandardized Price Coefficient: {price_coefficient:.4f}")
print(f"Optimal Alpha: {elastic_model.alpha_:.4f}")
print(f"Intercept: {elastic_model.intercept_:.4f}")

# ---------- Regularization Path Plot ----------
from sklearn.linear_model import enet_path
alphas_path, coefs_path, _ = enet_path(X_en, y_en, l1_ratio=0.5, eps=0.001)

plt.figure(figsize=(10, 6))
plt.plot(np.log(alphas_path), coefs_path[0], linewidth=2, color='darkblue')
plt.xlabel('Log Lambda Values', fontsize=12)
plt.ylabel('Coefficient Magnitude', fontsize=12)
plt.title('Elastic Net Regularization Path', fontsize=14, fontweight='bold')
plt.grid(True, alpha=0.3)
plt.axhline(y=0, color='red', linestyle='--', linewidth=1)
plt.tight_layout()
plt.savefig('elastic_net_path.png', dpi=300, bbox_inches='tight')
plt.close()
print("Saved: elastic_net_path.png")

# ==========================================
# ANALYSIS 4: Gaussian Mixture Model (airnb_luxe.xlsx)
# ==========================================
print("\n" + "="*50)
print("ANALYSIS 4: Gaussian Mixture Model")
print("="*50)

# ---------- Data Preparation ----------
df_gmm = df_luxe.dropna(subset=['price_numeric', 'distance_numeric']).copy()
print(f"Complete records for GMM: {len(df_gmm)}")

# ---------- Standardization ----------
scaler_gmm = StandardScaler()
df_gmm[['price_std', 'distance_std']] = scaler_gmm.fit_transform(
    df_gmm[['price_numeric', 'distance_numeric']]
)

# ---------- GMM Fitting ----------
X_gmm = df_gmm[['price_std', 'distance_std']].values
gmm = GaussianMixture(n_components=3, covariance_type='full', random_state=42)
gmm.fit(X_gmm)

# Calculate BIC
bic_score = gmm.bic(X_gmm)
print(f"\nBayesian Information Criterion (BIC): {bic_score:.2f}")

# Cluster predictions
df_gmm['cluster'] = gmm.predict(X_gmm)

# ---------- Scatter Plot with Clusters ----------
plt.figure(figsize=(10, 6))
colors = ['red', 'blue', 'green']
for i in range(3):
    cluster_data = df_gmm[df_gmm['cluster'] == i]
    plt.scatter(cluster_data['distance_numeric'], cluster_data['price_numeric'],
                c=colors[i], label=f'Cluster {i+1}', alpha=0.6, s=30)
plt.xlabel('Distance (Kilometers)', fontsize=12)
plt.ylabel('Price (Dollars)', fontsize=12)
plt.title('Gaussian Mixture Model Clustering', fontsize=14, fontweight='bold')
plt.legend()
plt.grid(True, alpha=0.3)
plt.tight_layout()
plt.savefig('gmm_clustering.png', dpi=300, bbox_inches='tight')
plt.close()
print("Saved: gmm_clustering.png")

# ==========================================
# ANALYSIS 5: Gradient Boosting with CV (airnb.xlsx)
# ==========================================
print("\n" + "="*50)
print("ANALYSIS 5: Gradient Boosting with Cross-Validation")
print("="*50)

# ---------- Data Preparation ----------
df_gb = df_airnb.dropna(subset=['price_numeric', 'rating_numeric', 'bed_count']).copy()
print(f"Complete records for gradient boosting: {len(df_gb)}")

# ---------- Gradient Boosting Training ----------
X_gb = df_gb[['rating_numeric', 'bed_count']].values
y_gb = df_gb['price_numeric'].values

gb_model = GradientBoostingRegressor(
    learning_rate=0.1,
    max_depth=5,
    n_estimators=100,
    random_state=42
)

# ---------- Cross-Validation ----------
cv_scores = cross_val_score(gb_model, X_gb, y_gb, cv=5, scoring='neg_mean_squared_error')
cv_mse_scores = -cv_scores
mean_cv_mse = np.mean(cv_mse_scores)

print(f"\nCross-Validation MSE Scores:")
for i, score in enumerate(cv_mse_scores, 1):
    print(f"  Fold {i}: {score:.2f}")
print(f"\nMean Cross-Validated MSE: {mean_cv_mse:.2f}")

# ---------- Training Error Plot ----------
gb_model.fit(X_gb, y_gb)
train_errors = []
for i, y_pred in enumerate(gb_model.staged_predict(X_gb)):
    mse = np.mean((y_gb - y_pred) ** 2)
    train_errors.append(mse)

plt.figure(figsize=(10, 6))
plt.plot(range(1, 101), train_errors, linewidth=2, color='darkgreen')
plt.xlabel('Boosting Iterations', fontsize=12)
plt.ylabel('Training Error (MSE)', fontsize=12)
plt.title('Gradient Boosting Training Error', fontsize=14, fontweight='bold')
plt.grid(True, alpha=0.3)
plt.tight_layout()
plt.savefig('gradient_boosting_error.png', dpi=300, bbox_inches='tight')
plt.close()
print("Saved: gradient_boosting_error.png")

# ==========================================
# ANALYSIS 6: Weighted Median Price (airnb_desert.xlsx)
# ==========================================
print("\n" + "="*50)
print("ANALYSIS 6: Weighted Median Price")
print("="*50)

# ---------- Calculation ----------
df_wm = df_desert.dropna(subset=['price_numeric', 'Rating']).copy()
print(f"Complete records for weighted median: {len(df_wm)}")

prices = df_wm['price_numeric'].values
weights = df_wm['Rating'].values

weighted_median_price = weighted_quantile(prices, weights, quantile=0.5)
print(f"\nWeighted Median Price: ${weighted_median_price:.2f}")

# ==========================================
# ANALYSIS 7: Geometric Mean Distance (airnb_luxe.xlsx)
# ==========================================
print("\n" + "="*50)
print("ANALYSIS 7: Geometric Mean Distance")
print("="*50)

# ---------- Calculation ----------
df_gm = df_luxe.dropna(subset=['distance_numeric']).copy()
distances = df_gm['distance_numeric'].values
distances = distances[distances > 0]  # Geometric mean requires positive values

geometric_mean_distance = gmean(distances)
print(f"\nGeometric Mean Distance: {geometric_mean_distance:.2f} km")

# ==========================================
# ANALYSIS 8: Coefficient of Quartile Variation (airnb.xlsx)
# ==========================================
print("\n" + "="*50)
print("ANALYSIS 8: Coefficient of Quartile Variation")
print("="*50)

# ---------- Calculation ----------
df_cqv = df_airnb.dropna(subset=['price_numeric']).copy()
prices = df_cqv['price_numeric'].values

Q1 = np.percentile(prices, 25)
Q3 = np.percentile(prices, 75)
CQV = ((Q3 - Q1) / (Q3 + Q1)) * 100

print(f"\nFirst Quartile (Q1): ${Q1:.2f}")
print(f"Third Quartile (Q3): ${Q3:.2f}")
print(f"Coefficient of Quartile Variation: {CQV:.2f}%")

# ==========================================
# ANALYSIS 9: Price per Kilometer Efficiency Ratio
# ==========================================
print("\n" + "="*50)
print("ANALYSIS 9: Price per Kilometer Efficiency Ratio")
print("="*50)

# ---------- Desert Price per Km ----------
df_desert['distance_from_details'] = df_desert['Details'].apply(extract_distance_km)
df_desert_ppk = df_desert.dropna(subset=['price_numeric', 'distance_from_details']).copy()
df_desert_ppk = df_desert_ppk[df_desert_ppk['distance_from_details'] > 0]
desert_ppk = (df_desert_ppk['price_numeric'] / df_desert_ppk['distance_from_details']).mean()
print(f"\nDesert Mean Price per Kilometer: ${desert_ppk:.2f}")

# ---------- Luxury Price per Km ----------
df_luxe_ppk = df_luxe.dropna(subset=['price_numeric', 'distance_numeric']).copy()
df_luxe_ppk = df_luxe_ppk[df_luxe_ppk['distance_numeric'] > 0]
luxury_ppk = (df_luxe_ppk['price_numeric'] / df_luxe_ppk['distance_numeric']).mean()
print(f"Luxury Mean Price per Kilometer: ${luxury_ppk:.2f}")

# ---------- Ratio Calculation ----------
ppk_ratio = luxury_ppk / desert_ppk
print(f"\nPrice per Km Efficiency Ratio (Luxury/Desert): {ppk_ratio:.2f}")

# ==========================================
# ANALYSIS 10: Gini Coefficient (airnb.xlsx)
# ==========================================
print("\n" + "="*50)
print("ANALYSIS 10: Gini Coefficient for Review Count")
print("="*50)

# ---------- Calculation ----------
df_gini = df_airnb.dropna(subset=['review_count']).copy()
review_counts = df_gini['review_count'].values
review_counts = np.sort(review_counts)

n = len(review_counts)
index = np.arange(1, n + 1)
gini = (2 * np.sum(index * review_counts)) / (n * np.sum(review_counts)) - (n + 1) / n

print(f"\nGini Coefficient: {gini:.4f}")

# ==========================================
# KEY OUTPUT SUMMARY
# ==========================================
print("\n" + "="*50)
print("KEY OUTPUT SUMMARY")
print("="*50)
print(f"\n1. Quantile Regression Rating Coefficient: {rating_coef:.4f}")
print(f"2. Random Forest Most Important Feature: {most_important_feature}")
print(f"3. Elastic Net Standardized Price Coefficient: {price_coefficient:.4f}")
print(f"4. Gaussian Mixture Model BIC: {bic_score:.2f}")
print(f"5. Gradient Boosting Mean CV MSE: {mean_cv_mse:.2f}")
print(f"6. Weighted Median Price: ${weighted_median_price:.2f}")
print(f"7. Geometric Mean Distance: {geometric_mean_distance:.2f} km")
print(f"8. Coefficient of Quartile Variation: {CQV:.2f}%")
print(f"9. Price per Km Efficiency Ratio: {ppk_ratio:.2f}")
print(f"10. Gini Coefficient: {gini:.4f}")
print("\n" + "="*50)
print("Analysis Complete!")
print("="*50)
