# ==========================================
# Advanced Job Market Statistical Analysis Script
# Requirements: pandas, numpy, matplotlib, scipy, scikit-learn, networkx
# Input files: glassdoor_jobs.xlsx, salary_data_cleaned.xlsx, eda_data.xlsx
# Output files: network_graph.png, pca_biplot.png, residuals_vs_fitted.png,
#               density_contour_plot.png, bootstrap_histogram.png
# ==========================================

import pandas as pd
import numpy as np
import matplotlib.pyplot as plt
import networkx as nx
from scipy.stats import wasserstein_distance, entropy
from scipy.spatial.distance import mahalanobis
from sklearn.covariance import EllipticEnvelope
from sklearn.decomposition import PCA
from sklearn.linear_model import RANSACRegressor
from sklearn.metrics import median_absolute_error
from sklearn.mixture import GaussianMixture
from statsmodels.stats.outliers_influence import variance_inflation_factor
import warnings
warnings.filterwarnings('ignore')

# ---------- Load Datasets ----------
print("Loading datasets...")
glassdoor_jobs = pd.read_excel('glassdoor_jobs.xlsx')
salary_data_cleaned = pd.read_excel('salary_data_cleaned.xlsx')
eda_data = pd.read_excel('eda_data.xlsx')

print(f"Glassdoor jobs shape: {glassdoor_jobs.shape}")
print(f"Salary data shape: {salary_data_cleaned.shape}")
print(f"EDA data shape: {eda_data.shape}")

# ---------- Data Cleaning: Remove Hourly Workers ----------
print("\nCleaning salary_data_cleaned...")
initial_salary_rows = len(salary_data_cleaned)
salary_data_cleaned = salary_data_cleaned[salary_data_cleaned['hourly'] != 1]

# Strip trailing whitespace from string columns
for col in salary_data_cleaned.select_dtypes(include=['object']).columns:
    salary_data_cleaned[col] = salary_data_cleaned[col].astype(str).str.strip()

print(f"Removed {initial_salary_rows - len(salary_data_cleaned)} hourly workers")

# ---------- Data Cleaning: EDA Data ----------
print("\nCleaning eda_data...")

# Strip trailing whitespace from string columns like job_state
for col in eda_data.select_dtypes(include=['object']).columns:
    eda_data[col] = eda_data[col].astype(str).str.strip()

# Coerce mixed-type numeric columns to proper numeric types
numeric_cols = ['num_comp', 'Rating', 'age', 'avg_salary', 'desc_len']
initial_eda_rows = len(eda_data)
for col in numeric_cols:
    if col in eda_data.columns:
        eda_data[col] = pd.to_numeric(eda_data[col], errors='coerce')

# Drop rows with missing values in critical numeric columns
eda_data = eda_data.dropna(subset=['num_comp'])
print(f"Removed {initial_eda_rows - len(eda_data)} rows with invalid values")

# ---------- Competitor Graph Construction ----------
print("\n" + "="*50)
print("COMPETITOR GRAPH ANALYSIS")
print("="*50)

# Parse Competitors column
G = nx.DiGraph()

for idx, row in glassdoor_jobs.iterrows():
    company = row['Company Name']
    competitors = row['Competitors']

    # Treat -1 (both int and string) as missing
    if pd.isna(competitors) or competitors == -1 or competitors == '-1':
        continue

    # Handle rating after newline - remove it
    competitors_str = str(competitors).split('\n')[0]

    # Parse competitors (comma-separated)
    comp_list = [c.strip() for c in competitors_str.split(',') if c.strip() and c.strip() != '-1']

    # Add edges
    for comp in comp_list:
        G.add_edge(company, comp)

# Calculate Graph Density
graph_density = nx.density(G)
print(f"\nOutput 1 - Graph Density: {graph_density:.6f}")

# ---------- Network Graph Visualization ----------
print("\nGenerating network graph visualization...")

# Get top 20 most connected companies
degree_dict = dict(G.degree())
top_20_nodes = sorted(degree_dict, key=degree_dict.get, reverse=True)[:20]
subgraph = G.subgraph(top_20_nodes)

plt.figure(figsize=(16, 12))
pos = nx.spring_layout(subgraph, k=2, iterations=50, seed=42)
node_sizes = [degree_dict[node] * 100 for node in subgraph.nodes()]
nx.draw_networkx(subgraph, pos,
                 node_size=node_sizes,
                 node_color='lightblue',
                 edge_color='gray',
                 alpha=0.7,
                 arrows=True,
                 arrowsize=15,
                 font_size=9,
                 font_weight='bold',
                 with_labels=True)
plt.title('Top 20 Most Connected Companies - Directed Competitor Network', fontsize=16, fontweight='bold')
plt.tight_layout()
plt.savefig('network_graph.png', dpi=300, bbox_inches='tight')
plt.close()
print("Saved: network_graph.png")

# ---------- Wasserstein Distance: CA vs NY Salaries ----------
print("\n" + "="*50)
print("WASSERSTEIN DISTANCE ANALYSIS")
print("="*50)

ca_salaries = salary_data_cleaned[salary_data_cleaned['job_state'] == 'CA']['avg_salary'].dropna()
ny_salaries = salary_data_cleaned[salary_data_cleaned['job_state'] == 'NY']['avg_salary'].dropna()

wasserstein_dist = wasserstein_distance(ca_salaries, ny_salaries)
print(f"\nOutput 2 - Wasserstein Distance (CA vs NY): {wasserstein_dist:.6f}")

# ---------- Shannon Entropy of Job State Distribution ----------
print("\n" + "="*50)
print("SHANNON ENTROPY ANALYSIS")
print("="*50)

state_counts = salary_data_cleaned['job_state'].value_counts()
state_probs = state_counts / state_counts.sum()
shannon_entropy = entropy(state_probs)
print(f"\nOutput 3 - Shannon Entropy (job_state): {shannon_entropy:.6f}")

# ---------- Multivariate Anomaly Detection ----------
print("\n" + "="*50)
print("MULTIVARIATE ANOMALY DETECTION")
print("="*50)

# Select numeric features
numeric_features = ['Rating', 'age', 'avg_salary', 'desc_len']
X_anomaly = eda_data[numeric_features].dropna()

# Fit EllipticEnvelope
ee = EllipticEnvelope(contamination=0.05, random_state=42)
outlier_labels = ee.fit_predict(X_anomaly)

outlier_proportion = (outlier_labels == -1).sum() / len(outlier_labels)
print(f"\nOutput 4 - Proportion of Outliers: {outlier_proportion:.6f}")

# ---------- PCA Biplot Visualization ----------
print("\nGenerating PCA biplot...")

pca = PCA(n_components=2, random_state=42)
X_pca = pca.fit_transform(X_anomaly)

plt.figure(figsize=(12, 8))
inliers = outlier_labels == 1
outliers = outlier_labels == -1

plt.scatter(X_pca[inliers, 0], X_pca[inliers, 1],
           c='blue', alpha=0.6, s=30, label='Inliers')
plt.scatter(X_pca[outliers, 0], X_pca[outliers, 1],
           c='red', alpha=0.8, s=60, marker='X', label='Outliers')

# Plot feature vectors
for i, feature in enumerate(numeric_features):
    plt.arrow(0, 0,
             pca.components_[0, i] * 3,
             pca.components_[1, i] * 3,
             head_width=0.15, head_length=0.15,
             fc='black', ec='black', linewidth=2)
    plt.text(pca.components_[0, i] * 3.5,
            pca.components_[1, i] * 3.5,
            feature, fontsize=12, fontweight='bold')

plt.xlabel(f'PC1 ({pca.explained_variance_ratio_[0]:.2%} variance)', fontsize=12)
plt.ylabel(f'PC2 ({pca.explained_variance_ratio_[1]:.2%} variance)', fontsize=12)
plt.title('PCA Biplot: Multivariate Anomaly Detection', fontsize=14, fontweight='bold')
plt.legend()
plt.grid(alpha=0.3)
plt.tight_layout()
plt.savefig('pca_biplot.png', dpi=300, bbox_inches='tight')
plt.close()
print("Saved: pca_biplot.png")

# ---------- RANSAC Regression ----------
print("\n" + "="*50)
print("RANSAC REGRESSION ANALYSIS")
print("="*50)

# Prepare data for RANSAC
predictors = ['Rating', 'age', 'desc_len']
X_ransac = eda_data[predictors].dropna()
y_ransac = eda_data.loc[X_ransac.index, 'avg_salary'].dropna()

# Align X and y
common_idx = X_ransac.index.intersection(y_ransac.index)
X_ransac = X_ransac.loc[common_idx]
y_ransac = y_ransac.loc[common_idx]

# Fit RANSAC Regressor
ransac = RANSACRegressor(random_state=42)
ransac.fit(X_ransac, y_ransac)

# Predictions
y_pred = ransac.predict(X_ransac)

# Calculate Median Absolute Error
mae = median_absolute_error(y_ransac, y_pred)
print(f"\nOutput 5 - Median Absolute Error (RANSAC): {mae:.6f}")

# ---------- Residuals vs Fitted Plot ----------
print("\nGenerating residuals vs fitted plot...")

residuals = y_ransac - y_pred
inlier_mask = ransac.inlier_mask_

plt.figure(figsize=(12, 8))
plt.scatter(y_pred[inlier_mask], residuals.values[inlier_mask],
           c='blue', alpha=0.6, s=30, label='Inliers')
plt.scatter(y_pred[~inlier_mask], residuals.values[~inlier_mask],
           c='red', alpha=0.8, s=60, marker='X', label='RANSAC Outliers')
plt.axhline(y=0, color='black', linestyle='--', linewidth=2)
plt.xlabel('Fitted Values', fontsize=12)
plt.ylabel('Residuals', fontsize=12)
plt.title('Residuals vs Fitted Plot (RANSAC Regression)', fontsize=14, fontweight='bold')
plt.legend()
plt.grid(alpha=0.3)
plt.tight_layout()
plt.savefig('residuals_vs_fitted.png', dpi=300, bbox_inches='tight')
plt.close()
print("Saved: residuals_vs_fitted.png")

# ---------- Variance Inflation Factor Analysis ----------
print("\n" + "="*50)
print("MULTICOLLINEARITY ANALYSIS (VIF)")
print("="*50)

# Prepare data for VIF
vif_features = ['Rating', 'age', 'desc_len', 'num_comp']
X_vif = eda_data[vif_features].dropna()

# Calculate VIF for each feature
vif_data = pd.DataFrame()
vif_data['Feature'] = vif_features
vif_data['VIF'] = [variance_inflation_factor(X_vif.values, i) for i in range(len(vif_features))]

max_vif = vif_data['VIF'].max()
print(f"\nVIF Values:")
print(vif_data)
print(f"\nOutput 6 - Highest VIF: {max_vif:.6f}")

# ---------- Gaussian Mixture Model ----------
print("\n" + "="*50)
print("GAUSSIAN MIXTURE MODEL ANALYSIS")
print("="*50)

# Fit GMM with 3 components
gmm_data = eda_data['avg_salary'].dropna().values.reshape(-1, 1)
gmm = GaussianMixture(n_components=3, random_state=42)
gmm.fit(gmm_data)

# Calculate BIC
bic = gmm.bic(gmm_data)
print(f"\nOutput 7 - BIC (GMM with 3 components): {bic:.6f}")

# ---------- Density Contour Plot ----------
print("\nGenerating density contour plot...")

# Prepare data for visualization
plot_data = eda_data[['avg_salary', 'Rating']].dropna()

plt.figure(figsize=(12, 8))

# Create 2D histogram for contour
H, xedges, yedges = np.histogram2d(plot_data['avg_salary'], plot_data['Rating'], bins=30)
extent = [xedges[0], xedges[-1], yedges[0], yedges[-1]]

plt.contourf(H.T, extent=extent, levels=15, cmap='viridis', alpha=0.7)
plt.colorbar(label='Density')

# Overlay GMM component means
component_means = gmm.means_.flatten()
for i, mean in enumerate(component_means):
    # For each component mean, find average rating
    component_probs = gmm.predict_proba(gmm_data)
    weights = component_probs[:, i]
    weighted_rating = (plot_data['Rating'].values * weights[:len(plot_data)]).sum() / weights[:len(plot_data)].sum()

    plt.scatter(mean, weighted_rating, c='red', s=300, marker='*',
               edgecolors='white', linewidths=2,
               label=f'Component {i+1}' if i == 0 else '', zorder=5)

plt.xlabel('Average Salary', fontsize=12)
plt.ylabel('Rating', fontsize=12)
plt.title('Density Contour Plot: avg_salary vs Rating with GMM Components', fontsize=14, fontweight='bold')
plt.legend(['GMM Component Means'], loc='best')
plt.tight_layout()
plt.savefig('density_contour_plot.png', dpi=300, bbox_inches='tight')
plt.close()
print("Saved: density_contour_plot.png")

# ---------- Bootstrap Analysis ----------
print("\n" + "="*50)
print("BOOTSTRAP CONFIDENCE INTERVAL ANALYSIS")
print("="*50)

# Filter data scientist and analyst roles
ds_salaries = eda_data[eda_data['job_simp'] == 'data scientist']['avg_salary'].dropna().values
analyst_salaries = eda_data[eda_data['job_simp'] == 'analyst']['avg_salary'].dropna().values

print(f"Data Scientists: {len(ds_salaries)} samples")
print(f"Analysts: {len(analyst_salaries)} samples")

# Bootstrap with 1000 iterations
n_iterations = 1000
np.random.seed(42)
bootstrap_diffs = []

for i in range(n_iterations):
    # Resample with replacement
    ds_sample = np.random.choice(ds_salaries, size=len(ds_salaries), replace=True)
    analyst_sample = np.random.choice(analyst_salaries, size=len(analyst_salaries), replace=True)

    # Calculate median difference
    median_diff = np.median(ds_sample) - np.median(analyst_sample)
    bootstrap_diffs.append(median_diff)

bootstrap_diffs = np.array(bootstrap_diffs)

# Calculate 95% confidence interval
ci_lower = np.percentile(bootstrap_diffs, 2.5)
ci_upper = np.percentile(bootstrap_diffs, 97.5)

print(f"\nOutput 8 - 95% CI Lower Bound: {ci_lower:.6f}")
print(f"Output 9 - 95% CI Upper Bound: {ci_upper:.6f}")

# ---------- Bootstrap Histogram ----------
print("\nGenerating bootstrap histogram...")

plt.figure(figsize=(12, 8))
plt.hist(bootstrap_diffs, bins=50, color='steelblue', alpha=0.7, edgecolor='black')
plt.axvline(ci_lower, color='red', linestyle='--', linewidth=2, label=f'95% CI Lower: {ci_lower:.2f}')
plt.axvline(ci_upper, color='red', linestyle='--', linewidth=2, label=f'95% CI Upper: {ci_upper:.2f}')
plt.axvline(np.median(bootstrap_diffs), color='green', linestyle='-', linewidth=2,
           label=f'Median: {np.median(bootstrap_diffs):.2f}')
plt.xlabel('Difference in Median Salary (Data Scientist - Analyst)', fontsize=12)
plt.ylabel('Frequency', fontsize=12)
plt.title('Bootstrap Distribution of Median Salary Difference\n(Data Scientist vs Analyst)',
         fontsize=14, fontweight='bold')
plt.legend()
plt.grid(alpha=0.3)
plt.tight_layout()
plt.savefig('bootstrap_histogram.png', dpi=300, bbox_inches='tight')
plt.close()
print("Saved: bootstrap_histogram.png")

# ---------- Summary of Key Outputs ----------
print("\n" + "="*50)
print("KEY OUTPUTS SUMMARY")
print("="*50)
print(f"Output 1 - Graph Density: {graph_density:.6f}")
print(f"Output 2 - Wasserstein Distance (CA vs NY): {wasserstein_dist:.6f}")
print(f"Output 3 - Shannon Entropy (job_state): {shannon_entropy:.6f}")
print(f"Output 4 - Proportion of Outliers: {outlier_proportion:.6f}")
print(f"Output 5 - Median Absolute Error (RANSAC): {mae:.6f}")
print(f"Output 6 - Highest VIF: {max_vif:.6f}")
print(f"Output 7 - BIC (GMM): {bic:.6f}")
print(f"Output 8 - 95% CI Lower Bound: {ci_lower:.6f}")
print(f"Output 9 - 95% CI Upper Bound: {ci_upper:.6f}")
print("\nAnalysis complete!")
