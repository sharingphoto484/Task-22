import pandas as pd
import numpy as np
import matplotlib.pyplot as plt
import seaborn as sns
from sklearn.preprocessing import StandardScaler
from sklearn.decomposition import PCA
from sklearn.cluster import KMeans
from sklearn.metrics import silhouette_score
from statsmodels.tsa.seasonal import seasonal_decompose
from scipy.stats import chi2_contingency, pearsonr
from sklearn.linear_model import LogisticRegression
import warnings
warnings.filterwarnings('ignore')

print("="*80)
print("INTEGRATED QUANTITATIVE ANALYSIS")
print("Sales Operations | Customer Behavior | Workforce Performance")
print("="*80)

# ============================================================================
# PART 1: SALES DATA ANALYSIS
# ============================================================================
print("\n" + "="*80)
print("PART 1: SALES DATA ANALYSIS")
print("="*80)

# Load sales data
sales_df = pd.read_csv('sales_data_sample.csv', encoding='latin-1')
print(f"\nOriginal sales data shape: {sales_df.shape}")

# Clean data: remove duplicates
sales_df = sales_df.drop_duplicates()
print(f"After removing duplicates: {sales_df.shape}")

# Impute missing numeric values with column means
numeric_cols_sales = sales_df.select_dtypes(include=[np.number]).columns
for col in numeric_cols_sales:
    if sales_df[col].isnull().any():
        sales_df[col].fillna(sales_df[col].mean(), inplace=True)

# Define Profit column
sales_df['Profit'] = sales_df['SALES'] - (sales_df['PRICEEACH'] * sales_df['QUANTITYORDERED'] * 0.7)
print(f"\nProfit column created. Sample values:\n{sales_df[['SALES', 'PRICEEACH', 'QUANTITYORDERED', 'Profit']].head()}")

# Aggregate total SALES by ORDERDATE and apply time-series decomposition
sales_df['ORDERDATE'] = pd.to_datetime(sales_df['ORDERDATE'], errors='coerce')
sales_ts = sales_df.groupby('ORDERDATE')['SALES'].sum().sort_index()

# Ensure we have enough data points and proper frequency
sales_ts = sales_ts.asfreq('D', fill_value=0)  # Daily frequency

print(f"\nTime series data points: {len(sales_ts)}")
print(f"Date range: {sales_ts.index.min()} to {sales_ts.index.max()}")

# Apply additive time-series decomposition
decomposition = seasonal_decompose(sales_ts, model='additive', period=30, extrapolate_trend='freq')

# Calculate trend magnitude (using standard deviation as magnitude measure)
trend_magnitude = np.nanstd(decomposition.trend)
print(f"\nTrend magnitude: {trend_magnitude:.3f}")

# Calculate relative strength of seasonal component
seasonal_variance = np.nanvar(decomposition.seasonal)
total_variance = np.nanvar(sales_ts)
seasonal_strength = seasonal_variance / total_variance if total_variance > 0 else 0
print(f"Seasonal component strength: {seasonal_strength:.3f}")

# PCA on continuous attributes
pca_features = ['SALES', 'QUANTITYORDERED', 'MSRP', 'Profit']
pca_data = sales_df[pca_features].dropna()

scaler = StandardScaler()
pca_data_scaled = scaler.fit_transform(pca_data)

pca = PCA(n_components=2)
pca.fit(pca_data_scaled)

variance_pc1 = pca.explained_variance_ratio_[0] * 100
variance_pc2 = pca.explained_variance_ratio_[1] * 100

print(f"\nPCA Results:")
print(f"Variance explained by PC1: {variance_pc1:.2f}%")
print(f"Variance explained by PC2: {variance_pc2:.2f}%")

# Generate correlation heatmap
corr_features = ['SALES', 'Profit', 'MSRP', 'QUANTITYORDERED']
corr_matrix = sales_df[corr_features].corr()

plt.figure(figsize=(10, 8))
sns.heatmap(corr_matrix, annot=True, fmt='.3f', cmap='coolwarm', center=0,
            square=True, linewidths=1, cbar_kws={"shrink": 0.8})
plt.title('Correlation Heatmap: Sales Data', fontsize=14, fontweight='bold')
plt.tight_layout()
plt.savefig('sales_correlation_heatmap.png', dpi=300, bbox_inches='tight')
print("\nCorrelation heatmap saved as 'sales_correlation_heatmap.png'")

# Calculate mean absolute correlation
# Get upper triangle values (excluding diagonal)
mask = np.triu(np.ones_like(corr_matrix, dtype=bool), k=1)
correlations = corr_matrix.where(mask).stack().values
mean_abs_correlation = np.mean(np.abs(correlations))
print(f"\nMean absolute correlation magnitude: {mean_abs_correlation:.3f}")

# ============================================================================
# PART 2: SHOPPING TRENDS AND CUSTOMER BEHAVIOR ANALYSIS
# ============================================================================
print("\n" + "="*80)
print("PART 2: CUSTOMER BEHAVIOR ANALYSIS")
print("="*80)

# Load shopping trends data
shopping_df = pd.read_csv('Shopping Trends And Customer Behaviour Dataset.csv')
print(f"\nOriginal shopping data shape: {shopping_df.shape}")

# Clean data: remove duplicates
shopping_df = shopping_df.drop_duplicates()
print(f"After removing duplicates: {shopping_df.shape}")

# Impute missing numeric values with column means
numeric_cols_shopping = shopping_df.select_dtypes(include=[np.number]).columns
for col in numeric_cols_shopping:
    if shopping_df[col].isnull().any():
        shopping_df[col].fillna(shopping_df[col].mean(), inplace=True)

# Define Annual Spending
shopping_df['Annual_Spending'] = shopping_df['Purchase Amount (USD)'] * 12

# Map Frequency of Purchases to numeric values
frequency_mapping = {
    'Weekly': 4,
    'Fortnightly': 2,
    'Monthly': 1,
    'Quarterly': 0.33,
    'Annually': 0.08,
    'Bi-Weekly': 2,
    'Every 3 Months': 0.33
}

# Check the actual column name and values
print(f"\nFrequency column values: {shopping_df['Frequency of Purchases'].unique()}")

# Apply mapping
shopping_df['Purchase_Frequency'] = shopping_df['Frequency of Purchases'].map(frequency_mapping)

# If there are unmapped values, fill with median
if shopping_df['Purchase_Frequency'].isnull().any():
    shopping_df['Purchase_Frequency'].fillna(shopping_df['Purchase_Frequency'].median(), inplace=True)

# Prepare data for clustering
clustering_features = ['Age', 'Annual_Spending', 'Purchase_Frequency']
clustering_data = shopping_df[clustering_features].dropna()

# Standardize variables
scaler_cluster = StandardScaler()
clustering_data_scaled = scaler_cluster.fit_transform(clustering_data)

# Determine optimal cluster count using silhouette analysis
silhouette_scores = []
K_range = range(2, 11)

for k in K_range:
    kmeans = KMeans(n_clusters=k, random_state=42, n_init=10)
    cluster_labels = kmeans.fit_predict(clustering_data_scaled)
    silhouette_avg = silhouette_score(clustering_data_scaled, cluster_labels)
    silhouette_scores.append(silhouette_avg)

optimal_k = K_range[np.argmax(silhouette_scores)]
optimal_silhouette = max(silhouette_scores)

print(f"\nK-means Clustering Results:")
print(f"Optimal number of clusters: {optimal_k:.2f}")
print(f"Corresponding silhouette score: {optimal_silhouette:.2f}")

# Calculate Pearson correlation between Age and Annual Spending
age_spending_corr, _ = pearsonr(shopping_df['Age'], shopping_df['Annual_Spending'])
print(f"\nPearson correlation (Age vs Annual Spending): {age_spending_corr:.4f}")

# ============================================================================
# PART 3: HR WORKFORCE ANALYSIS
# ============================================================================
print("\n" + "="*80)
print("PART 3: WORKFORCE PERFORMANCE ANALYSIS")
print("="*80)

# Load HR data
hr_df = pd.read_csv('HR_comma_sep.csv')
print(f"\nOriginal HR data shape: {hr_df.shape}")

# Clean data: remove duplicates
hr_df = hr_df.drop_duplicates()
print(f"After removing duplicates: {hr_df.shape}")

# Impute missing numeric values with column means
numeric_cols_hr = hr_df.select_dtypes(include=[np.number]).columns
for col in numeric_cols_hr:
    if hr_df[col].isnull().any():
        hr_df[col].fillna(hr_df[col].mean(), inplace=True)

# Prepare data for SEM
# Standardize indicators for SEM
hr_df['satisfaction_level_std'] = (hr_df['satisfaction_level'] - hr_df['satisfaction_level'].mean()) / hr_df['satisfaction_level'].std()
hr_df['last_evaluation_std'] = (hr_df['last_evaluation'] - hr_df['last_evaluation'].mean()) / hr_df['last_evaluation'].std()

# Implement SEM using PCA for latent variable construction and path analysis
print("\nStructural Equation Model Implementation:")
print("Using PCA-based approach for latent variable construction")

# Step 1: Create latent variable JobSat using PCA on satisfaction_level and last_evaluation
indicators = hr_df[['satisfaction_level_std', 'last_evaluation_std']].values
pca_sem = PCA(n_components=1)
JobSat_scores = pca_sem.fit_transform(indicators).flatten()

# Get factor loadings (standardized path coefficients from latent to indicators)
loadings = pca_sem.components_[0]
loading_satisfaction = loadings[0] * np.sqrt(pca_sem.explained_variance_[0])
loading_evaluation = loadings[1] * np.sqrt(pca_sem.explained_variance_[0])

print(f"\nMeasurement Model - Factor Loadings:")
print(f"JobSat -> satisfaction_level: {loading_satisfaction:.4f}")
print(f"JobSat -> last_evaluation: {loading_evaluation:.4f}")

# Step 2: Structural model - regress 'left' on JobSat
# Standardize JobSat scores
JobSat_std = (JobSat_scores - JobSat_scores.mean()) / JobSat_scores.std()

# Calculate standardized path coefficient (correlation for single predictor)
path_jobsat_left = pearsonr(JobSat_std, hr_df['left'])[0]

print(f"\nStructural Model - Path Coefficient:")
print(f"JobSat -> Turnover: {path_jobsat_left:.4f}")

# Also report direct effects for completeness
corr_sat_left = pearsonr(hr_df['satisfaction_level_std'], hr_df['left'])[0]
corr_eval_left = pearsonr(hr_df['last_evaluation_std'], hr_df['left'])[0]

print(f"\nDirect Effects (for reference):")
print(f"satisfaction_level -> Turnover: {corr_sat_left:.4f}")
print(f"last_evaluation -> Turnover: {corr_eval_left:.4f}")

# Store the key coefficient for later comparison
jobsat_turnover_coef = abs(path_jobsat_left)

# Chi-Square test of independence between salary and left
contingency_table = pd.crosstab(hr_df['salary'], hr_df['left'])
chi2, p_value, dof, expected = chi2_contingency(contingency_table)

print(f"\nChi-Square Test Results:")
print(f"Chi-Square statistic: {chi2:.4f}")
print(f"p-value: {p_value:.4f}")

# ============================================================================
# FINAL COMPARISON AND SUMMARY
# ============================================================================
print("\n" + "="*80)
print("FINAL COMPARISON AND SUMMARY")
print("="*80)

exceeds = jobsat_turnover_coef > mean_abs_correlation

print(f"\nMean absolute correlation (Sales heatmap): {mean_abs_correlation:.3f}")
print(f"Standardized path coefficient (JobSat -> Turnover): {jobsat_turnover_coef:.4f}")
print(f"\nDoes JobSat->Turnover coefficient exceed mean correlation? {exceeds}")

# ============================================================================
# COMPREHENSIVE RESULTS SUMMARY
# ============================================================================
print("\n" + "="*80)
print("COMPREHENSIVE RESULTS SUMMARY")
print("="*80)

print("\n[SALES DATA ANALYSIS]")
print(f"  • Trend magnitude: {trend_magnitude:.3f}")
print(f"  • Seasonal strength: {seasonal_strength:.3f}")
print(f"  • PCA variance (PC1): {variance_pc1:.2f}%")
print(f"  • PCA variance (PC2): {variance_pc2:.2f}%")
print(f"  • Mean absolute correlation: {mean_abs_correlation:.3f}")

print("\n[CUSTOMER BEHAVIOR ANALYSIS]")
print(f"  • Optimal clusters: {optimal_k:.2f}")
print(f"  • Silhouette score: {optimal_silhouette:.2f}")
print(f"  • Pearson correlation (Age vs Annual Spending): {age_spending_corr:.4f}")

print("\n[WORKFORCE PERFORMANCE ANALYSIS]")
print(f"  • Path coefficient (JobSat -> satisfaction_level): {loading_satisfaction:.4f}")
print(f"  • Path coefficient (JobSat -> last_evaluation): {loading_evaluation:.4f}")
print(f"  • Path coefficient (JobSat -> Turnover): {path_jobsat_left:.4f}")
print(f"  • Chi-Square statistic: {chi2:.4f}")
print(f"  • p-value: {p_value:.4f}")

print("\n[COMPARATIVE ANALYSIS]")
print(f"  • JobSat->Turnover exceeds mean correlation: {exceeds}")

print("\n" + "="*80)
print("ANALYSIS COMPLETE")
print("="*80)
