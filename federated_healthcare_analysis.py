# ==========================================
# Federated Healthcare Efficiency Analysis Script
# ==========================================
# Requirements: pandas, numpy, matplotlib, scipy, scikit-learn, statsmodels
# Input files: FL_BIHDS_dataset.csv, optimization_metrics.csv, healthcare_data.csv (in same directory)
# Output files: flow_field_visualization.png, analysis_summary.json
# ==========================================

import pandas as pd
import numpy as np
import matplotlib.pyplot as plt
import json
from scipy import stats
from sklearn.preprocessing import StandardScaler
from sklearn.linear_model import LinearRegression
from sklearn.cluster import KMeans
from sklearn.metrics import silhouette_score, r2_score
from statsmodels.tsa.stattools import grangercausalitytests
import warnings
warnings.filterwarnings('ignore')

# ---------- Load CSVs Robustly ----------
print("Loading datasets...")
fl_bihds_df = pd.read_csv('FL_BIHDS_dataset.csv')
optimization_df = pd.read_csv('optimization_metrics.csv')
healthcare_df = pd.read_csv('healthcare_data.csv')
print(f"Loaded FL_BIHDS: {fl_bihds_df.shape}, Optimization: {optimization_df.shape}, Healthcare: {healthcare_df.shape}")

# ---------- Data Cleaning: Remove Missing and Non-Numeric Values ----------
print("\nCleaning datasets...")

def clean_dataset(df, numeric_cols):
    """Remove rows with missing or non-numeric values in specified columns"""
    df_clean = df.copy()
    initial_rows = len(df_clean)

    # Remove rows with missing values in numeric columns
    df_clean = df_clean.dropna(subset=numeric_cols)

    # Convert to numeric and remove non-numeric values
    for col in numeric_cols:
        df_clean[col] = pd.to_numeric(df_clean[col], errors='coerce')

    # Drop rows with NaN after conversion
    df_clean = df_clean.dropna(subset=numeric_cols)

    final_rows = len(df_clean)
    print(f"  Removed {initial_rows - final_rows} rows with missing/non-numeric values")

    return df_clean

# Clean FL_BIHDS dataset
fl_numeric_cols = ['Round', 'Latency_ms', 'Bandwidth_Mbps', 'Aggregation_Time_ms', 'Resource_Allocation_Score']
fl_bihds_clean = clean_dataset(fl_bihds_df, fl_numeric_cols)

# Clean optimization_metrics dataset
opt_numeric_cols = ['Data_Size_MB', 'Latency_ms', 'Bandwidth_Mbps', 'Aggregation_Time_ms', 'Resource_Allocation_Score']
optimization_clean = clean_dataset(optimization_df, opt_numeric_cols)

# Clean healthcare_data dataset
health_numeric_cols = ['BP', 'Glucose', 'HeartRate', 'Cholesterol']
healthcare_clean = clean_dataset(healthcare_df, health_numeric_cols)

print(f"\nCleaned datasets:")
print(f"  FL_BIHDS: {fl_bihds_clean.shape}")
print(f"  Optimization: {optimization_clean.shape}")
print(f"  Healthcare: {healthcare_clean.shape}")

# ---------- 1. Multiple Linear Regression with Standardized Predictors ----------
print("\n" + "="*60)
print("1. MULTIPLE LINEAR REGRESSION")
print("="*60)

# Prepare data for regression
X_reg = optimization_clean[['Latency_ms', 'Bandwidth_Mbps', 'Data_Size_MB']].values
y_reg = optimization_clean['Resource_Allocation_Score'].values

# Standardize predictors
scaler = StandardScaler()
X_reg_standardized = scaler.fit_transform(X_reg)

# Fit regression model
reg_model = LinearRegression()
reg_model.fit(X_reg_standardized, y_reg)

# Calculate R-squared
y_pred = reg_model.predict(X_reg_standardized)
r_squared = r2_score(y_reg, y_pred)

print(f"Dependent Variable: Resource_Allocation_Score")
print(f"Predictors (standardized): Latency_ms, Bandwidth_Mbps, Data_Size_MB")
print(f"R-squared: {r_squared:.3f}")
print(f"Coefficients:")
for i, pred_name in enumerate(['Latency_ms', 'Bandwidth_Mbps', 'Data_Size_MB']):
    print(f"  {pred_name}: {reg_model.coef_[i]:.4f}")

# Store regression coefficients for later analysis
regression_coefficients = {
    'Latency_ms': abs(reg_model.coef_[0]),
    'Bandwidth_Mbps': abs(reg_model.coef_[1]),
    'Aggregation_Time_ms': 0  # Not in regression model, placeholder
}

# ---------- 2. Granger Causality Test ----------
print("\n" + "="*60)
print("2. GRANGER CAUSALITY TEST")
print("="*60)

# Aggregate FL_BIHDS data by Round to get time series
fl_ts = fl_bihds_clean.groupby('Round').agg({
    'Bandwidth_Mbps': 'mean',
    'Latency_ms': 'mean'
}).reset_index().sort_values('Round')

# Prepare data for Granger causality (Latency_ms ~ Bandwidth_Mbps)
granger_data = fl_ts[['Latency_ms', 'Bandwidth_Mbps']].values

# Conduct Granger causality test with max lag of 1
print(f"Testing: Does Bandwidth_Mbps help predict Latency_ms?")
print(f"Max lag: 1 round")

try:
    granger_result = grangercausalitytests(granger_data, maxlag=1, verbose=False)

    # Extract F-statistic and p-value for lag 1
    f_stat = granger_result[1][0]['ssr_ftest'][0]
    p_value = granger_result[1][0]['ssr_ftest'][1]

    print(f"F-statistic: {f_stat:.3f}")
    print(f"P-value: {p_value:.3f}")

    if p_value < 0.05:
        print(f"Result: Bandwidth_Mbps significantly predicts Latency_ms (p < 0.05)")
    else:
        print(f"Result: Bandwidth_Mbps does not significantly predict Latency_ms (p >= 0.05)")

except Exception as e:
    print(f"Error in Granger causality test: {e}")
    f_stat = 0.0
    p_value = 1.0

# ---------- 3. One-Way ANOVA ----------
print("\n" + "="*60)
print("3. ONE-WAY ANOVA")
print("="*60)

# Group BP by Diagnosis_Label
diagnosis_groups = healthcare_clean.groupby('Diagnosis_Label')['BP'].apply(list)

print(f"Testing: Does average BP differ among Diagnosis_Label groups?")
print(f"Groups: {list(diagnosis_groups.index)}")
print(f"Group sizes: {[len(g) for g in diagnosis_groups]}")

# Conduct one-way ANOVA
f_stat_anova, p_value_anova = stats.f_oneway(*diagnosis_groups)

print(f"F-statistic: {f_stat_anova:.3f}")
print(f"P-value: {p_value_anova:.4f}")

if p_value_anova < 0.05:
    print(f"Result: BP differs significantly among diagnosis groups (p < 0.05)")
else:
    print(f"Result: BP does not differ significantly among diagnosis groups (p >= 0.05)")

# ---------- 4. K-Means Clustering with Silhouette Score ----------
print("\n" + "="*60)
print("4. K-MEANS CLUSTERING")
print("="*60)

# Prepare clustering data
cluster_features = ['BP', 'Glucose', 'HeartRate', 'Cholesterol']
X_cluster = healthcare_clean[cluster_features].values

# Standardize features for clustering
scaler_cluster = StandardScaler()
X_cluster_scaled = scaler_cluster.fit_transform(X_cluster)

print(f"Clustering features: {cluster_features}")
print(f"Testing k from 2 to 10...")

# Find optimal number of clusters using silhouette score
silhouette_scores = {}
for k in range(2, 11):
    kmeans = KMeans(n_clusters=k, random_state=42, n_init=10)
    cluster_labels = kmeans.fit_predict(X_cluster_scaled)
    silhouette_avg = silhouette_score(X_cluster_scaled, cluster_labels)
    silhouette_scores[k] = silhouette_avg
    print(f"  k={k}: Silhouette Score = {silhouette_avg:.4f}")

# Find optimal k
optimal_k = max(silhouette_scores, key=silhouette_scores.get)
optimal_silhouette = silhouette_scores[optimal_k]

print(f"\nOptimal number of clusters: {optimal_k}")
print(f"Best Silhouette Score: {optimal_silhouette:.4f}")

# ---------- 5. Causal Inference Analysis ----------
print("\n" + "="*60)
print("5. CAUSAL INFERENCE ANALYSIS")
print("="*60)

# Estimate causal effect of Bandwidth_Mbps on Aggregation_Time_ms, controlling for Data_Size_MB
print(f"Estimating: Causal effect of Bandwidth_Mbps on Aggregation_Time_ms")
print(f"Control variable: Data_Size_MB")

X_causal = optimization_clean[['Bandwidth_Mbps', 'Data_Size_MB']].values
y_causal = optimization_clean['Aggregation_Time_ms'].values

# Fit linear model for causal inference
causal_model = LinearRegression()
causal_model.fit(X_causal, y_causal)

# Extract causal coefficient for Bandwidth_Mbps
causal_coefficient = causal_model.coef_[0]

print(f"Causal coefficient (Bandwidth_Mbps -> Aggregation_Time_ms): {causal_coefficient:.4f}")
print(f"Control coefficient (Data_Size_MB): {causal_model.coef_[1]:.4f}")

# Store for influence analysis
causal_effect_magnitude = abs(causal_coefficient)

# ---------- 6. Flow Field Visualization ----------
print("\n" + "="*60)
print("6. FLOW FIELD VISUALIZATION")
print("="*60)

# Aggregate FL_BIHDS data for visualization
fl_viz = fl_bihds_clean.copy()

# Create grid for flow field
x_bw = fl_viz['Bandwidth_Mbps'].values
y_lat = fl_viz['Latency_ms'].values
resource_score = fl_viz['Resource_Allocation_Score'].values

# Calculate directional vectors (gradient of Resource_Allocation_Score)
# Use Resource_Allocation_Score as magnitude, compute direction from local gradients
print(f"Creating flow field with {len(fl_viz)} data points")

# For flow field, we'll compute direction vectors based on local gradients
# Create a grid and interpolate
from scipy.interpolate import griddata

# Create regular grid
grid_x, grid_y = np.mgrid[x_bw.min():x_bw.max():20j, y_lat.min():y_lat.max():20j]

# Interpolate resource scores onto grid
grid_z = griddata((x_bw, y_lat), resource_score, (grid_x, grid_y), method='cubic')

# Compute gradients (U, V components)
U, V = np.gradient(grid_z)

# Calculate vector magnitudes
magnitudes = np.sqrt(U**2 + V**2)
mean_magnitude = np.nanmean(magnitudes)

print(f"Mean vector magnitude: {mean_magnitude:.3f}")

# Create visualization
plt.figure(figsize=(12, 8))
plt.quiver(grid_x, grid_y, U, V, magnitudes, cmap='viridis', alpha=0.7)
plt.scatter(x_bw, y_lat, c=resource_score, s=30, cmap='coolwarm', edgecolors='black', linewidths=0.5)
plt.colorbar(label='Resource Allocation Score')
plt.xlabel('Bandwidth (Mbps)', fontsize=12)
plt.ylabel('Latency (ms)', fontsize=12)
plt.title('Flow Field Visualization: Federated Learning Performance', fontsize=14, fontweight='bold')
plt.grid(True, alpha=0.3)
plt.tight_layout()
plt.savefig('flow_field_visualization.png', dpi=300, bbox_inches='tight')
print("Flow field visualization saved: flow_field_visualization.png")
plt.close()

# ---------- 7. Identify Strongest Influence Variable ----------
print("\n" + "="*60)
print("7. STRONGEST INFLUENCE VARIABLE ANALYSIS")
print("="*60)

# Analyze influence based on:
# 1. Regression coefficients (standardized)
# 2. Causal coefficients
# 3. Correlation with Resource_Allocation_Score

print("Analyzing variable influences on Resource_Allocation_Score...")

# Calculate correlations
corr_bandwidth = abs(optimization_clean[['Bandwidth_Mbps', 'Resource_Allocation_Score']].corr().iloc[0, 1])
corr_latency = abs(optimization_clean[['Latency_ms', 'Resource_Allocation_Score']].corr().iloc[0, 1])
corr_aggregation = abs(optimization_clean[['Aggregation_Time_ms', 'Resource_Allocation_Score']].corr().iloc[0, 1])

print("\n1. Correlation with Resource_Allocation_Score:")
print(f"   Bandwidth_Mbps: {corr_bandwidth:.4f}")
print(f"   Latency_ms: {corr_latency:.4f}")
print(f"   Aggregation_Time_ms: {corr_aggregation:.4f}")

print("\n2. Regression coefficients (standardized):")
print(f"   Bandwidth_Mbps: {abs(reg_model.coef_[1]):.4f}")
print(f"   Latency_ms: {abs(reg_model.coef_[0]):.4f}")
print(f"   Aggregation_Time_ms: Not in model")

print("\n3. Causal effect magnitude:")
print(f"   Bandwidth_Mbps -> Aggregation_Time_ms: {causal_effect_magnitude:.4f}")

# Aggregate regression with Aggregation_Time_ms
X_reg_full = optimization_clean[['Latency_ms', 'Bandwidth_Mbps', 'Aggregation_Time_ms']].values
X_reg_full_std = StandardScaler().fit_transform(X_reg_full)
reg_full = LinearRegression()
reg_full.fit(X_reg_full_std, y_reg)

print("\n4. Full regression coefficients (all three variables, standardized):")
print(f"   Latency_ms: {abs(reg_full.coef_[0]):.4f}")
print(f"   Bandwidth_Mbps: {abs(reg_full.coef_[1]):.4f}")
print(f"   Aggregation_Time_ms: {abs(reg_full.coef_[2]):.4f}")

# Determine strongest influence based on regression coefficients
influence_scores = {
    'Bandwidth_Mbps': abs(reg_full.coef_[1]),
    'Latency_ms': abs(reg_full.coef_[0]),
    'Aggregation_Time_ms': abs(reg_full.coef_[2])
}

strongest_variable = max(influence_scores, key=influence_scores.get)

print(f"\nStrongest influence variable: {strongest_variable}")
print(f"Influence score: {influence_scores[strongest_variable]:.4f}")

# ---------- 8. Generate Analysis Summary JSON ----------
print("\n" + "="*60)
print("8. GENERATING ANALYSIS SUMMARY")
print("="*60)

summary = {
    "analysis_metadata": {
        "script": "Federated Healthcare Efficiency Analysis",
        "datasets": {
            "FL_BIHDS": {"original_rows": int(len(fl_bihds_df)), "clean_rows": int(len(fl_bihds_clean))},
            "optimization_metrics": {"original_rows": int(len(optimization_df)), "clean_rows": int(len(optimization_clean))},
            "healthcare_data": {"original_rows": int(len(healthcare_df)), "clean_rows": int(len(healthcare_clean))}
        }
    },
    "multiple_linear_regression": {
        "dependent_variable": "Resource_Allocation_Score",
        "predictors": ["Latency_ms", "Bandwidth_Mbps", "Data_Size_MB"],
        "standardized": "yes",
        "r_squared": round(r_squared, 3),
        "coefficients": {
            "Latency_ms": round(float(reg_model.coef_[0]), 4),
            "Bandwidth_Mbps": round(float(reg_model.coef_[1]), 4),
            "Data_Size_MB": round(float(reg_model.coef_[2]), 4)
        }
    },
    "granger_causality_test": {
        "hypothesis": "Bandwidth_Mbps helps predict Latency_ms",
        "max_lag": 1,
        "f_statistic": round(float(f_stat), 3),
        "p_value": round(float(p_value), 3),
        "significant": "yes" if p_value < 0.05 else "no"
    },
    "one_way_anova": {
        "test": "BP across Diagnosis_Label groups",
        "f_statistic": round(float(f_stat_anova), 3),
        "p_value": round(float(p_value_anova), 4),
        "significant": "yes" if p_value_anova < 0.05 else "no"
    },
    "kmeans_clustering": {
        "features": cluster_features,
        "optimal_clusters": int(optimal_k),
        "silhouette_score": round(float(optimal_silhouette), 4),
        "all_silhouette_scores": {int(k): round(float(v), 4) for k, v in silhouette_scores.items()}
    },
    "causal_inference": {
        "treatment": "Bandwidth_Mbps",
        "outcome": "Aggregation_Time_ms",
        "control": "Data_Size_MB",
        "causal_coefficient": round(float(causal_coefficient), 4)
    },
    "flow_field_visualization": {
        "x_axis": "Bandwidth_Mbps",
        "y_axis": "Latency_ms",
        "vector_field": "Resource_Allocation_Score gradient",
        "mean_vector_magnitude": round(float(mean_magnitude), 3),
        "output_file": "flow_field_visualization.png"
    },
    "strongest_influence_variable": {
        "variable_name": strongest_variable,
        "influence_scores": {k: round(float(v), 4) for k, v in influence_scores.items()},
        "method": "Multiple linear regression with all three variables (standardized)"
    }
}

# Save to JSON
with open('analysis_summary.json', 'w') as f:
    json.dump(summary, f, indent=2)

print("Analysis summary saved: analysis_summary.json")

# ---------- Print Final Summary ----------
print("\n" + "="*60)
print("FINAL SUMMARY")
print("="*60)
print(f"1. Multiple Linear Regression R²: {r_squared:.3f}")
print(f"2. Granger Causality F-stat: {f_stat:.3f}, P-value: {p_value:.3f}")
print(f"3. One-Way ANOVA F-stat: {f_stat_anova:.3f}")
print(f"4. Optimal K-means Clusters: {optimal_k}")
print(f"5. Causal Coefficient (Bandwidth -> Aggregation): {causal_coefficient:.4f}")
print(f"6. Mean Flow Field Vector Magnitude: {mean_magnitude:.3f}")
print(f"7. Strongest Influence Variable: {strongest_variable}")
print("\nAll outputs saved successfully!")
print("="*60)
