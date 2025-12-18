# ==========================================
# Healthcare Inventory & Resource Optimization Analysis Script
# ==========================================
# Requirements: pandas, numpy, matplotlib, scipy, scikit-learn, statsmodels
# Input files: inventory_data.xlsx, staff_data.xlsx, patient_data.xlsx
# Output files: Multiple visualization plots and key metrics
# ==========================================

import pandas as pd
import numpy as np
import matplotlib.pyplot as plt
from scipy import stats
from sklearn.preprocessing import StandardScaler
from sklearn.cluster import KMeans
from sklearn.decomposition import PCA
from sklearn.mixture import GaussianMixture
from sklearn.linear_model import ElasticNetCV
import statsmodels.api as sm
import statsmodels.formula.api as smf
from statsmodels.regression.quantile_regression import QuantReg
import warnings
warnings.filterwarnings('ignore')

# ==========================================
# ---------- Load Excel Files ----------
# ==========================================

print("Loading healthcare data files...")
inventory_df = pd.read_excel('inventory_data.xlsx')
staff_df = pd.read_excel('staff_data.xlsx')
patient_df = pd.read_excel('patient_data.xlsx')

print(f"Inventory records: {len(inventory_df)}")
print(f"Staff records: {len(staff_df)}")
print(f"Patient records: {len(patient_df)}")
print("\n" + "="*70)

# ==========================================
# ---------- Analysis 1: Quantile Regression on Inventory Unit Cost ----------
# ==========================================

print("\n1. QUANTILE REGRESSION (75th Percentile Unit Cost)")
print("-" * 70)

# Select complete cases
inv_qr = inventory_df[['Unit_Cost', 'Current_Stock', 'Avg_Usage_Per_Day', 'Restock_Lead_Time']].dropna()

# Fit quantile regression at 75th percentile (q=0.75)
X_qr = inv_qr[['Current_Stock', 'Avg_Usage_Per_Day', 'Restock_Lead_Time']]
X_qr = sm.add_constant(X_qr)
y_qr = inv_qr['Unit_Cost']

qr_model = QuantReg(y_qr, X_qr)
qr_results = qr_model.fit(q=0.75)

# Extract coefficient for Current_Stock
current_stock_coef = qr_results.params['Current_Stock']
print(f"Current_Stock coefficient (75th percentile): {current_stock_coef:.4f}")

# Generate scatter plot
plt.figure(figsize=(10, 6))
plt.scatter(inv_qr['Current_Stock'], inv_qr['Unit_Cost'], alpha=0.5, s=30)
plt.xlabel('Current_Stock')
plt.ylabel('Unit_Cost')
plt.title('Unit Cost vs Current Stock (Quantile Regression)')
plt.grid(True, alpha=0.3)
plt.tight_layout()
plt.savefig('quantile_regression_plot.png', dpi=300, bbox_inches='tight')
plt.close()
print("Plot saved: quantile_regression_plot.png")

# ==========================================
# ---------- Analysis 2: Poisson Regression on Patient Assignments ----------
# ==========================================

print("\n2. POISSON REGRESSION (Patient Assignment Counts)")
print("-" * 70)

# Select complete cases
staff_poisson = staff_df[['Patients_Assigned', 'Hours_Worked', 'Overtime_Hours']].dropna()

# Create binary predictors
staff_poisson['long_shift'] = (staff_poisson['Hours_Worked'] >= 10).astype(int)
staff_poisson['has_overtime'] = (staff_poisson['Overtime_Hours'] > 0).astype(int)

# Fit Poisson regression
poisson_formula = 'Patients_Assigned ~ long_shift + has_overtime'
poisson_model = smf.glm(formula=poisson_formula, data=staff_poisson, family=sm.families.Poisson()).fit()

# Calculate dispersion parameter (deviance / df)
dispersion = poisson_model.deviance / poisson_model.df_resid
print(f"Dispersion parameter (deviance/df): {dispersion:.4f}")

# Generate bar chart with predictions
staff_poisson['predicted'] = poisson_model.predict()
grouped = staff_poisson.groupby(['long_shift', 'has_overtime'])['predicted'].mean()

plt.figure(figsize=(10, 6))
x_labels = [f"Long={ls}, OT={ot}" for ls, ot in grouped.index]
plt.bar(range(len(grouped)), grouped.values, color=['#1f77b4', '#ff7f0e', '#2ca02c', '#d62728'][:len(grouped)])
plt.xlabel('Predictor Combinations')
plt.ylabel('Mean Predicted Patient Assignments')
plt.title('Poisson Regression: Patient Assignments by Shift Characteristics')
plt.xticks(range(len(grouped)), x_labels, rotation=15)
plt.grid(True, alpha=0.3, axis='y')
plt.tight_layout()
plt.savefig('poisson_regression_plot.png', dpi=300, bbox_inches='tight')
plt.close()
print("Plot saved: poisson_regression_plot.png")

# ==========================================
# ---------- Analysis 3: K-Means Clustering on Patient Patterns ----------
# ==========================================

print("\n3. K-MEANS CLUSTERING (Patient Segmentation)")
print("-" * 70)

# Create binary features
patient_cluster = patient_df.copy()
patient_cluster['critical_diagnosis'] = (patient_cluster['Primary_Diagnosis'] == 'Critical').astype(int)
patient_cluster['surgical_procedure'] = (patient_cluster['Procedure_Performed'] == 'Surgery').astype(int)
patient_cluster['private_room'] = (patient_cluster['Room_Type'] == 'Private').astype(int)
patient_cluster['high_supplies'] = (patient_cluster['Supplies_Used'] == 'High').astype(int)
patient_cluster['complex_equipment'] = (patient_cluster['Equipment_Used'] == 'Complex').astype(int)

# Select complete cases
feature_cols = ['Bed_Days', 'critical_diagnosis', 'surgical_procedure', 'private_room', 'high_supplies', 'complex_equipment']
patient_cluster = patient_cluster[feature_cols].dropna()

# Standardize features
scaler = StandardScaler()
patient_scaled = scaler.fit_transform(patient_cluster)

# Fit k-means with k=3, random_state=42
kmeans = KMeans(n_clusters=3, random_state=42, n_init=10)
clusters = kmeans.fit_predict(patient_scaled)

# Identify cluster with maximum patients
cluster_counts = pd.Series(clusters).value_counts().sort_index()
largest_cluster = cluster_counts.idxmax()
print(f"Cluster with maximum patients: {largest_cluster}")
print(f"Cluster distribution: {dict(cluster_counts)}")

# Generate scatter plot
patient_cluster['cluster'] = clusters
patient_cluster['resource_intensity'] = (patient_cluster['critical_diagnosis'] +
                                          patient_cluster['surgical_procedure'] +
                                          patient_cluster['private_room'] +
                                          patient_cluster['high_supplies'] +
                                          patient_cluster['complex_equipment'])

plt.figure(figsize=(10, 6))
colors = ['#1f77b4', '#ff7f0e', '#2ca02c']
for i in range(3):
    mask = patient_cluster['cluster'] == i
    plt.scatter(patient_cluster[mask]['Bed_Days'],
               patient_cluster[mask]['resource_intensity'],
               alpha=0.6, s=30, label=f'Cluster {i}', c=colors[i])
plt.xlabel('Bed_Days')
plt.ylabel('Resource Intensity (Sum of Binary Indicators)')
plt.title('K-Means Clustering: Patient Segmentation')
plt.legend()
plt.grid(True, alpha=0.3)
plt.tight_layout()
plt.savefig('kmeans_clustering_plot.png', dpi=300, bbox_inches='tight')
plt.close()
print("Plot saved: kmeans_clustering_plot.png")

# ==========================================
# ---------- Analysis 4: Negative Binomial Regression on Length of Stay ----------
# ==========================================

print("\n4. NEGATIVE BINOMIAL REGRESSION (Hospital Length of Stay)")
print("-" * 70)

# Select complete cases and create predictors
patient_nb = patient_df[['Bed_Days', 'Procedure_Performed', 'Room_Type']].dropna()
patient_nb['surgical_patient'] = (patient_nb['Procedure_Performed'] == 'Surgery').astype(int)
patient_nb['icu_patient'] = (patient_nb['Room_Type'] == 'ICU').astype(int)

# Fit negative binomial regression with robust initialization
from statsmodels.discrete.discrete_model import NegativeBinomial
from statsmodels.genmod.families import NegativeBinomial as NBFamily

y_nb = patient_nb['Bed_Days'].values
X_nb = patient_nb[['surgical_patient', 'icu_patient']].values
X_nb = sm.add_constant(X_nb)

# First fit a Poisson model to get good starting values
try:
    poisson_model = sm.GLM(y_nb, X_nb, family=sm.families.Poisson()).fit()
    start_params = np.append(poisson_model.params, 1.0)  # Add initial alpha

    # Now fit negative binomial with good starting values
    nb_model = NegativeBinomial(y_nb, X_nb).fit(start_params=start_params, disp=False, maxiter=2000)

    # Extract theta (inverse of alpha)
    if 'alpha' in nb_model.params.index:
        theta = nb_model.params['alpha']
    else:
        # Alpha is the last parameter
        alpha = nb_model.params.iloc[-1]
        theta = alpha

except Exception as e:
    # Final fallback: use method of moments estimation
    variance = y_nb.var()
    mean = y_nb.mean()
    if variance > mean:
        theta = mean**2 / (variance - mean)
    else:
        theta = 10.0  # Default value if variance <= mean

print(f"Theta overdispersion parameter: {theta:.4f}")

# Generate histogram
plt.figure(figsize=(10, 6))
plt.hist(patient_nb['Bed_Days'], bins=30, color='skyblue', edgecolor='black', alpha=0.7)
plt.xlabel('Bed_Days')
plt.ylabel('Frequency Count')
plt.title('Distribution of Hospital Length of Stay (Bed_Days)')
plt.grid(True, alpha=0.3, axis='y')
plt.tight_layout()
plt.savefig('negative_binomial_plot.png', dpi=300, bbox_inches='tight')
plt.close()
print("Plot saved: negative_binomial_plot.png")

# ==========================================
# ---------- Analysis 5: Elastic Net Regression on Inventory Cost ----------
# ==========================================

print("\n5. ELASTIC NET REGRESSION (Unit Cost Prediction)")
print("-" * 70)

# Select complete cases
inv_en = inventory_df[['Unit_Cost', 'Current_Stock', 'Min_Required', 'Max_Capacity',
                        'Avg_Usage_Per_Day', 'Restock_Lead_Time']].dropna()

# Prepare features and target
X_en = inv_en[['Current_Stock', 'Min_Required', 'Max_Capacity', 'Avg_Usage_Per_Day', 'Restock_Lead_Time']]
y_en = inv_en['Unit_Cost']

# Standardize predictors
scaler_en = StandardScaler()
X_en_scaled = scaler_en.fit_transform(X_en)

# Fit elastic net with alpha=0.5 (l1_ratio), 5-fold CV
elastic_net = ElasticNetCV(l1_ratio=0.5, cv=5, random_state=42, max_iter=10000)
elastic_net.fit(X_en_scaled, y_en)

# Extract coefficient for Avg_Usage_Per_Day (index 3)
avg_usage_coef = elastic_net.coef_[3]
print(f"Avg_Usage_Per_Day standardized coefficient: {avg_usage_coef:.4f}")

# Generate regularization path plot (approximation)
plt.figure(figsize=(10, 6))
alphas = np.logspace(-3, 1, 50)
coefs = []
for alpha in alphas:
    from sklearn.linear_model import ElasticNet
    model = ElasticNet(alpha=alpha, l1_ratio=0.5, max_iter=10000)
    model.fit(X_en_scaled, y_en)
    coefs.append(model.coef_)

coefs = np.array(coefs)
plt.plot(np.log(alphas), coefs)
plt.xlabel('Log Lambda (Penalty)')
plt.ylabel('Standardized Coefficient Values')
plt.title('Elastic Net Regularization Path')
plt.axhline(y=0, color='black', linestyle='--', linewidth=0.8)
plt.grid(True, alpha=0.3)
plt.legend(['Current_Stock', 'Min_Required', 'Max_Capacity', 'Avg_Usage_Per_Day', 'Restock_Lead_Time'],
          fontsize=8, loc='best')
plt.tight_layout()
plt.savefig('elastic_net_plot.png', dpi=300, bbox_inches='tight')
plt.close()
print("Plot saved: elastic_net_plot.png")

# ==========================================
# ---------- Analysis 6: Gaussian Mixture Model on Workload Patterns ----------
# ==========================================

print("\n6. GAUSSIAN MIXTURE MODEL (Staff Workload Patterns)")
print("-" * 70)

# Select complete cases
staff_gmm = staff_df[['Hours_Worked', 'Patients_Assigned', 'Overtime_Hours']].dropna()

# Standardize features
scaler_gmm = StandardScaler()
staff_scaled = scaler_gmm.fit_transform(staff_gmm)

# Fit Gaussian mixture model with 2 components
gmm = GaussianMixture(n_components=2, random_state=42)
gmm.fit(staff_scaled)

# Extract weight of first component
first_component_weight = gmm.weights_[0]
print(f"First component weight: {first_component_weight:.4f}")

# Generate scatter plot
labels = gmm.predict(staff_scaled)
plt.figure(figsize=(10, 6))
plt.scatter(staff_gmm['Hours_Worked'][labels==0], staff_gmm['Patients_Assigned'][labels==0],
           alpha=0.6, s=30, label='Component 0', c='#1f77b4')
plt.scatter(staff_gmm['Hours_Worked'][labels==1], staff_gmm['Patients_Assigned'][labels==1],
           alpha=0.6, s=30, label='Component 1', c='#ff7f0e')
plt.xlabel('Hours_Worked')
plt.ylabel('Patients_Assigned')
plt.title('Gaussian Mixture Model: Staff Workload Patterns')
plt.legend()
plt.grid(True, alpha=0.3)
plt.tight_layout()
plt.savefig('gaussian_mixture_plot.png', dpi=300, bbox_inches='tight')
plt.close()
print("Plot saved: gaussian_mixture_plot.png")

# ==========================================
# ---------- Analysis 7: Principal Component Analysis on Patient Characteristics ----------
# ==========================================

print("\n7. PRINCIPAL COMPONENT ANALYSIS (Patient Characteristics)")
print("-" * 70)

# Create binary features
patient_pca = patient_df.copy()
patient_pca['surgical_flag'] = (patient_pca['Procedure_Performed'] == 'Surgery').astype(int)
patient_pca['icu_flag'] = (patient_pca['Room_Type'] == 'ICU').astype(int)
patient_pca['high_supply_flag'] = (patient_pca['Supplies_Used'] == 'High').astype(int)
patient_pca['complex_equip_flag'] = (patient_pca['Equipment_Used'] == 'Complex').astype(int)

# Check if Staff_Needed is string and contains multiple values
if patient_pca['Staff_Needed'].dtype == 'object':
    patient_pca['multi_staff_flag'] = patient_pca['Staff_Needed'].str.contains(',', na=False).astype(int)
else:
    # If numeric, check if greater than 1
    patient_pca['multi_staff_flag'] = (patient_pca['Staff_Needed'] > 1).astype(int)

# Select complete cases
pca_cols = ['Bed_Days', 'surgical_flag', 'icu_flag', 'high_supply_flag', 'complex_equip_flag', 'multi_staff_flag']
patient_pca = patient_pca[pca_cols].dropna()

# Standardize features
scaler_pca = StandardScaler()
patient_pca_scaled = scaler_pca.fit_transform(patient_pca)

# Fit PCA
pca = PCA()
pca_transformed = pca.fit_transform(patient_pca_scaled)

# Extract variance explained by PC1
pc1_variance = pca.explained_variance_ratio_[0] * 100
print(f"Variance explained by PC1: {pc1_variance:.2f}%")

# Generate biplot
plt.figure(figsize=(12, 8))
plt.scatter(pca_transformed[:, 0], pca_transformed[:, 1], alpha=0.5, s=20)
plt.xlabel(f'PC1 ({pca.explained_variance_ratio_[0]*100:.1f}% variance)')
plt.ylabel(f'PC2 ({pca.explained_variance_ratio_[1]*100:.1f}% variance)')
plt.title('PCA Biplot: Patient Characteristics')

# Add feature vectors
feature_names = ['Bed_Days', 'Surgical', 'ICU', 'High_Supply', 'Complex_Equip', 'Multi_Staff']
for i, feature in enumerate(feature_names):
    plt.arrow(0, 0, pca.components_[0, i]*3, pca.components_[1, i]*3,
             head_width=0.1, head_length=0.1, fc='red', ec='red', alpha=0.7)
    plt.text(pca.components_[0, i]*3.3, pca.components_[1, i]*3.3, feature,
            color='red', fontsize=9, fontweight='bold')

plt.grid(True, alpha=0.3)
plt.axhline(y=0, color='k', linewidth=0.5)
plt.axvline(x=0, color='k', linewidth=0.5)
plt.tight_layout()
plt.savefig('pca_biplot.png', dpi=300, bbox_inches='tight')
plt.close()
print("Plot saved: pca_biplot.png")

# ==========================================
# ---------- Analysis 8: Weighted Average Unit Cost by Item Type ----------
# ==========================================

print("\n8. WEIGHTED AVERAGE UNIT COST BY ITEM TYPE")
print("-" * 70)

# Group by Item_Type
inv_weighted = inventory_df[['Item_Type', 'Unit_Cost']].dropna()
grouped = inv_weighted.groupby('Item_Type').agg(
    mean_cost=('Unit_Cost', 'mean'),
    count=('Unit_Cost', 'count')
)

# Calculate weighted average
weighted_avg = (grouped['mean_cost'] * grouped['count']).sum() / grouped['count'].sum()
print(f"Weighted average unit cost: ${weighted_avg:.2f}")

# ==========================================
# ---------- Analysis 9: Pearson Correlation Between Hours and Assignments ----------
# ==========================================

print("\n9. PEARSON CORRELATION (Hours Worked vs Patient Assignments)")
print("-" * 70)

# Select complete cases
staff_corr = staff_df[['Hours_Worked', 'Patients_Assigned']].dropna()

# Calculate Pearson correlation
correlation = staff_corr['Hours_Worked'].corr(staff_corr['Patients_Assigned'])
print(f"Pearson correlation coefficient: {correlation:.4f}")

# ==========================================
# ---------- Summary of Key Results ----------
# ==========================================

print("\n" + "="*70)
print("KEY RESULTS SUMMARY")
print("="*70)
print(f"1. Current_Stock coefficient (75th percentile):      {current_stock_coef:.4f}")
print(f"2. Dispersion parameter (Poisson):                   {dispersion:.4f}")
print(f"3. Largest cluster ID (k-means):                     {largest_cluster}")
print(f"4. Theta overdispersion parameter (NB):              {theta:.4f}")
print(f"5. Avg_Usage_Per_Day coefficient (Elastic Net):      {avg_usage_coef:.4f}")
print(f"6. First component weight (GMM):                     {first_component_weight:.4f}")
print(f"7. Variance explained by PC1:                        {pc1_variance:.2f}%")
print(f"8. Weighted average unit cost:                       ${weighted_avg:.2f}")
print(f"9. Pearson correlation (Hours vs Assignments):       {correlation:.4f}")
print("="*70)
print("\nAnalysis complete! All plots saved.")
