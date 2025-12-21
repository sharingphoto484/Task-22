# ==========================================
# College Basketball Tournament Prediction Analysis
# ==========================================
# Requirements: pandas, numpy, matplotlib, scipy, scikit-learn, openpyxl
# Input files: cbb13.xlsx, cbb14.xlsx, cbb15.xlsx
# Output files: Various visualization plots
#
# This script performs comprehensive machine learning analysis on college basketball
# data including stepwise regression, SVM classification, DBSCAN clustering,
# multinomial logistic regression, polynomial features, Huber regression,
# AdaBoost classification, and statistical analyses.
# ==========================================

import pandas as pd
import numpy as np
import matplotlib.pyplot as plt
from sklearn.preprocessing import StandardScaler, PolynomialFeatures
from sklearn.model_selection import train_test_split
from sklearn.svm import SVC
from sklearn.cluster import DBSCAN
from sklearn.linear_model import LinearRegression, HuberRegressor, LogisticRegression
from sklearn.ensemble import AdaBoostClassifier
from sklearn.metrics import accuracy_score, f1_score
from scipy.stats import spearmanr
import warnings
warnings.filterwarnings('ignore')

# ---------- Load Excel Files ----------
print("Loading datasets...")
cbb13 = pd.read_excel('cbb13.xlsx')
cbb14 = pd.read_excel('cbb14.xlsx')
cbb15 = pd.read_excel('cbb15.xlsx')

# ==========================================
# TASK 1: Stepwise Forward Selection for BARTHAG (cbb15)
# ==========================================
print("\n" + "="*60)
print("TASK 1: Stepwise Forward Selection for BARTHAG")
print("="*60)

# ---------- Filter teams with non-null BARTHAG ----------
df_task1 = cbb15[cbb15['BARTHAG'].notna()].copy()
y_barthag = df_task1['BARTHAG']

# Candidate predictor pool
candidate_features = ['ADJOE', 'ADJDE', 'EFG_O', 'EFG_D', 'ORB', 'DRB', '3P_O', '3P_D']
X_candidates = df_task1[candidate_features].copy()

# ---------- Stepwise Forward Selection ----------
def calculate_aic(n, mse, num_params):
    """Calculate Akaike Information Criterion"""
    return n * np.log(mse) + 2 * num_params

selected_features = []
remaining_features = candidate_features.copy()
current_aic = np.inf

# Initialize with intercept only
n = len(y_barthag)
baseline_mse = np.mean((y_barthag - np.mean(y_barthag))**2)
current_aic = calculate_aic(n, baseline_mse, 1)

# Iterative forward selection
while remaining_features:
    aic_values = {}

    for feature in remaining_features:
        test_features = selected_features + [feature]
        X_test = X_candidates[test_features]

        # Fit linear regression
        lr = LinearRegression()
        lr.fit(X_test, y_barthag)
        predictions = lr.predict(X_test)
        mse = np.mean((y_barthag - predictions)**2)

        # Calculate AIC
        num_params = len(test_features) + 1  # features + intercept
        aic = calculate_aic(n, mse, num_params)
        aic_values[feature] = aic

    # Find feature with minimum AIC
    best_feature = min(aic_values, key=aic_values.get)
    best_aic = aic_values[best_feature]

    # Check if AIC improves by at least 2 units
    if current_aic - best_aic >= 2:
        selected_features.append(best_feature)
        remaining_features.remove(best_feature)
        current_aic = best_aic
    else:
        break

num_selected_features = len(selected_features)
print(f"\nSelected Features: {selected_features}")
print(f"Number of Selected Features: {num_selected_features}")

# ---------- Generate Scatter Plot Matrix ----------
if num_selected_features > 0:
    n_features = len(selected_features)
    fig, axes = plt.subplots(1, n_features, figsize=(5*n_features, 4))
    if n_features == 1:
        axes = [axes]

    for idx, feature in enumerate(selected_features):
        axes[idx].scatter(df_task1[feature], df_task1['BARTHAG'], alpha=0.6, s=20)
        axes[idx].set_xlabel(feature, fontsize=10)
        axes[idx].set_ylabel('BARTHAG', fontsize=10)
        axes[idx].set_title(f'{feature} vs BARTHAG', fontsize=10)
        axes[idx].grid(True, alpha=0.3)

    plt.tight_layout()
    plt.savefig('stepwise_selection_scatter_matrix.png', dpi=100, bbox_inches='tight')
    plt.close()

# ==========================================
# TASK 2: SVM with RBF Kernel for Tournament Qualification
# ==========================================
print("\n" + "="*60)
print("TASK 2: SVM Classification for Tournament Qualification")
print("="*60)

# ---------- Combine datasets vertically ----------
combined_data = pd.concat([cbb13, cbb14, cbb15], ignore_index=True)

# Create binary target variable
combined_data['tournament_qualified'] = combined_data['POSTSEASON'].notna().astype(int)

# Select predictor variables
svm_features = ['ADJOE', 'ADJDE', 'WAB', 'BARTHAG']
df_task2 = combined_data[svm_features + ['tournament_qualified']].dropna()

X_svm = df_task2[svm_features]
y_svm = df_task2['tournament_qualified']

# ---------- Standardize features ----------
scaler_svm = StandardScaler()
X_svm_scaled = scaler_svm.fit_transform(X_svm)

# ---------- Train-test split ----------
X_train_svm, X_test_svm, y_train_svm, y_test_svm = train_test_split(
    X_svm_scaled, y_svm, test_size=0.25, random_state=42
)

# ---------- Fit SVM with RBF kernel ----------
svm_classifier = SVC(kernel='rbf', gamma=0.1, C=10, random_state=42)
svm_classifier.fit(X_train_svm, y_train_svm)

# Generate predictions
y_pred_svm = svm_classifier.predict(X_test_svm)
svm_accuracy = accuracy_score(y_test_svm, y_pred_svm)
svm_accuracy_pct = round(svm_accuracy * 100, 2)

print(f"\nSVM Classification Accuracy: {svm_accuracy_pct}%")

# ---------- Decision boundary visualization ----------
# Use first two features (ADJOE and ADJDE) for visualization
X_vis = df_task2[['ADJOE', 'ADJDE']].values
y_vis = df_task2['tournament_qualified'].values

scaler_vis = StandardScaler()
X_vis_scaled = scaler_vis.fit_transform(X_vis)

svm_vis = SVC(kernel='rbf', gamma=0.1, C=10, random_state=42)
svm_vis.fit(X_vis_scaled, y_vis)

# Create mesh
x_min, x_max = X_vis_scaled[:, 0].min() - 0.5, X_vis_scaled[:, 0].max() + 0.5
y_min, y_max = X_vis_scaled[:, 1].min() - 0.5, X_vis_scaled[:, 1].max() + 0.5
xx, yy = np.meshgrid(np.linspace(x_min, x_max, 200), np.linspace(y_min, y_max, 200))

Z = svm_vis.predict(np.c_[xx.ravel(), yy.ravel()])
Z = Z.reshape(xx.shape)

plt.figure(figsize=(10, 8))
plt.contourf(xx, yy, Z, alpha=0.4, cmap='RdYlBu')
scatter = plt.scatter(X_vis_scaled[:, 0], X_vis_scaled[:, 1], c=y_vis,
                     cmap='RdYlBu', edgecolors='k', s=40, alpha=0.7)
plt.xlabel('ADJOE (standardized)', fontsize=12)
plt.ylabel('ADJDE (standardized)', fontsize=12)
plt.title('SVM Decision Boundary for Tournament Qualification', fontsize=14)
plt.colorbar(scatter, label='Tournament Qualified')
plt.grid(True, alpha=0.3)
plt.savefig('svm_decision_boundary.png', dpi=100, bbox_inches='tight')
plt.close()

# ==========================================
# TASK 3: DBSCAN Clustering (cbb14)
# ==========================================
print("\n" + "="*60)
print("TASK 3: DBSCAN Clustering for Performance Tiers")
print("="*60)

# ---------- Select teams with complete ADJOE and ADJDE ----------
df_task3 = cbb14[['ADJOE', 'ADJDE']].dropna()

# ---------- Standardize features ----------
scaler_dbscan = StandardScaler()
X_dbscan = scaler_dbscan.fit_transform(df_task3)

# ---------- Apply DBSCAN ----------
dbscan = DBSCAN(eps=0.5, min_samples=5)
clusters = dbscan.fit_predict(X_dbscan)

# Count noise points (cluster label = -1)
noise_count = np.sum(clusters == -1)
print(f"\nNumber of Noise Points: {noise_count}")

# ---------- Scatter visualization ----------
plt.figure(figsize=(10, 8))
unique_clusters = np.unique(clusters)
colors = plt.cm.rainbow(np.linspace(0, 1, len(unique_clusters)))

for cluster_id, color in zip(unique_clusters, colors):
    mask = clusters == cluster_id
    if cluster_id == -1:
        # Noise points
        plt.scatter(X_dbscan[mask, 0], X_dbscan[mask, 1],
                   c='black', marker='x', s=100, label='Noise', alpha=0.6)
    else:
        plt.scatter(X_dbscan[mask, 0], X_dbscan[mask, 1],
                   c=[color], s=60, label=f'Cluster {cluster_id}', alpha=0.7)

plt.xlabel('Standardized ADJOE', fontsize=12)
plt.ylabel('Standardized ADJDE', fontsize=12)
plt.title('DBSCAN Clustering of Team Performance', fontsize=14)
plt.legend()
plt.grid(True, alpha=0.3)
plt.savefig('dbscan_clustering.png', dpi=100, bbox_inches='tight')
plt.close()

# ==========================================
# TASK 4: Multinomial Logistic Regression for Conference Tiers
# ==========================================
print("\n" + "="*60)
print("TASK 4: Multinomial Logistic Regression for Conference Tiers")
print("="*60)

# ---------- Combine all datasets ----------
combined_full = pd.concat([cbb13, cbb14, cbb15], ignore_index=True)

# ---------- Create conference tier categorical target ----------
power_conferences = ['ACC', 'B10', 'B12', 'SEC', 'P12']
mid_major_conferences = ['A10', 'Amer', 'BE', 'MVC', 'WCC']

def assign_tier(conf):
    if conf in power_conferences:
        return 'high'
    elif conf in mid_major_conferences:
        return 'mid'
    else:
        return 'low'

combined_full['conference_tier'] = combined_full['CONF'].apply(assign_tier)

# Select predictor features
mlr_features = ['W', 'ADJOE', 'ADJDE', 'EFG_O', 'TOR']
df_task4 = combined_full[mlr_features + ['conference_tier']].dropna()

X_mlr = df_task4[mlr_features]
y_mlr = df_task4['conference_tier']

# ---------- Train-test split ----------
X_train_mlr, X_test_mlr, y_train_mlr, y_test_mlr = train_test_split(
    X_mlr, y_mlr, test_size=0.20, random_state=42
)

# ---------- Fit multinomial logistic regression ----------
mlr_model = LogisticRegression(solver='lbfgs', max_iter=1000, random_state=42)
mlr_model.fit(X_train_mlr, y_train_mlr)

# Predict and calculate macro F1 score
y_pred_mlr = mlr_model.predict(X_test_mlr)
macro_f1 = f1_score(y_test_mlr, y_pred_mlr, average='macro')
macro_f1_rounded = round(macro_f1, 4)

print(f"\nMacro-averaged F1 Score: {macro_f1_rounded}")

# ---------- Stacked bar chart ----------
tier_order = ['low', 'mid', 'high']
confusion_data = pd.DataFrame({
    'True': y_test_mlr.values,
    'Predicted': y_pred_mlr
})

# Calculate proportions
proportions = []
for true_tier in tier_order:
    tier_data = confusion_data[confusion_data['True'] == true_tier]
    counts = tier_data['Predicted'].value_counts()
    total = len(tier_data)
    props = {tier: counts.get(tier, 0) / total for tier in tier_order}
    proportions.append(props)

# Create stacked bar chart
fig, ax = plt.subplots(figsize=(10, 6))
x_pos = np.arange(len(tier_order))
bottom = np.zeros(len(tier_order))

colors = ['#d62728', '#ff7f0e', '#2ca02c']
for idx, pred_tier in enumerate(tier_order):
    values = [proportions[i][pred_tier] for i in range(len(tier_order))]
    ax.bar(x_pos, values, bottom=bottom, label=f'Predicted {pred_tier}', color=colors[idx])
    bottom += values

ax.set_xlabel('True Conference Tier', fontsize=12)
ax.set_ylabel('Proportion', fontsize=12)
ax.set_title('Conference Tier Prediction Distribution', fontsize=14)
ax.set_xticks(x_pos)
ax.set_xticklabels(tier_order)
ax.legend()
ax.grid(True, alpha=0.3, axis='y')
plt.savefig('conference_tier_stacked_bar.png', dpi=100, bbox_inches='tight')
plt.close()

# ==========================================
# TASK 5: Polynomial Features for Non-linear Relationships (cbb15)
# ==========================================
print("\n" + "="*60)
print("TASK 5: Polynomial Features for Offensive-Defensive Relationships")
print("="*60)

# ---------- Extract features ----------
df_task5 = cbb15[['ADJOE', 'ADJDE', 'W']].dropna()
X_poly_base = df_task5[['ADJOE', 'ADJDE']].values
y_wins = df_task5['W'].values

# ---------- Generate polynomial features of degree 2 ----------
poly = PolynomialFeatures(degree=2, include_bias=False)
X_poly = poly.fit_transform(X_poly_base)

# Fit linear regression
lr_poly = LinearRegression()
lr_poly.fit(X_poly, y_wins)

# Extract coefficient for interaction term (ADJOE * ADJDE)
# Feature order: [ADJOE, ADJDE, ADJOE^2, ADJOE*ADJDE, ADJDE^2]
interaction_coef = lr_poly.coef_[3]
interaction_coef_rounded = round(interaction_coef, 4)

print(f"\nInteraction Coefficient (ADJOE * ADJDE): {interaction_coef_rounded}")

# ---------- 3D surface plot ----------
from mpl_toolkits.mplot3d import Axes3D

adjoe_range = np.linspace(df_task5['ADJOE'].min(), df_task5['ADJOE'].max(), 50)
adjde_range = np.linspace(df_task5['ADJDE'].min(), df_task5['ADJDE'].max(), 50)
adjoe_mesh, adjde_mesh = np.meshgrid(adjoe_range, adjde_range)

X_surface = np.c_[adjoe_mesh.ravel(), adjde_mesh.ravel()]
X_surface_poly = poly.transform(X_surface)
wins_pred = lr_poly.predict(X_surface_poly).reshape(adjoe_mesh.shape)

fig = plt.figure(figsize=(12, 8))
ax = fig.add_subplot(111, projection='3d')
surf = ax.plot_surface(adjoe_mesh, adjde_mesh, wins_pred, cmap='viridis', alpha=0.7)
ax.scatter(df_task5['ADJOE'], df_task5['ADJDE'], df_task5['W'],
          c='red', marker='o', s=20, alpha=0.5)
ax.set_xlabel('ADJOE', fontsize=11)
ax.set_ylabel('ADJDE', fontsize=11)
ax.set_zlabel('Predicted Wins', fontsize=11)
ax.set_title('Polynomial Regression: Wins vs ADJOE & ADJDE', fontsize=13)
fig.colorbar(surf, shrink=0.5)
plt.savefig('polynomial_surface_plot.png', dpi=100, bbox_inches='tight')
plt.close()

# ==========================================
# TASK 6: Huber Regression for Outlier-Robust Win Prediction
# ==========================================
print("\n" + "="*60)
print("TASK 6: Huber Regression for Win Percentage Prediction")
print("="*60)

# ---------- Combine datasets ----------
combined_huber = pd.concat([cbb13, cbb14, cbb15], ignore_index=True)

# Create win percentage target
combined_huber['win_percentage'] = combined_huber['W'] / combined_huber['G']

# Select predictors
huber_features = ['BARTHAG', 'EFG_O', 'EFG_D', 'ORB', 'DRB']
df_task6 = combined_huber[huber_features + ['win_percentage']].dropna()

X_huber = df_task6[huber_features]
y_huber = df_task6['win_percentage']

# ---------- Fit Huber regressor ----------
huber_reg = HuberRegressor(epsilon=1.35, max_iter=200)
huber_reg.fit(X_huber, y_huber)

# Calculate predictions and MAE
y_pred_huber = huber_reg.predict(X_huber)
mae_huber = np.mean(np.abs(y_huber - y_pred_huber))
mae_huber_rounded = round(mae_huber, 4)

print(f"\nMean Absolute Error: {mae_huber_rounded}")

# ---------- Residual plot ----------
residuals = y_huber - y_pred_huber

plt.figure(figsize=(10, 6))
plt.scatter(y_pred_huber, residuals, alpha=0.5, s=30)
plt.axhline(y=0, color='red', linestyle='--', linewidth=2)
plt.xlabel('Predicted Win Percentage', fontsize=12)
plt.ylabel('Residuals', fontsize=12)
plt.title('Huber Regression Residual Plot', fontsize=14)
plt.grid(True, alpha=0.3)
plt.savefig('huber_residual_plot.png', dpi=100, bbox_inches='tight')
plt.close()

# ==========================================
# TASK 7: AdaBoost for Postseason Round Prediction
# ==========================================
print("\n" + "="*60)
print("TASK 7: AdaBoost Classification for Postseason Round Advancement")
print("="*60)

# ---------- Combine datasets ----------
combined_ada = pd.concat([cbb13, cbb14, cbb15], ignore_index=True)

# ---------- Create multi-class target ----------
postseason_mapping = {
    None: 0,
    'R68': 1,
    'R64': 2,
    'R32': 3,
    'S16': 4,
    'E8': 5,
    'F4': 6,
    '2ND': 7,
    'Champions': 8
}

combined_ada['postseason_round'] = combined_ada['POSTSEASON'].map(lambda x: postseason_mapping.get(x, 0))

# Select features
ada_features = ['ADJOE', 'ADJDE', 'BARTHAG', 'WAB', 'EFG_O', 'EFG_D']
df_task7 = combined_ada[ada_features + ['postseason_round']].dropna()

X_ada = df_task7[ada_features]
y_ada = df_task7['postseason_round']

# ---------- Train-test split ----------
X_train_ada, X_test_ada, y_train_ada, y_test_ada = train_test_split(
    X_ada, y_ada, test_size=0.20, random_state=42
)

# ---------- Fit AdaBoost classifier ----------
ada_classifier = AdaBoostClassifier(n_estimators=50, learning_rate=1.0, random_state=42)
ada_classifier.fit(X_train_ada, y_train_ada)

# Predict and calculate weighted accuracy
y_pred_ada = ada_classifier.predict(X_test_ada)

# Calculate inverse class frequency weights
from sklearn.utils.class_weight import compute_class_weight
class_weights = compute_class_weight('balanced', classes=np.unique(y_test_ada), y=y_test_ada)
weight_dict = dict(zip(np.unique(y_test_ada), class_weights))

# Calculate weighted accuracy
sample_weights = np.array([weight_dict[y] for y in y_test_ada])
weighted_acc = accuracy_score(y_test_ada, y_pred_ada, sample_weight=sample_weights)
weighted_acc_pct = round(weighted_acc * 100, 2)

print(f"\nWeighted Accuracy: {weighted_acc_pct}%")

# ---------- Feature importance bar chart ----------
feature_importance = ada_classifier.feature_importances_

plt.figure(figsize=(10, 6))
plt.bar(ada_features, feature_importance, color='steelblue', alpha=0.8)
plt.xlabel('Features', fontsize=12)
plt.ylabel('Relative Importance', fontsize=12)
plt.title('AdaBoost Feature Importance for Postseason Round Prediction', fontsize=14)
plt.xticks(rotation=45, ha='right')
plt.grid(True, alpha=0.3, axis='y')
plt.tight_layout()
plt.savefig('adaboost_feature_importance.png', dpi=100, bbox_inches='tight')
plt.close()

# ==========================================
# TASK 8: Spearman Rank Correlation (cbb13)
# ==========================================
print("\n" + "="*60)
print("TASK 8: Spearman Correlation Between Tempo and Tournament Success")
print("="*60)

# ---------- Filter tournament teams ----------
df_task8 = cbb13[cbb13['SEED'].notna()][['ADJ_T', 'SEED']].dropna()

tempo = df_task8['ADJ_T']
seed = df_task8['SEED']

# ---------- Compute Spearman correlation ----------
spearman_corr, _ = spearmanr(tempo, seed)
spearman_corr_rounded = round(spearman_corr, 4)

print(f"\nSpearman Rank Correlation (Tempo vs Seed): {spearman_corr_rounded}")

# ==========================================
# TASK 9: Weighted Variance of Conference Win Rates (cbb14)
# ==========================================
print("\n" + "="*60)
print("TASK 9: Weighted Variance of Conference Win Rates")
print("="*60)

# ---------- Calculate conference metrics ----------
df_task9 = cbb14[['CONF', 'W', 'G']].copy()
df_task9['win_rate'] = df_task9['W'] / df_task9['G']

conf_stats = df_task9.groupby('CONF').agg({
    'win_rate': 'mean',
    'CONF': 'size'
}).rename(columns={'CONF': 'conference_size'})

# Calculate grand mean
grand_mean = (conf_stats['win_rate'] * conf_stats['conference_size']).sum() / conf_stats['conference_size'].sum()

# ---------- Compute weighted variance ----------
conf_stats['squared_dev'] = (conf_stats['win_rate'] - grand_mean) ** 2
weighted_variance = (conf_stats['squared_dev'] * conf_stats['conference_size']).sum() / conf_stats['conference_size'].sum()
weighted_variance_rounded = round(weighted_variance, 4)

print(f"\nWeighted Variance of Conference Win Rates: {weighted_variance_rounded}")

# ==========================================
# TASK 10: Efficiency Differential Ratio
# ==========================================
print("\n" + "="*60)
print("TASK 10: Efficiency Differential Ratio (Tournament vs Non-Tournament)")
print("="*60)

# ---------- Combine datasets ----------
combined_eff = pd.concat([cbb13, cbb14, cbb15], ignore_index=True)

# Create tournament team binary
combined_eff['tournament_team'] = combined_eff['POSTSEASON'].notna().astype(int)

# Calculate efficiency differential
combined_eff['efficiency_differential'] = combined_eff['ADJOE'] - combined_eff['ADJDE']

df_task10 = combined_eff[['tournament_team', 'efficiency_differential']].dropna()

# ---------- Calculate means ----------
tournament_mean = df_task10[df_task10['tournament_team'] == 1]['efficiency_differential'].mean()
non_tournament_mean = df_task10[df_task10['tournament_team'] == 0]['efficiency_differential'].mean()

# Calculate ratio
eff_ratio = tournament_mean / non_tournament_mean
eff_ratio_rounded = round(eff_ratio, 2)

print(f"\nTournament Team Mean Efficiency Differential: {round(tournament_mean, 2)}")
print(f"Non-Tournament Team Mean Efficiency Differential: {round(non_tournament_mean, 2)}")
print(f"Efficiency Differential Ratio: {eff_ratio_rounded}")

# ==========================================
# TASK 11: Conference with Maximum 3P_D Standard Deviation
# ==========================================
print("\n" + "="*60)
print("TASK 11: Conference with Maximum 3P_D Standard Deviation")
print("="*60)

# ---------- Combine datasets ----------
combined_3pd = pd.concat([cbb13, cbb14, cbb15], ignore_index=True)

# ---------- Calculate standard deviation by conference ----------
conf_3pd_std = combined_3pd.groupby('CONF')['3P_D'].std()

# Find conference with maximum standard deviation
max_std_conf = conf_3pd_std.idxmax()
max_std_value = conf_3pd_std.max()

print(f"\nConference with Maximum 3P_D Standard Deviation: {max_std_conf}")
print(f"Standard Deviation Value: {round(max_std_value, 4)}")

# ==========================================
# SUMMARY OF KEY OUTPUTS
# ==========================================
print("\n" + "="*60)
print("SUMMARY OF KEY OUTPUTS")
print("="*60)
print(f"1. Number of Selected Features (Stepwise): {num_selected_features}")
print(f"2. SVM Classification Accuracy: {svm_accuracy_pct}%")
print(f"3. DBSCAN Noise Points Count: {noise_count}")
print(f"4. Multinomial Logistic Regression Macro F1: {macro_f1_rounded}")
print(f"5. Polynomial Interaction Coefficient: {interaction_coef_rounded}")
print(f"6. Huber Regression MAE: {mae_huber_rounded}")
print(f"7. AdaBoost Weighted Accuracy: {weighted_acc_pct}%")
print(f"8. Spearman Correlation (Tempo vs Seed): {spearman_corr_rounded}")
print(f"9. Weighted Variance of Conference Win Rates: {weighted_variance_rounded}")
print(f"10. Efficiency Differential Ratio: {eff_ratio_rounded}")
print(f"11. Conference with Max 3P_D Std Dev: {max_std_conf}")
print("="*60)

print("\n✓ Analysis complete! All visualizations saved.")
