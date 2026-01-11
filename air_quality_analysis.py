# ==========================================
# Urban Air Quality Analysis Script
# Requirements: pandas, numpy, matplotlib, scikit-learn, xgboost, seaborn
# Input files: Sydney_Air_Quality.xlsx, New_York_Air_Quality.xlsx, London_Air_Quality.xlsx
# Output files: feature_importance.png, confusion_matrix.png, pls_scatter.png,
#               residual_scatter.png, roc_curve.png
# ==========================================

import pandas as pd
import numpy as np
import matplotlib.pyplot as plt
import seaborn as sns
from sklearn.model_selection import train_test_split
from sklearn.ensemble import RandomForestRegressor, GradientBoostingClassifier, AdaBoostClassifier
from sklearn.cross_decomposition import PLSRegression
from sklearn.linear_model import Ridge, Lasso, ElasticNetCV
from sklearn.svm import SVR
from sklearn.cluster import KMeans
from sklearn.decomposition import PCA
from sklearn.preprocessing import StandardScaler
from sklearn.metrics import mean_squared_error, mean_absolute_error, r2_score, accuracy_score, f1_score, roc_curve, auc, silhouette_score, confusion_matrix
from xgboost import XGBRegressor
import warnings
warnings.filterwarnings('ignore')

# ---------- Load Excel Files ----------
print("Loading air quality datasets...")
sydney_df = pd.read_excel('Sydney_Air_Quality.xlsx')
newyork_df = pd.read_excel('New_York_Air_Quality.xlsx')
london_df = pd.read_excel('London_Air_Quality.xlsx')

print(f"Sydney: {sydney_df.shape}, New York: {newyork_df.shape}, London: {london_df.shape}")

# ==========================================
# 1. Sydney - Random Forest Regression (AQI Prediction)
# ==========================================
print("\n" + "="*50)
print("1. Sydney - Random Forest Regression")
print("="*50)

# ---------- Prepare Sydney Data for RF ----------
sydney_rf = sydney_df[['CO', 'NO2', 'SO2', 'O3', 'PM2.5', 'PM10', 'AQI']].dropna()
X_sydney_rf = sydney_rf[['CO', 'NO2', 'SO2', 'O3', 'PM2.5', 'PM10']]
y_sydney_rf = sydney_rf['AQI']

# ---------- Train-Test Split 75-25 ----------
X_train_rf, X_test_rf, y_train_rf, y_test_rf = train_test_split(
    X_sydney_rf, y_sydney_rf, test_size=0.25, random_state=42
)

# ---------- Train Random Forest ----------
rf_model = RandomForestRegressor(
    n_estimators=100,
    max_depth=10,
    min_samples_split=5,
    max_features='sqrt',
    random_state=42
)
rf_model.fit(X_train_rf, y_train_rf)

# ---------- Evaluate and Output RMSE ----------
y_pred_rf = rf_model.predict(X_test_rf)
rmse_rf = np.sqrt(mean_squared_error(y_test_rf, y_pred_rf))
print(f"RMSE: {rmse_rf:.2f}")

# ---------- Plot Feature Importance ----------
plt.figure(figsize=(8, 5))
feature_importance = pd.DataFrame({
    'feature': X_sydney_rf.columns,
    'importance': rf_model.feature_importances_
}).sort_values('importance')

plt.barh(feature_importance['feature'], feature_importance['importance'])
plt.xlabel('Importance')
plt.ylabel('Feature')
plt.title('Sydney AQI - Random Forest Feature Importance')
plt.tight_layout()
plt.savefig('feature_importance.png', dpi=300, bbox_inches='tight')
plt.close()
print("Saved: feature_importance.png")

# ==========================================
# 2. New York - Gradient Boosting Classification
# ==========================================
print("\n" + "="*50)
print("2. New York - Gradient Boosting Classification")
print("="*50)

# ---------- Create Categorical Target ----------
ny_gb = newyork_df[['CO', 'NO2', 'PM2.5', 'PM10', 'AQI']].dropna()
ny_gb['AQI_Category'] = pd.cut(
    ny_gb['AQI'],
    bins=[0, 35, 55, float('inf')],
    labels=['good', 'moderate', 'poor'],
    include_lowest=True
)

X_ny_gb = ny_gb[['CO', 'NO2', 'PM2.5', 'PM10']]
y_ny_gb = ny_gb['AQI_Category']

# ---------- Train-Test Split 70-30 ----------
X_train_gb, X_test_gb, y_train_gb, y_test_gb = train_test_split(
    X_ny_gb, y_ny_gb, test_size=0.30, random_state=42
)

# ---------- Train Gradient Boosting Classifier ----------
gb_model = GradientBoostingClassifier(
    n_estimators=100,
    learning_rate=0.1,
    max_depth=5,
    subsample=0.8,
    random_state=42
)
gb_model.fit(X_train_gb, y_train_gb)

# ---------- Evaluate and Output Accuracy ----------
y_pred_gb = gb_model.predict(X_test_gb)
accuracy_gb = accuracy_score(y_test_gb, y_pred_gb) * 100
print(f"Accuracy: {accuracy_gb:.2f}%")

# ---------- Plot Confusion Matrix ----------
cm = confusion_matrix(y_test_gb, y_pred_gb, labels=['good', 'moderate', 'poor'])
plt.figure(figsize=(7, 6))
sns.heatmap(cm, annot=True, fmt='d', cmap='Blues',
            xticklabels=['good', 'moderate', 'poor'],
            yticklabels=['good', 'moderate', 'poor'])
plt.xlabel('Predicted')
plt.ylabel('True')
plt.title('New York AQI Classification - Confusion Matrix')
plt.tight_layout()
plt.savefig('confusion_matrix.png', dpi=300, bbox_inches='tight')
plt.close()
print("Saved: confusion_matrix.png")

# ==========================================
# 3. London - PLS Regression (PM10 Prediction)
# ==========================================
print("\n" + "="*50)
print("3. London - PLS Regression")
print("="*50)

# ---------- Prepare London Data for PLS ----------
london_pls = london_df[['CO', 'NO2', 'SO2', 'O3', 'PM2.5', 'PM10']].dropna()
X_london_pls = london_pls[['CO', 'NO2', 'SO2', 'O3', 'PM2.5']]
y_london_pls = london_pls['PM10']

# ---------- Train PLS Regression ----------
pls_model = PLSRegression(n_components=3)
pls_model.fit(X_london_pls, y_london_pls)

# ---------- Evaluate and Output R-squared ----------
y_pred_pls = pls_model.predict(X_london_pls)
r2_pls = r2_score(y_london_pls, y_pred_pls)
print(f"R-squared: {r2_pls:.4f}")

# ---------- Plot Observed vs Predicted ----------
plt.figure(figsize=(7, 7))
plt.scatter(y_london_pls, y_pred_pls, alpha=0.5, s=10)
min_val = min(y_london_pls.min(), y_pred_pls.min())
max_val = max(y_london_pls.max(), y_pred_pls.max())
plt.plot([min_val, max_val], [min_val, max_val], 'r--', lw=2, label='45° line')
plt.xlabel('Observed PM10')
plt.ylabel('Predicted PM10')
plt.title('London PM10 - PLS Regression')
plt.legend()
plt.tight_layout()
plt.savefig('pls_scatter.png', dpi=300, bbox_inches='tight')
plt.close()
print("Saved: pls_scatter.png")

# ==========================================
# 4. Sydney - XGBoost Regression (CO Prediction)
# ==========================================
print("\n" + "="*50)
print("4. Sydney - XGBoost Regression")
print("="*50)

# ---------- Prepare Sydney Data for XGBoost ----------
sydney_xgb = sydney_df[['NO2', 'SO2', 'O3', 'PM2.5', 'PM10', 'AQI', 'CO']].dropna()
X_sydney_xgb = sydney_xgb[['NO2', 'SO2', 'O3', 'PM2.5', 'PM10', 'AQI']]
y_sydney_xgb = sydney_xgb['CO']

# ---------- Train-Test Split 80-20 ----------
X_train_xgb, X_test_xgb, y_train_xgb, y_test_xgb = train_test_split(
    X_sydney_xgb, y_sydney_xgb, test_size=0.20, random_state=42
)

# ---------- Train XGBoost Regressor ----------
xgb_model = XGBRegressor(
    n_estimators=100,
    max_depth=6,
    learning_rate=0.1,
    subsample=0.8,
    colsample_bytree=0.8,
    random_state=42
)
xgb_model.fit(X_train_xgb, y_train_xgb)

# ---------- Evaluate and Output MAE ----------
y_pred_xgb = xgb_model.predict(X_test_xgb)
mae_xgb = mean_absolute_error(y_test_xgb, y_pred_xgb)
print(f"MAE: {mae_xgb:.2f}")

# ---------- Plot Residuals ----------
residuals = y_test_xgb - y_pred_xgb
plt.figure(figsize=(8, 6))
plt.scatter(y_pred_xgb, residuals, alpha=0.5, s=10)
plt.axhline(y=0, color='r', linestyle='--', lw=2)
plt.xlabel('Predicted CO')
plt.ylabel('Residuals')
plt.title('Sydney CO - XGBoost Residual Plot')
plt.tight_layout()
plt.savefig('residual_scatter.png', dpi=300, bbox_inches='tight')
plt.close()
print("Saved: residual_scatter.png")

# ==========================================
# 5. New York - AdaBoost Binary Classification
# ==========================================
print("\n" + "="*50)
print("5. New York - AdaBoost Binary Classification")
print("="*50)

# ---------- Create Binary Target ----------
ny_ada = newyork_df[['CO', 'NO2', 'SO2', 'O3', 'AQI', 'PM2.5']].dropna()
ny_ada['high_pollution'] = (ny_ada['PM2.5'] > 40).astype(int)

X_ny_ada = ny_ada[['CO', 'NO2', 'SO2', 'O3', 'AQI']]
y_ny_ada = ny_ada['high_pollution']

# ---------- Train-Test Split 75-25 ----------
X_train_ada, X_test_ada, y_train_ada, y_test_ada = train_test_split(
    X_ny_ada, y_ny_ada, test_size=0.25, random_state=42
)

# ---------- Train AdaBoost Classifier ----------
ada_model = AdaBoostClassifier(
    n_estimators=50,
    learning_rate=1.0,
    random_state=42
)
ada_model.fit(X_train_ada, y_train_ada)

# ---------- Evaluate and Output F1 Score ----------
y_pred_ada = ada_model.predict(X_test_ada)
f1_ada = f1_score(y_test_ada, y_pred_ada)
print(f"F1 Score: {f1_ada:.4f}")

# ---------- Plot ROC Curve ----------
y_pred_proba_ada = ada_model.predict_proba(X_test_ada)[:, 1]
fpr, tpr, _ = roc_curve(y_test_ada, y_pred_proba_ada)
roc_auc = auc(fpr, tpr)

plt.figure(figsize=(7, 7))
plt.plot(fpr, tpr, lw=2, label=f'ROC curve (AUC = {roc_auc:.2f})')
plt.plot([0, 1], [0, 1], 'k--', lw=2)
plt.xlabel('False Positive Rate')
plt.ylabel('True Positive Rate')
plt.title('New York High Pollution - ROC Curve')
plt.legend()
plt.tight_layout()
plt.savefig('roc_curve.png', dpi=300, bbox_inches='tight')
plt.close()
print("Saved: roc_curve.png")

# ==========================================
# 6. London - ElasticNetCV (NO2 Prediction)
# ==========================================
print("\n" + "="*50)
print("6. London - ElasticNetCV")
print("="*50)

# ---------- Prepare and Standardize London Data ----------
london_en = london_df[['CO', 'SO2', 'O3', 'PM2.5', 'PM10', 'NO2']].dropna()
X_london_en = london_en[['CO', 'SO2', 'O3', 'PM2.5', 'PM10']]
y_london_en = london_en['NO2']

scaler_en = StandardScaler()
X_london_en_scaled = scaler_en.fit_transform(X_london_en)

# ---------- Train ElasticNetCV ----------
alphas = np.logspace(-2, 2, 100)
en_model = ElasticNetCV(
    l1_ratio=[0.1, 0.5, 0.9],
    alphas=alphas,
    cv=5,
    random_state=42
)
en_model.fit(X_london_en_scaled, y_london_en)

# ---------- Output Optimal Alpha ----------
optimal_alpha = en_model.alpha_
print(f"Optimal Alpha: {optimal_alpha:.4f}")

# ==========================================
# 7. Sydney - Ridge Regression (PM2.5 Prediction)
# ==========================================
print("\n" + "="*50)
print("7. Sydney - Ridge Regression")
print("="*50)

# ---------- Prepare and Standardize Sydney Data ----------
sydney_ridge = sydney_df[['CO', 'NO2', 'SO2', 'O3', 'AQI', 'PM2.5']].dropna()
X_sydney_ridge = sydney_ridge[['CO', 'NO2', 'SO2', 'O3', 'AQI']]
y_sydney_ridge = sydney_ridge['PM2.5']

scaler_ridge = StandardScaler()
X_sydney_ridge_scaled = scaler_ridge.fit_transform(X_sydney_ridge)

# ---------- Train Ridge Regression ----------
ridge_model = Ridge(alpha=10.0)
ridge_model.fit(X_sydney_ridge_scaled, y_sydney_ridge)

# ---------- Evaluate and Output MSE ----------
y_pred_ridge = ridge_model.predict(X_sydney_ridge_scaled)
mse_ridge = mean_squared_error(y_sydney_ridge, y_pred_ridge)
print(f"MSE: {mse_ridge:.2f}")

# ==========================================
# 8. New York - Lasso Regression (AQI Prediction)
# ==========================================
print("\n" + "="*50)
print("8. New York - Lasso Regression")
print("="*50)

# ---------- Prepare and Standardize New York Data ----------
ny_lasso = newyork_df[['CO', 'NO2', 'SO2', 'O3', 'PM2.5', 'PM10', 'AQI']].dropna()
X_ny_lasso = ny_lasso[['CO', 'NO2', 'SO2', 'O3', 'PM2.5', 'PM10']]
y_ny_lasso = ny_lasso['AQI']

scaler_lasso = StandardScaler()
X_ny_lasso_scaled = scaler_lasso.fit_transform(X_ny_lasso)

# ---------- Train Lasso Regression ----------
lasso_model = Lasso(alpha=1.0)
lasso_model.fit(X_ny_lasso_scaled, y_ny_lasso)

# ---------- Count Non-zero Coefficients ----------
non_zero_count = np.sum(lasso_model.coef_ != 0)
print(f"Non-zero Coefficients: {non_zero_count}")

# ==========================================
# 9. London - SVR (O3 Prediction)
# ==========================================
print("\n" + "="*50)
print("9. London - SVR")
print("="*50)

# ---------- Prepare and Standardize London Data ----------
london_svr = london_df[['CO', 'NO2', 'SO2', 'PM2.5', 'PM10', 'O3']].dropna()
X_london_svr = london_svr[['CO', 'NO2', 'SO2', 'PM2.5', 'PM10']]
y_london_svr = london_svr['O3']

scaler_svr = StandardScaler()
X_london_svr_scaled = scaler_svr.fit_transform(X_london_svr)

# ---------- Train-Test Split 70-30 ----------
X_train_svr, X_test_svr, y_train_svr, y_test_svr = train_test_split(
    X_london_svr_scaled, y_london_svr, test_size=0.30, random_state=42
)

# ---------- Train SVR ----------
svr_model = SVR(kernel='rbf', C=100, gamma=0.1, epsilon=0.1)
svr_model.fit(X_train_svr, y_train_svr)

# ---------- Evaluate and Output R-squared ----------
y_pred_svr = svr_model.predict(X_test_svr)
r2_svr = r2_score(y_test_svr, y_pred_svr)
print(f"R-squared: {r2_svr:.4f}")

# ==========================================
# 10. Combined Sydney + London - KMeans Clustering
# ==========================================
print("\n" + "="*50)
print("10. Combined Sydney + London - KMeans Clustering")
print("="*50)

# ---------- Combine Datasets ----------
sydney_kmeans = sydney_df[['PM2.5', 'PM10', 'AQI']].dropna()
london_kmeans = london_df[['PM2.5', 'PM10', 'AQI']].dropna()
combined_kmeans = pd.concat([sydney_kmeans, london_kmeans], axis=0, ignore_index=True)

# ---------- Standardize Features ----------
scaler_kmeans = StandardScaler()
X_kmeans_scaled = scaler_kmeans.fit_transform(combined_kmeans)

# ---------- Train KMeans ----------
kmeans_model = KMeans(n_clusters=5, max_iter=300, random_state=42)
kmeans_model.fit(X_kmeans_scaled)

# ---------- Calculate and Output Silhouette Score ----------
silhouette_avg = silhouette_score(X_kmeans_scaled, kmeans_model.labels_)
print(f"Silhouette Score: {silhouette_avg:.4f}")

# ==========================================
# 11. New York - PCA (Dimensionality Reduction)
# ==========================================
print("\n" + "="*50)
print("11. New York - PCA")
print("="*50)

# ---------- Prepare and Standardize New York Data ----------
ny_pca = newyork_df[['CO', 'NO2', 'SO2', 'O3', 'PM2.5', 'PM10']].dropna()

scaler_pca = StandardScaler()
X_ny_pca_scaled = scaler_pca.fit_transform(ny_pca)

# ---------- Train PCA ----------
pca_model = PCA(n_components=4)
pca_model.fit(X_ny_pca_scaled)

# ---------- Calculate and Output Cumulative Variance ----------
cumulative_variance = np.sum(pca_model.explained_variance_ratio_)
print(f"Cumulative Explained Variance: {cumulative_variance:.4f}")

# ==========================================
# 12. Open-Ended Comparison
# ==========================================
print("\n" + "="*50)
print("12. XGBoost vs Random Forest Comparison")
print("="*50)

# XGBoost vs RandomForest - XGBoost achieves better prediction through sequential residual correction while RandomForest relies on independent parallel trees.

print(f"\nXGBoost MAE: {mae_xgb:.2f}")
print(f"Random Forest RMSE: {rmse_rf:.2f}")
print("\nConclusion: XGBoost generally achieves superior predictive performance through")
print("sequential boosting that learns from previous errors, whereas Random Forest uses")
print("parallel ensemble averaging which may miss complex sequential patterns.")

print("\n" + "="*50)
print("Analysis Complete!")
print("="*50)
print("\nKey Outputs Generated:")
print(f"1. Sydney RF RMSE: {rmse_rf:.2f}")
print(f"2. New York GB Accuracy: {accuracy_gb:.2f}%")
print(f"3. London PLS R²: {r2_pls:.4f}")
print(f"4. Sydney XGB MAE: {mae_xgb:.2f}")
print(f"5. New York AdaBoost F1: {f1_ada:.4f}")
print(f"6. London ElasticNet Alpha: {optimal_alpha:.4f}")
print(f"7. Sydney Ridge MSE: {mse_ridge:.2f}")
print(f"8. New York Lasso Non-zero Coefs: {non_zero_count}")
print(f"9. London SVR R²: {r2_svr:.4f}")
print(f"10. Combined KMeans Silhouette: {silhouette_avg:.4f}")
print(f"11. New York PCA Cumulative Variance: {cumulative_variance:.4f}")
print("\nVisualization Files:")
print("- feature_importance.png")
print("- confusion_matrix.png")
print("- pls_scatter.png")
print("- residual_scatter.png")
print("- roc_curve.png")
