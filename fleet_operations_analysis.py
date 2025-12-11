# ==========================================
# Fleet Operations Analysis Script
# ==========================================
# Requirements: pandas, numpy, matplotlib, scipy, scikit-learn, openpyxl, statsmodels
# Input files: customers.xlsx, driver_monthly_metrics.xlsx, maintenance_records.xlsx (in same directory)
# Output files: svm_driver_classification.png, polynomial_regression_maintenance.png,
#               knn_revenue_prediction.png, exponential_smoothing_forecast.png,
#               gmm_driver_clusters_3d.png
# ==========================================

import pandas as pd
import numpy as np
import matplotlib.pyplot as plt
from sklearn.svm import SVC
from sklearn.preprocessing import PolynomialFeatures, StandardScaler
from sklearn.linear_model import LinearRegression
from sklearn.neighbors import KNeighborsRegressor
from sklearn.model_selection import train_test_split
from sklearn.metrics import accuracy_score, r2_score, mean_squared_error
from sklearn.mixture import GaussianMixture
from statsmodels.tsa.holtwinters import SimpleExpSmoothing
from scipy.stats import spearmanr, linregress
import warnings
warnings.filterwarnings('ignore')

# ---------- Load Excel Files ----------
print("Loading data files...")
customers = pd.read_excel('customers.xlsx')
driver_metrics = pd.read_excel('driver_monthly_metrics.xlsx')
maintenance = pd.read_excel('maintenance_records.xlsx')
print("Data loaded successfully.\n")

# ==========================================
# TASK 1: SVM Classification for High-Performing Drivers
# ==========================================
print("=" * 60)
print("TASK 1: SVM Classification for High-Performing Drivers")
print("=" * 60)

# ---------- Aggregate Driver Metrics ----------
driver_agg_svm = driver_metrics.groupby('driver_id').agg({
    'on_time_delivery_rate': 'mean',
    'average_mpg': 'mean',
    'trips_completed': 'sum',
    'total_miles': 'sum'
}).reset_index()

driver_agg_svm.columns = ['driver_id', 'mean_on_time_delivery_rate', 'mean_average_mpg',
                          'total_trips_completed', 'total_total_miles']

# ---------- Create Binary Target Variable ----------
driver_agg_svm['high_performer'] = (driver_agg_svm['mean_on_time_delivery_rate'] >= 0.5).astype(int)

# ---------- Prepare Features and Target ----------
X_svm = driver_agg_svm[['mean_on_time_delivery_rate', 'mean_average_mpg',
                        'total_trips_completed', 'total_total_miles']]
y_svm = driver_agg_svm['high_performer']

# ---------- Train-Test Split ----------
# Note: Stratification removed due to highly imbalanced classes (only 1 high performer)
X_train_svm, X_test_svm, y_train_svm, y_test_svm = train_test_split(
    X_svm, y_svm, test_size=0.30, random_state=25
)

# ---------- Train SVM Classifier with RBF Kernel ----------
svm_model = SVC(kernel='rbf', random_state=25)
svm_model.fit(X_train_svm, y_train_svm)

# ---------- Predict and Evaluate ----------
y_pred_svm = svm_model.predict(X_test_svm)
svm_accuracy = accuracy_score(y_test_svm, y_pred_svm)

print(f"SVM Classification Accuracy on Test Set: {svm_accuracy:.4f}")

# ---------- Generate Scatter Plot ----------
y_pred_full_svm = svm_model.predict(X_svm)
plt.figure(figsize=(10, 6))
scatter = plt.scatter(driver_agg_svm['mean_on_time_delivery_rate'],
                     driver_agg_svm['mean_average_mpg'],
                     c=y_pred_full_svm, cmap='coolwarm', alpha=0.6, edgecolors='k')
plt.xlabel('Mean On-Time Delivery Rate', fontsize=12)
plt.ylabel('Mean Average MPG', fontsize=12)
plt.title('SVM Driver Classification: Predicted Classes', fontsize=14, fontweight='bold')
plt.colorbar(scatter, label='Predicted Class (0=Low, 1=High)')
plt.grid(alpha=0.3)
plt.tight_layout()
plt.savefig('svm_driver_classification.png', dpi=300)
print("Saved: svm_driver_classification.png\n")
plt.close()

# ==========================================
# TASK 2: Polynomial Regression for Maintenance Cost
# ==========================================
print("=" * 60)
print("TASK 2: Polynomial Regression for Maintenance Cost")
print("=" * 60)

# ---------- Filter Preventive Maintenance Records ----------
preventive_maint = maintenance[maintenance['maintenance_type'] == 'Preventive'].copy()

# ---------- Prepare Features and Target ----------
X_poly = preventive_maint[['odometer_reading']].values
y_poly = preventive_maint['total_cost'].values

# ---------- Train-Test Split ----------
X_train_poly, X_test_poly, y_train_poly, y_test_poly = train_test_split(
    X_poly, y_poly, test_size=0.15, random_state=30
)

# ---------- Create Polynomial Features (Degree 3) ----------
poly_features = PolynomialFeatures(degree=3)
X_train_poly_transformed = poly_features.fit_transform(X_train_poly)
X_test_poly_transformed = poly_features.transform(X_test_poly)

# ---------- Fit Polynomial Regression Model ----------
poly_model = LinearRegression()
poly_model.fit(X_train_poly_transformed, y_train_poly)

# ---------- Predict and Evaluate ----------
y_pred_poly = poly_model.predict(X_test_poly_transformed)
poly_r2 = r2_score(y_test_poly, y_pred_poly)

print(f"Polynomial Regression R-squared on Test Set: {poly_r2:.4f}")

# ---------- Generate Scatter Plot with Fitted Curve ----------
X_range = np.linspace(X_poly.min(), X_poly.max(), 300).reshape(-1, 1)
X_range_transformed = poly_features.transform(X_range)
y_range_pred = poly_model.predict(X_range_transformed)

plt.figure(figsize=(10, 6))
plt.scatter(X_test_poly, y_test_poly, alpha=0.6, label='Actual Test Data', color='blue', edgecolors='k')
plt.plot(X_range, y_range_pred, color='red', linewidth=2, label='Polynomial Fit (Degree 3)')
plt.xlabel('Odometer Reading (miles)', fontsize=12)
plt.ylabel('Total Cost (dollars)', fontsize=12)
plt.title('Polynomial Regression: Preventive Maintenance Cost vs Odometer', fontsize=14, fontweight='bold')
plt.legend()
plt.grid(alpha=0.3)
plt.tight_layout()
plt.savefig('polynomial_regression_maintenance.png', dpi=300)
print("Saved: polynomial_regression_maintenance.png\n")
plt.close()

# ==========================================
# TASK 3: KNN Regression for Revenue Prediction
# ==========================================
print("=" * 60)
print("TASK 3: KNN Regression for Revenue Prediction")
print("=" * 60)

# ---------- Aggregate Driver Revenue by Month ----------
driver_metrics['year_month'] = pd.to_datetime(driver_metrics['month']).dt.to_period('M')
monthly_revenue = driver_metrics.groupby('year_month')['total_revenue'].sum().reset_index()
monthly_revenue.columns = ['year_month', 'monthly_driver_revenue']

# ---------- Create Monthly Customer Records ----------
customers['monthly_contribution'] = customers['annual_revenue_potential'] / 12
customers['is_Dedicated'] = (customers['customer_type'] == 'Dedicated').astype(int)
customers['is_Active'] = (customers['account_status'] == 'Active').astype(int)

# ---------- Merge with Monthly Revenue ----------
# Create cross join of months and customers
months = monthly_revenue['year_month'].unique()
customer_monthly = []
for month in months:
    for _, customer in customers.iterrows():
        customer_monthly.append({
            'year_month': month,
            'credit_terms_days': customer['credit_terms_days'],
            'is_Dedicated': customer['is_Dedicated'],
            'is_Active': customer['is_Active'],
            'monthly_contribution': customer['monthly_contribution']
        })

customer_monthly_df = pd.DataFrame(customer_monthly)
merged_knn = customer_monthly_df.merge(monthly_revenue, on='year_month', how='left')

# ---------- Prepare Features and Target ----------
X_knn = merged_knn[['credit_terms_days', 'is_Dedicated', 'is_Active']]
y_knn = merged_knn['monthly_driver_revenue']

# ---------- Train-Test Split ----------
X_train_knn, X_test_knn, y_train_knn, y_test_knn = train_test_split(
    X_knn, y_knn, test_size=0.25, random_state=40
)

# ---------- Train KNN Regressor (k=7) ----------
knn_model = KNeighborsRegressor(n_neighbors=7)
knn_model.fit(X_train_knn, y_train_knn)

# ---------- Predict and Evaluate ----------
y_pred_knn = knn_model.predict(X_test_knn)
knn_mse = mean_squared_error(y_test_knn, y_pred_knn)

print(f"KNN Regression Mean Squared Error on Test Set: {knn_mse:.2f}")

# ---------- Generate Actual vs Predicted Scatter Plot ----------
plt.figure(figsize=(10, 6))
plt.scatter(y_test_knn, y_pred_knn, alpha=0.6, color='green', edgecolors='k')
plt.plot([y_test_knn.min(), y_test_knn.max()],
         [y_test_knn.min(), y_test_knn.max()],
         'r--', linewidth=2, label='Perfect Prediction')
plt.xlabel('Actual Monthly Revenue', fontsize=12)
plt.ylabel('Predicted Monthly Revenue', fontsize=12)
plt.title('KNN Regression: Actual vs Predicted Monthly Revenue', fontsize=14, fontweight='bold')
plt.legend()
plt.grid(alpha=0.3)
plt.tight_layout()
plt.savefig('knn_revenue_prediction.png', dpi=300)
print("Saved: knn_revenue_prediction.png\n")
plt.close()

# ==========================================
# TASK 4: Exponential Smoothing for Maintenance Cost Forecast
# ==========================================
print("=" * 60)
print("TASK 4: Exponential Smoothing for Maintenance Cost Forecast")
print("=" * 60)

# ---------- Aggregate Monthly Maintenance Costs ----------
maintenance['year_month'] = pd.to_datetime(maintenance['maintenance_date']).dt.to_period('M')
monthly_costs = maintenance.groupby('year_month')['total_cost'].sum().reset_index()
monthly_costs = monthly_costs.sort_values('year_month')
monthly_costs['year_month_dt'] = monthly_costs['year_month'].dt.to_timestamp()

# ---------- Apply Simple Exponential Smoothing (alpha=0.3) ----------
ses_model = SimpleExpSmoothing(monthly_costs['total_cost'].values)
ses_fit = ses_model.fit(smoothing_level=0.3, optimized=False)

# ---------- Generate 6-Month Forecast ----------
forecast_steps = 6
forecast_values = ses_fit.forecast(steps=forecast_steps)
sixth_month_forecast = forecast_values[5]

print(f"Forecasted Monthly Maintenance Cost for 6th Month Ahead: ${sixth_month_forecast:.2f}")

# ---------- Generate Line Plot ----------
last_date = monthly_costs['year_month_dt'].iloc[-1]
forecast_dates = pd.date_range(start=last_date + pd.DateOffset(months=1), periods=forecast_steps, freq='MS')

plt.figure(figsize=(14, 6))
plt.plot(monthly_costs['year_month_dt'], monthly_costs['total_cost'],
         marker='o', linewidth=2, label='Historical Actual', color='blue')
plt.plot(forecast_dates, forecast_values,
         marker='s', linestyle='--', linewidth=2, label='6-Month Forecast', color='orange')
plt.xlabel('Month', fontsize=12)
plt.ylabel('Monthly Maintenance Cost (dollars)', fontsize=12)
plt.title('Exponential Smoothing: Maintenance Cost Forecast', fontsize=14, fontweight='bold')
plt.legend()
plt.grid(alpha=0.3)
plt.xticks(rotation=45)
plt.tight_layout()
plt.savefig('exponential_smoothing_forecast.png', dpi=300)
print("Saved: exponential_smoothing_forecast.png\n")
plt.close()

# ==========================================
# TASK 5: Gaussian Mixture Model Clustering
# ==========================================
print("=" * 60)
print("TASK 5: Gaussian Mixture Model Clustering")
print("=" * 60)

# ---------- Aggregate Driver Metrics ----------
driver_agg_gmm = driver_metrics.groupby('driver_id').agg({
    'average_mpg': 'mean',
    'on_time_delivery_rate': 'mean',
    'average_idle_hours': 'mean'
}).reset_index()

driver_agg_gmm.columns = ['driver_id', 'mean_average_mpg',
                          'mean_on_time_delivery_rate', 'mean_average_idle_hours']

# ---------- Standardize Features ----------
scaler = StandardScaler()
features_gmm = driver_agg_gmm[['mean_average_mpg', 'mean_on_time_delivery_rate',
                               'mean_average_idle_hours']].values
features_gmm_scaled = scaler.fit_transform(features_gmm)

# ---------- Apply Gaussian Mixture Model (4 Components) ----------
gmm_model = GaussianMixture(n_components=4, covariance_type='full', random_state=15)
gmm_clusters = gmm_model.fit_predict(features_gmm_scaled)
driver_agg_gmm['cluster'] = gmm_clusters

# ---------- Identify Cluster with Highest Average On-Time Delivery Rate ----------
cluster_stats = driver_agg_gmm.groupby('cluster')['mean_on_time_delivery_rate'].mean()
best_cluster = cluster_stats.idxmax()

print(f"Cluster with Highest Average On-Time Delivery Rate: {best_cluster}")
print(f"Average On-Time Delivery Rate: {cluster_stats[best_cluster]:.4f}")

# ---------- Generate 3D Scatter Plot ----------
fig = plt.figure(figsize=(12, 8))
ax = fig.add_subplot(111, projection='3d')
scatter = ax.scatter(driver_agg_gmm['mean_average_mpg'],
                    driver_agg_gmm['mean_on_time_delivery_rate'],
                    driver_agg_gmm['mean_average_idle_hours'],
                    c=driver_agg_gmm['cluster'], cmap='viridis',
                    s=50, alpha=0.6, edgecolors='k')
ax.set_xlabel('Mean Average MPG', fontsize=11)
ax.set_ylabel('Mean On-Time Delivery Rate', fontsize=11)
ax.set_zlabel('Mean Average Idle Hours', fontsize=11)
ax.set_title('GMM Clustering: Driver Performance Profiles', fontsize=14, fontweight='bold')
plt.colorbar(scatter, label='Cluster (0-3)', pad=0.1)
plt.tight_layout()
plt.savefig('gmm_driver_clusters_3d.png', dpi=300)
print("Saved: gmm_driver_clusters_3d.png\n")
plt.close()

# ==========================================
# TASK 6: Facility Location with Maximum Average Downtime
# ==========================================
print("=" * 60)
print("TASK 6: Facility Location with Maximum Average Downtime")
print("=" * 60)

# ---------- Group by Facility and Calculate Mean Downtime ----------
facility_downtime = maintenance.groupby('facility_location')['downtime_hours'].mean()
max_downtime_facility = facility_downtime.idxmax()
max_downtime_value = facility_downtime.max()

print(f"Facility Location with Maximum Average Downtime: {max_downtime_facility}")
print(f"Average Downtime Hours: {max_downtime_value:.2f}\n")

# ==========================================
# TASK 7: Primary Freight Type with Maximum Revenue Potential
# ==========================================
print("=" * 60)
print("TASK 7: Primary Freight Type with Maximum Revenue Potential")
print("=" * 60)

# ---------- Group by Freight Type and Sum Revenue Potential ----------
freight_revenue = customers.groupby('primary_freight_type')['annual_revenue_potential'].sum()
max_revenue_freight = freight_revenue.idxmax()
max_revenue_value = freight_revenue.max()

print(f"Primary Freight Type with Maximum Total Revenue Potential: {max_revenue_freight}")
print(f"Total Annual Revenue Potential: ${max_revenue_value:,.2f}\n")

# ==========================================
# TASK 8: Driver with Highest On-Time Delivery Rate Improvement
# ==========================================
print("=" * 60)
print("TASK 8: Driver with Highest On-Time Delivery Rate Improvement")
print("=" * 60)

# ---------- Calculate Linear Trend Slope for Each Driver ----------
driver_metrics_sorted = driver_metrics.sort_values(['driver_id', 'month'])
driver_slopes = {}

for driver_id, group in driver_metrics_sorted.groupby('driver_id'):
    group = group.sort_values('month').reset_index(drop=True)
    if len(group) > 1:
        month_indices = np.arange(1, len(group) + 1)
        otr_values = group['on_time_delivery_rate'].values
        slope, _, _, _, _ = linregress(month_indices, otr_values)
        driver_slopes[driver_id] = slope

# ---------- Find Driver with Maximum Slope ----------
best_improvement_driver = max(driver_slopes, key=driver_slopes.get)
best_slope = driver_slopes[best_improvement_driver]

print(f"Driver with Maximum On-Time Delivery Rate Improvement: {best_improvement_driver}")
print(f"Improvement Slope: {best_slope:.6f}\n")

# ==========================================
# TASK 9: Emergency vs Scheduled Maintenance Cost Ratio
# ==========================================
print("=" * 60)
print("TASK 9: Emergency vs Scheduled Maintenance Cost Ratio")
print("=" * 60)

# ---------- Calculate Total Costs ----------
emergency_cost = maintenance[maintenance['maintenance_type'] == 'Repair']['total_cost'].sum()
scheduled_cost = maintenance[maintenance['maintenance_type'] == 'Preventive']['total_cost'].sum()
cost_ratio = emergency_cost / scheduled_cost

print(f"Emergency Maintenance Total Cost: ${emergency_cost:,.2f}")
print(f"Scheduled Maintenance Total Cost: ${scheduled_cost:,.2f}")
print(f"Ratio of Emergency to Scheduled Maintenance Costs: {cost_ratio:.4f}\n")

# ==========================================
# TASK 10: Spearman Correlation Between MPG and Idle Hours
# ==========================================
print("=" * 60)
print("TASK 10: Spearman Correlation Between MPG and Idle Hours")
print("=" * 60)

# ---------- Calculate Spearman Correlation ----------
mpg_values = driver_metrics['average_mpg'].values
idle_values = driver_metrics['average_idle_hours'].values

spearman_corr, spearman_pvalue = spearmanr(mpg_values, idle_values)

print(f"Spearman Correlation Coefficient between Average MPG and Average Idle Hours: {spearman_corr:.4f}")
print(f"P-value: {spearman_pvalue:.6f}")

# ---------- Interpretation ----------
if spearman_corr < -0.3:
    print("\nDriver Characteristic Predicting Fuel Efficiency Degradation:")
    print("Higher idle hours are associated with lower fuel efficiency (negative correlation).")
    print("Drivers who idle excessively tend to have reduced MPG performance.\n")
elif spearman_corr > 0.3:
    print("\nDriver Characteristic Predicting Fuel Efficiency Degradation:")
    print("Positive correlation detected - unusual pattern requiring further investigation.\n")
else:
    print("\nDriver Characteristic Predicting Fuel Efficiency Degradation:")
    print("Weak correlation - idle hours may not be the primary predictor of fuel efficiency.\n")

# ==========================================
# Analysis Complete
# ==========================================
print("=" * 60)
print("FLEET OPERATIONS ANALYSIS COMPLETED SUCCESSFULLY")
print("=" * 60)
print("\nKey Outputs Generated:")
print(f"1. SVM Classification Accuracy: {svm_accuracy:.4f}")
print(f"2. Polynomial Regression R²: {poly_r2:.4f}")
print(f"3. KNN Regression MSE: {knn_mse:.2f}")
print(f"4. 6-Month Maintenance Forecast: ${sixth_month_forecast:.2f}")
print(f"5. Best Performance Cluster: {best_cluster}")
print(f"6. Max Downtime Facility: {max_downtime_facility}")
print(f"7. Top Revenue Freight Type: {max_revenue_freight}")
print(f"8. Top Improvement Driver: {best_improvement_driver}")
print(f"9. Emergency/Scheduled Cost Ratio: {cost_ratio:.4f}")
print(f"10. MPG-Idle Spearman Correlation: {spearman_corr:.4f}")
print("\n5 Visualization Files Saved:")
print("- svm_driver_classification.png")
print("- polynomial_regression_maintenance.png")
print("- knn_revenue_prediction.png")
print("- exponential_smoothing_forecast.png")
print("- gmm_driver_clusters_3d.png")
print("=" * 60)
