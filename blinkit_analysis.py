# ==========================================
# Integrated Blinkit Delivery Analysis Script
# ==========================================
# Requirements: pandas, numpy, matplotlib, scipy, scikit-learn
# Input files: blinkit_customers.xlsx, blinkit_customer_feedback.xlsx, blinkit_delivery_performance.xlsx
# Output files: ridge_alpha_mse.png, lasso_coefficients.png, svr_predictions.png,
#               exponential_smoothing.png, roc_curve.png
# ==========================================

import pandas as pd
import numpy as np
import matplotlib.pyplot as plt
from sklearn.linear_model import Ridge, Lasso, LogisticRegression
from sklearn.svm import SVR
from sklearn.preprocessing import StandardScaler
from sklearn.model_selection import cross_val_score
from sklearn.metrics import mean_squared_error, mean_absolute_percentage_error, roc_curve
from scipy import stats
from datetime import datetime
import warnings
warnings.filterwarnings('ignore')

# ---------- Load Datasets ----------
print("Loading datasets...")
customers = pd.read_excel('blinkit_customers.xlsx')
feedback = pd.read_excel('blinkit_customer_feedback.xlsx')
delivery = pd.read_excel('blinkit_delivery_performance.xlsx')

# Convert date columns to datetime
customers['registration_date'] = pd.to_datetime(customers['registration_date'])
feedback['feedback_date'] = pd.to_datetime(feedback['feedback_date'])
delivery['promised_time'] = pd.to_datetime(delivery['promised_time'])
delivery['actual_time'] = pd.to_datetime(delivery['actual_time'])

print(f"Customers: {customers.shape[0]} records")
print(f"Feedback: {feedback.shape[0]} records")
print(f"Delivery: {delivery.shape[0]} records\n")

# ==========================================
# ANALYSIS 1: Ridge Regression with Cross-Validation
# ==========================================
print("=" * 60)
print("ANALYSIS 1: Ridge Regression for Avg Order Value Prediction")
print("=" * 60)

# ---------- Feature Engineering for Ridge Regression ----------
reference_date = pd.Timestamp('2024-12-20')
customers['account_age_months'] = ((reference_date - customers['registration_date']).dt.days / 30.44).round()

# Calculate pincode variance (first 3 digits)
customers['pincode_prefix'] = customers['pincode'].astype(str).str[:3]
pincode_variance = customers.groupby('pincode_prefix')['avg_order_value'].std().fillna(0)
customers['pincode_variance'] = customers['pincode_prefix'].map(pincode_variance).fillna(0)

# ---------- Prepare Ridge Regression Data ----------
ridge_features = ['total_orders', 'account_age_months', 'pincode_variance']
ridge_data = customers[ridge_features + ['avg_order_value']].dropna()

X_ridge = ridge_data[ridge_features].values
y_ridge = ridge_data['avg_order_value'].values

# Standardize features
scaler_ridge = StandardScaler()
X_ridge_scaled = scaler_ridge.fit_transform(X_ridge)

# ---------- Cross-Validation for Optimal Alpha ----------
alphas = np.logspace(-2, 2, 50)  # 0.01 to 100 on log scale
mse_scores = []

for alpha in alphas:
    ridge_model = Ridge(alpha=alpha)
    cv_scores = cross_val_score(ridge_model, X_ridge_scaled, y_ridge,
                                 cv=5, scoring='neg_mean_squared_error')
    mse_scores.append(-cv_scores.mean())

optimal_alpha_idx = np.argmin(mse_scores)
optimal_alpha = alphas[optimal_alpha_idx]

# ---------- Fit Ridge with Optimal Alpha ----------
ridge_final = Ridge(alpha=optimal_alpha)
ridge_final.fit(X_ridge_scaled, y_ridge)
total_orders_coef = ridge_final.coef_[0]

print(f"Optimal Alpha: {optimal_alpha:.4f}")
print(f"Minimum CV MSE: {mse_scores[optimal_alpha_idx]:.2f}")
print(f"Total Orders Coefficient: {total_orders_coef:.4f}")

# ---------- Plot Alpha vs MSE ----------
plt.figure(figsize=(10, 6))
plt.semilogx(alphas, mse_scores, 'b-', linewidth=2)
plt.axvline(optimal_alpha, color='r', linestyle='--', label=f'Optimal α = {optimal_alpha:.4f}')
plt.xlabel('Alpha (Log Scale)', fontsize=12)
plt.ylabel('Mean Squared Error', fontsize=12)
plt.title('Ridge Regression: Cross-Validated MSE vs Alpha', fontsize=14, fontweight='bold')
plt.grid(True, alpha=0.3)
plt.legend()
plt.tight_layout()
plt.savefig('ridge_alpha_mse.png', dpi=300, bbox_inches='tight')
plt.close()

# ==========================================
# ANALYSIS 2: Lasso Regression with Feature Selection
# ==========================================
print("\n" + "=" * 60)
print("ANALYSIS 2: Lasso Regression for Rating Prediction")
print("=" * 60)

# ---------- Feature Engineering for Lasso ----------
# Create dummy variables for feedback_category
feedback_dummies = pd.get_dummies(feedback['feedback_category'], prefix='category')

# Sentiment strength mapping
sentiment_map = {'Positive': 2, 'Neutral': 1, 'Negative': 0}
feedback['sentiment_strength'] = feedback['sentiment'].map(sentiment_map)

# Extract month from feedback_date
feedback['feedback_month'] = feedback['feedback_date'].dt.month

# ---------- Prepare Lasso Data ----------
lasso_features = pd.concat([feedback_dummies,
                            feedback[['sentiment_strength', 'feedback_month']]], axis=1)
lasso_data = pd.concat([lasso_features, feedback['rating']], axis=1).dropna()

feature_names = lasso_features.columns.tolist()
X_lasso = lasso_data[feature_names].values
y_lasso = lasso_data['rating'].values

# Standardize features
scaler_lasso = StandardScaler()
X_lasso_scaled = scaler_lasso.fit_transform(X_lasso)

# ---------- Fit Lasso with alpha=0.1 ----------
lasso_model = Lasso(alpha=0.1, random_state=42)
lasso_model.fit(X_lasso_scaled, y_lasso)

# Count non-zero coefficients
non_zero_coefs = np.sum(lasso_model.coef_ != 0)
abs_coefs = np.abs(lasso_model.coef_)

print(f"Number of Non-Zero Coefficients: {non_zero_coefs}")
print(f"Features Selected: {non_zero_coefs} out of {len(feature_names)}")

# ---------- Plot Feature Coefficients ----------
plt.figure(figsize=(12, 6))
plt.bar(range(len(feature_names)), abs_coefs, color='steelblue', edgecolor='black')
plt.xlabel('Feature Index', fontsize=12)
plt.ylabel('Absolute Coefficient Value', fontsize=12)
plt.title('Lasso Regression: Absolute Coefficient Values', fontsize=14, fontweight='bold')
plt.xticks(range(len(feature_names)), feature_names, rotation=45, ha='right')
plt.grid(True, alpha=0.3, axis='y')
plt.tight_layout()
plt.savefig('lasso_coefficients.png', dpi=300, bbox_inches='tight')
plt.close()

# ==========================================
# ANALYSIS 3: Support Vector Regression
# ==========================================
print("\n" + "=" * 60)
print("ANALYSIS 3: Support Vector Regression for Delivery Time")
print("=" * 60)

# ---------- Feature Engineering for SVR ----------
delivery['hour_of_day'] = delivery['promised_time'].dt.hour

# Filter valid delivery times (-5 to 30 minutes)
svr_data = delivery[(delivery['delivery_time_minutes'] >= -5) &
                     (delivery['delivery_time_minutes'] <= 30)].copy()

# ---------- Prepare SVR Data ----------
svr_features = ['distance_km', 'hour_of_day']
svr_complete = svr_data[svr_features + ['delivery_time_minutes']].dropna()

# Split into train and validation (80-20)
train_size = int(0.8 * len(svr_complete))
svr_train = svr_complete.iloc[:train_size]
svr_val = svr_complete.iloc[train_size:]

X_train_svr = svr_train[svr_features].values
y_train_svr = svr_train['delivery_time_minutes'].values
X_val_svr = svr_val[svr_features].values
y_val_svr = svr_val['delivery_time_minutes'].values

# Standardize features
scaler_svr = StandardScaler()
X_train_svr_scaled = scaler_svr.fit_transform(X_train_svr)
X_val_svr_scaled = scaler_svr.transform(X_val_svr)

# ---------- Fit SVR with RBF Kernel ----------
svr_model = SVR(kernel='rbf', gamma=0.1, C=1.0, epsilon=0.1)
svr_model.fit(X_train_svr_scaled, y_train_svr)

# Predict on validation set
y_pred_svr = svr_model.predict(X_val_svr_scaled)

# Calculate MAPE (avoid division by zero)
mask = y_val_svr != 0
mape_svr = np.mean(np.abs((y_val_svr[mask] - y_pred_svr[mask]) / y_val_svr[mask])) * 100

print(f"Mean Absolute Percentage Error: {mape_svr:.2f}%")
print(f"Training samples: {len(X_train_svr)}")
print(f"Validation samples: {len(X_val_svr)}")

# ---------- Plot Actual vs Predicted ----------
plt.figure(figsize=(10, 8))
plt.scatter(y_val_svr, y_pred_svr, alpha=0.5, s=20, color='blue', edgecolors='black', linewidth=0.5)
plt.plot([y_val_svr.min(), y_val_svr.max()], [y_val_svr.min(), y_val_svr.max()],
         'r--', lw=2, label='Perfect Prediction')
plt.xlabel('Actual Delivery Time (minutes)', fontsize=12)
plt.ylabel('Predicted Delivery Time (minutes)', fontsize=12)
plt.title('SVR: Actual vs Predicted Delivery Time', fontsize=14, fontweight='bold')
plt.legend()
plt.grid(True, alpha=0.3)
plt.tight_layout()
plt.savefig('svr_predictions.png', dpi=300, bbox_inches='tight')
plt.close()

# ==========================================
# ANALYSIS 4: Exponential Smoothing Time Series
# ==========================================
print("\n" + "=" * 60)
print("ANALYSIS 4: Exponential Smoothing for Daily Orders")
print("=" * 60)

# ---------- Aggregate Daily Order Counts ----------
delivery['date'] = delivery['promised_time'].dt.date
daily_orders = delivery.groupby('date').size().reset_index(name='order_count')
daily_orders = daily_orders.sort_values('date').reset_index(drop=True)
daily_orders['date'] = pd.to_datetime(daily_orders['date'])

# ---------- Train-Test Split (80-20) ----------
train_size_ts = int(0.8 * len(daily_orders))
train_ts = daily_orders.iloc[:train_size_ts]
test_ts = daily_orders.iloc[train_size_ts:]

# ---------- Simple Exponential Smoothing ----------
alpha_es = 0.3
forecast = []
initial_forecast = train_ts['order_count'].mean()

# Generate forecasts for test period
previous_forecast = train_ts['order_count'].iloc[-1]  # Use last training actual as initial
for i in range(len(test_ts)):
    if i == 0:
        # First forecast uses last training value
        f = alpha_es * train_ts['order_count'].iloc[-1] + (1 - alpha_es) * initial_forecast
    else:
        # Subsequent forecasts use previous actual
        f = alpha_es * test_ts['order_count'].iloc[i-1] + (1 - alpha_es) * forecast[i-1]
    forecast.append(f)

# Calculate RMSE
rmse_es = np.sqrt(mean_squared_error(test_ts['order_count'], forecast))

print(f"Training period: {len(train_ts)} days")
print(f"Test period: {len(test_ts)} days")
print(f"RMSE: {rmse_es:.2f}")

# ---------- Plot Time Series Forecast ----------
plt.figure(figsize=(14, 6))
plt.plot(train_ts['date'], train_ts['order_count'], 'b-', label='Training Actual', linewidth=2)
plt.plot(test_ts['date'], test_ts['order_count'], 'g-', label='Test Actual', linewidth=2)
plt.plot(test_ts['date'], forecast, 'r--', label='Forecast', linewidth=2)
plt.xlabel('Date', fontsize=12)
plt.ylabel('Order Count', fontsize=12)
plt.title('Exponential Smoothing: Daily Order Forecasting', fontsize=14, fontweight='bold')
plt.legend()
plt.grid(True, alpha=0.3)
plt.xticks(rotation=45)
plt.tight_layout()
plt.savefig('exponential_smoothing.png', dpi=300, bbox_inches='tight')
plt.close()

# ==========================================
# ANALYSIS 5: Logistic Regression for High Rating
# ==========================================
print("\n" + "=" * 60)
print("ANALYSIS 5: Logistic Regression for High Rating Classification")
print("=" * 60)

# ---------- Merge Datasets ----------
merged_data = feedback.merge(delivery, on='order_id', how='inner')

# ---------- Create Binary Target and Features ----------
merged_data['high_rating'] = (merged_data['rating'] >= 4).astype(int)
merged_data['on_time_delivery'] = (merged_data['delivery_status'] == 'On Time').astype(int)
merged_data['delivery_speed_category'] = (merged_data['delivery_time_minutes'] < 5).astype(int)
merged_data['is_positive_sentiment'] = (merged_data['sentiment'] == 'Positive').astype(int)

# ---------- Prepare Logistic Regression Data ----------
logit_features = ['on_time_delivery', 'delivery_speed_category', 'is_positive_sentiment']
logit_data = merged_data[logit_features + ['high_rating']].dropna()

X_logit = logit_data[logit_features].values
y_logit = logit_data['high_rating'].values

# ---------- Fit Logistic Regression ----------
logit_model = LogisticRegression(random_state=42)
logit_model.fit(X_logit, y_logit)

# Predict probabilities
y_proba = logit_model.predict_proba(X_logit)[:, 1]

# ---------- ROC Curve and Optimal Threshold ----------
fpr, tpr, thresholds = roc_curve(y_logit, y_proba)

# Find threshold where sensitivity equals specificity
optimal_idx = np.argmin(np.abs(tpr - (1 - fpr)))
optimal_threshold = thresholds[optimal_idx]

print(f"Optimal Threshold (Sensitivity = Specificity): {optimal_threshold:.4f}")
print(f"TPR at optimal threshold: {tpr[optimal_idx]:.4f}")
print(f"FPR at optimal threshold: {fpr[optimal_idx]:.4f}")

# ---------- Plot ROC Curve ----------
plt.figure(figsize=(10, 8))
plt.plot(fpr, tpr, 'b-', linewidth=2, label='ROC Curve')
plt.plot([0, 1], [0, 1], 'r--', linewidth=2, label='Random Classifier')
plt.scatter([fpr[optimal_idx]], [tpr[optimal_idx]], color='red', s=100, zorder=5,
            label=f'Optimal Threshold = {optimal_threshold:.4f}')
plt.xlabel('False Positive Rate', fontsize=12)
plt.ylabel('True Positive Rate', fontsize=12)
plt.title('ROC Curve: High Rating Classification', fontsize=14, fontweight='bold')
plt.legend()
plt.grid(True, alpha=0.3)
plt.tight_layout()
plt.savefig('roc_curve.png', dpi=300, bbox_inches='tight')
plt.close()

# ==========================================
# ANALYSIS 6: Chi-Square Test for Independence
# ==========================================
print("\n" + "=" * 60)
print("ANALYSIS 6: Chi-Square Test - Category vs Sentiment")
print("=" * 60)

# ---------- Create Contingency Table ----------
contingency_table = pd.crosstab(feedback['feedback_category'], feedback['sentiment'])

# ---------- Calculate Chi-Square Statistic ----------
chi2, p_value, dof, expected = stats.chi2_contingency(contingency_table)

print(f"Chi-Square Statistic: {chi2:.2f}")
print(f"P-value: {p_value:.6f}")
print(f"Degrees of Freedom: {dof}")

# ==========================================
# ANALYSIS 7: Z-Score Normalization and Outlier Detection
# ==========================================
print("\n" + "=" * 60)
print("ANALYSIS 7: Z-Score Outlier Detection for Order Value")
print("=" * 60)

# ---------- Calculate Z-Scores ----------
mean_order_value = customers['avg_order_value'].mean()
std_order_value = customers['avg_order_value'].std()
customers['z_score'] = (customers['avg_order_value'] - mean_order_value) / std_order_value

# ---------- Identify Outliers (|z| > 3) ----------
outliers = customers[np.abs(customers['z_score']) > 3]
outlier_count = len(outliers)

print(f"Mean Order Value: ${mean_order_value:.2f}")
print(f"Std Dev: ${std_order_value:.2f}")
print(f"Number of Outliers (|z| > 3): {outlier_count}")

# ==========================================
# ANALYSIS 8: Median Absolute Deviation (MAD)
# ==========================================
print("\n" + "=" * 60)
print("ANALYSIS 8: MAD for Delivery Time Variability")
print("=" * 60)

# ---------- Calculate MAD ----------
median_delivery = delivery['delivery_time_minutes'].median()
absolute_deviations = np.abs(delivery['delivery_time_minutes'] - median_delivery)
mad_value = absolute_deviations.median() * 1.4826

print(f"Median Delivery Time: {median_delivery:.2f} minutes")
print(f"MAD: {mad_value:.2f}")

# ==========================================
# ANALYSIS 9: Interquartile Range Ratio
# ==========================================
print("\n" + "=" * 60)
print("ANALYSIS 9: IQR Ratio between Customer Segments")
print("=" * 60)

# ---------- Calculate IQR for Premium Segment ----------
premium_orders = customers[customers['customer_segment'] == 'Premium']['total_orders']
q1_premium = premium_orders.quantile(0.25)
q3_premium = premium_orders.quantile(0.75)
iqr_premium = q3_premium - q1_premium

# ---------- Calculate IQR for New Segment ----------
new_orders = customers[customers['customer_segment'] == 'New']['total_orders']
q1_new = new_orders.quantile(0.25)
q3_new = new_orders.quantile(0.75)
iqr_new = q3_new - q1_new

# ---------- Calculate Ratio ----------
iqr_ratio = iqr_premium / iqr_new

print(f"Premium IQR: {iqr_premium:.2f}")
print(f"New IQR: {iqr_new:.2f}")
print(f"IQR Ratio (Premium/New): {iqr_ratio:.2f}")

# ==========================================
# ANALYSIS 10: Variance Inflation Factor (VIF)
# ==========================================
print("\n" + "=" * 60)
print("ANALYSIS 10: VIF for Multicollinearity Detection")
print("=" * 60)

# ---------- Merge Datasets for VIF ----------
# Merge delivery with feedback to get customer_id
delivery_customers = delivery.merge(feedback[['order_id', 'customer_id']], on='order_id', how='inner')
# Merge with customers
vif_data = delivery_customers.merge(customers[['customer_id', 'total_orders', 'avg_order_value']],
                                     on='customer_id', how='inner')

# ---------- Prepare VIF Features ----------
vif_features = ['distance_km', 'total_orders', 'avg_order_value']
vif_complete = vif_data[vif_features].dropna()

# ---------- Calculate VIF for Each Feature ----------
from sklearn.linear_model import LinearRegression

vif_values = []
for i in range(len(vif_features)):
    X_vif = vif_complete.drop(columns=[vif_features[i]]).values
    y_vif = vif_complete[vif_features[i]].values

    lr = LinearRegression()
    lr.fit(X_vif, y_vif)
    r2 = lr.score(X_vif, y_vif)

    vif = 1 / (1 - r2) if r2 < 1 else np.inf
    vif_values.append(vif)

max_vif = np.max(vif_values)
max_vif_feature = vif_features[np.argmax(vif_values)]

print(f"VIF values:")
for feat, vif_val in zip(vif_features, vif_values):
    print(f"  {feat}: {vif_val:.2f}")
print(f"Maximum VIF: {max_vif:.2f} ({max_vif_feature})")

# ==========================================
# ANALYSIS 11: RMSE for Baseline Prediction
# ==========================================
print("\n" + "=" * 60)
print("ANALYSIS 11: RMSE for Baseline Delivery Time Prediction")
print("=" * 60)

# ---------- Split Data and Calculate Baseline ----------
delivery_complete = delivery['delivery_time_minutes'].dropna()
train_size_baseline = int(0.8 * len(delivery_complete))
train_baseline = delivery_complete.iloc[:train_size_baseline]
test_baseline = delivery_complete.iloc[train_size_baseline:]

# Baseline prediction (mean of training data)
baseline_prediction = train_baseline.mean()

# Calculate RMSE
baseline_rmse = np.sqrt(mean_squared_error(test_baseline,
                                            np.full(len(test_baseline), baseline_prediction)))

print(f"Baseline Prediction (Mean): {baseline_prediction:.2f} minutes")
print(f"Test samples: {len(test_baseline)}")
print(f"RMSE: {baseline_rmse:.2f} minutes")

# ==========================================
# ANALYSIS 12: F-Statistic for Variance Equality
# ==========================================
print("\n" + "=" * 60)
print("ANALYSIS 12: F-Statistic for Variance Equality Test")
print("=" * 60)

# ---------- Select Premium and Regular Segments ----------
premium_values = customers[customers['customer_segment'] == 'Premium']['avg_order_value']
regular_values = customers[customers['customer_segment'] == 'Regular']['avg_order_value']

# ---------- Calculate Between-Group Variance ----------
grand_mean = np.mean([premium_values.mean(), regular_values.mean()])
n_premium = len(premium_values)
n_regular = len(regular_values)

between_group_var = ((n_premium * (premium_values.mean() - grand_mean)**2 +
                      n_regular * (regular_values.mean() - grand_mean)**2) / 1)

# ---------- Calculate Within-Group Variance ----------
within_group_var = ((n_premium - 1) * premium_values.var() +
                    (n_regular - 1) * regular_values.var()) / (n_premium + n_regular - 2)

# ---------- Calculate F-Statistic ----------
f_statistic = between_group_var / within_group_var

print(f"Premium segment: n={n_premium}, mean=${premium_values.mean():.2f}")
print(f"Regular segment: n={n_regular}, mean=${regular_values.mean():.2f}")
print(f"F-Statistic: {f_statistic:.2f}")

# ==========================================
# FINAL INSIGHTS
# ==========================================
print("\n" + "=" * 60)
print("KEY FINDINGS SUMMARY")
print("=" * 60)
print(f"1. Ridge Total Orders Coefficient: {total_orders_coef:.4f}")
print(f"2. Lasso Non-Zero Features: {non_zero_coefs}")
print(f"3. SVR MAPE: {mape_svr:.2f}%")
print(f"4. Exponential Smoothing RMSE: {rmse_es:.2f}")
print(f"5. Logistic Optimal Threshold: {optimal_threshold:.4f}")
print(f"6. Chi-Square Statistic: {chi2:.2f}")
print(f"7. Z-Score Outliers: {outlier_count}")
print(f"8. MAD: {mad_value:.2f}")
print(f"9. IQR Ratio: {iqr_ratio:.2f}")
print(f"10. Maximum VIF: {max_vif:.2f}")
print(f"11. Baseline RMSE: {baseline_rmse:.2f}")
print(f"12. F-Statistic: {f_statistic:.2f}")

# ==========================================
# ANSWER TO OPEN-ENDED QUESTION
# ==========================================
print("\n" + "=" * 60)
print("MODELING APPROACH RECOMMENDATION")
print("=" * 60)
print("Based on ridge regularization strength and lasso feature selection:")
print("Ridge regression with optimal alpha balances bias-variance tradeoff for continuous prediction,")
print("while lasso's L1 penalty enables automatic feature selection identifying the most influential")
print("predictors, making lasso superior when interpretability and dimensionality reduction are priorities,")
print("whereas ridge excels for multicollinearity scenarios with correlated predictors requiring all features.")
print("\n" + "=" * 60)
