# ==========================================
# Integrated Blinkit Analytics Script
# ==========================================
# Requirements: pandas, numpy, matplotlib, scipy, scikit-learn, statsmodels, openpyxl
# Input files: blinkit_orders.xlsx, blinkit_products.xlsx, blinkit_marketing_performance.xlsx
# Output files: arima_forecast.png, knn_predictions.png, feature_importance.png,
#               confusion_matrix.png, granger_causality.png, pca_biplot.png,
#               pls_components.png, bootstrap_histogram.png
# ==========================================

import pandas as pd
import numpy as np
import matplotlib.pyplot as plt
import seaborn as sns
from sklearn.model_selection import train_test_split
from sklearn.preprocessing import StandardScaler
from sklearn.neighbors import KNeighborsRegressor
from sklearn.tree import DecisionTreeRegressor
from sklearn.naive_bayes import GaussianNB
from sklearn.decomposition import PCA
from sklearn.linear_model import LinearRegression
from sklearn.cross_decomposition import PLSRegression
from sklearn.metrics import r2_score, accuracy_score, confusion_matrix
from statsmodels.tsa.arima.model import ARIMA
from statsmodels.tsa.stattools import grangercausalitytests
import warnings
warnings.filterwarnings('ignore')

# Set plot style
sns.set_style("whitegrid")
plt.rcParams['figure.figsize'] = (10, 6)

# ---------- Load Excel Files ----------
print("Loading datasets...")
orders_df = pd.read_excel('blinkit_orders.xlsx')
products_df = pd.read_excel('blinkit_products.xlsx')
marketing_df = pd.read_excel('blinkit_marketing_performance.xlsx')

print(f"Orders: {len(orders_df)} records")
print(f"Products: {len(products_df)} records")
print(f"Marketing: {len(marketing_df)} records")
print()

# ==========================================
# ANALYSIS 1: ARIMA Time Series Forecasting
# ==========================================
print("=" * 60)
print("ANALYSIS 1: ARIMA Revenue Forecasting")
print("=" * 60)

# ---------- Aggregate Daily Revenue ----------
marketing_df['date'] = pd.to_datetime(marketing_df['date'])
daily_revenue = marketing_df.groupby('date')['revenue_generated'].sum().reset_index()
daily_revenue = daily_revenue.sort_values('date').reset_index(drop=True)
daily_revenue.columns = ['date', 'revenue']

print(f"Time series length: {len(daily_revenue)} days")

# ---------- Train-Test Split (80-20) ----------
split_idx = int(len(daily_revenue) * 0.8)
train_revenue = daily_revenue.iloc[:split_idx]
test_revenue = daily_revenue.iloc[split_idx:]

# ---------- Fit ARIMA(2,1,2) Model ----------
print("Fitting ARIMA(2,1,2) model...")
model = ARIMA(train_revenue['revenue'], order=(2, 1, 2))
fitted_model = model.fit()

# ---------- Generate Forecasts ----------
forecast_values = []
for i in range(len(test_revenue)):
    forecast = fitted_model.forecast(steps=1)
    forecast_values.append(forecast.iloc[0])

# ---------- Calculate MAE ----------
actual_test = test_revenue['revenue'].values
mae = np.mean(np.abs(actual_test - forecast_values))
print(f"Mean Absolute Error: ${mae:.2f}")

# ---------- Plot Actual vs Forecast ----------
plt.figure(figsize=(12, 6))
plt.plot(train_revenue['date'], train_revenue['revenue'], label='Training Data', color='blue', alpha=0.7)
plt.plot(test_revenue['date'], actual_test, label='Actual Revenue', color='green', linewidth=2)
plt.plot(test_revenue['date'], forecast_values, label='Forecasted Revenue', color='red', linestyle='--', linewidth=2)
plt.xlabel('Date')
plt.ylabel('Daily Revenue ($)')
plt.title('ARIMA(2,1,2) Revenue Forecasting')
plt.legend()
plt.xticks(rotation=45)
plt.tight_layout()
plt.savefig('arima_forecast.png', dpi=300, bbox_inches='tight')
plt.close()
print("Saved: arima_forecast.png")
print()

# ==========================================
# ANALYSIS 2: KNN Regression for Order Total
# ==========================================
print("=" * 60)
print("ANALYSIS 2: K-Nearest Neighbors Regression")
print("=" * 60)

# ---------- Prepare Orders Data ----------
# Create customer-level features from orders
customer_features = orders_df.groupby('customer_id').agg({
    'order_total': 'mean',
    'order_id': 'count'
}).reset_index()
customer_features.columns = ['customer_id', 'avg_order_total', 'order_count']

# Merge orders with customer features
orders_merged = orders_df.merge(customer_features, on='customer_id', how='left')

# Create proxy features (simplified approach)
# Feature 1: Average price proxy (using order_total as indicator)
# Feature 2: Count of categories proxy (using customer order frequency)
# Feature 3: Margin proxy (normalized order total)

orders_knn = orders_merged[['order_total', 'avg_order_total', 'order_count']].copy()
orders_knn['normalized_total'] = orders_knn['order_total'] / (orders_knn['order_total'].mean() + 1)
orders_knn = orders_knn.dropna()

# ---------- Prepare Features and Target ----------
X_knn = orders_knn[['avg_order_total', 'order_count', 'normalized_total']]
y_knn = orders_knn['order_total']

# ---------- Train-Test Split ----------
X_train, X_test, y_train, y_test = train_test_split(X_knn, y_knn, test_size=0.2, random_state=42)

# ---------- Standardize Features ----------
scaler_knn = StandardScaler()
X_train_scaled = scaler_knn.fit_transform(X_train)
X_test_scaled = scaler_knn.transform(X_test)

# ---------- Fit KNN Regressor (k=7) ----------
print("Fitting KNN regressor with k=7...")
knn = KNeighborsRegressor(n_neighbors=7, metric='euclidean')
knn.fit(X_train_scaled, y_train)

# ---------- Predict and Calculate R-squared ----------
y_pred_knn = knn.predict(X_test_scaled)
r2_knn = r2_score(y_test, y_pred_knn)
print(f"R-squared: {r2_knn:.4f}")

# ---------- Scatter Plot: Actual vs Predicted ----------
plt.figure(figsize=(10, 6))
plt.scatter(y_test, y_pred_knn, alpha=0.5, color='blue')
plt.plot([y_test.min(), y_test.max()], [y_test.min(), y_test.max()], 'r--', linewidth=2)
plt.xlabel('Actual Order Total ($)')
plt.ylabel('Predicted Order Total ($)')
plt.title(f'KNN Regression: Actual vs Predicted (R² = {r2_knn:.4f})')
plt.tight_layout()
plt.savefig('knn_predictions.png', dpi=300, bbox_inches='tight')
plt.close()
print("Saved: knn_predictions.png")
print()

# ==========================================
# ANALYSIS 3: Decision Tree - Price Prediction
# ==========================================
print("=" * 60)
print("ANALYSIS 3: Decision Tree Regression with Feature Importance")
print("=" * 60)

# ---------- Prepare Product Features ----------
products_dt = products_df[['price', 'margin_percentage', 'shelf_life_days',
                            'min_stock_level', 'max_stock_level']].copy()
products_dt = products_dt.dropna()

# ---------- Features and Target ----------
X_dt = products_dt[['margin_percentage', 'shelf_life_days', 'min_stock_level', 'max_stock_level']]
y_dt = products_dt['price']

# ---------- Fit Decision Tree ----------
print("Fitting Decision Tree (max_depth=6, min_samples_split=10)...")
dt_regressor = DecisionTreeRegressor(max_depth=6, min_samples_split=10, random_state=42)
dt_regressor.fit(X_dt, y_dt)

# ---------- Extract Feature Importances ----------
feature_importances = dt_regressor.feature_importances_
feature_names = X_dt.columns.tolist()
importance_df = pd.DataFrame({
    'feature': feature_names,
    'importance': feature_importances
}).sort_values('importance', ascending=True)

# ---------- Identify Most Important Feature ----------
most_important_feature = importance_df.iloc[-1]['feature']
print(f"Most important feature: {most_important_feature}")

# ---------- Horizontal Bar Chart ----------
plt.figure(figsize=(10, 6))
plt.barh(importance_df['feature'], importance_df['importance'], color='steelblue')
plt.xlabel('Importance Score')
plt.ylabel('Feature Names')
plt.title('Decision Tree: Feature Importance for Price Prediction')
plt.xlim(0, 1)
plt.tight_layout()
plt.savefig('feature_importance.png', dpi=300, bbox_inches='tight')
plt.close()
print("Saved: feature_importance.png")
print()

# ==========================================
# ANALYSIS 4: Naive Bayes - Delivery Prediction
# ==========================================
print("=" * 60)
print("ANALYSIS 4: Naive Bayes Classification")
print("=" * 60)

# ---------- Prepare Delivery Data ----------
orders_nb = orders_df.copy()

# Create binary target
orders_nb['on_time'] = (orders_nb['delivery_status'] == 'On Time').astype(int)

# Extract hours from delivery times
orders_nb['promised_delivery_time'] = pd.to_datetime(orders_nb['promised_delivery_time'])
orders_nb['actual_delivery_time'] = pd.to_datetime(orders_nb['actual_delivery_time'])
orders_nb['promised_hour'] = orders_nb['promised_delivery_time'].dt.hour

# Customer order frequency
customer_freq = orders_nb.groupby('customer_id').size().reset_index(name='customer_order_frequency')
orders_nb = orders_nb.merge(customer_freq, on='customer_id', how='left')

# ---------- Select Features ----------
orders_nb_clean = orders_nb[['on_time', 'promised_hour', 'order_total', 'customer_order_frequency']].dropna()

X_nb = orders_nb_clean[['promised_hour', 'order_total', 'customer_order_frequency']]
y_nb = orders_nb_clean['on_time']

# ---------- Train-Test Split ----------
X_train_nb, X_test_nb, y_train_nb, y_test_nb = train_test_split(X_nb, y_nb, test_size=0.2, random_state=42)

# ---------- Fit Gaussian Naive Bayes ----------
print("Fitting Gaussian Naive Bayes classifier...")
gnb = GaussianNB()
gnb.fit(X_train_nb, y_train_nb)

# ---------- Predict and Calculate Accuracy ----------
y_pred_nb = gnb.predict(X_test_nb)
accuracy_nb = accuracy_score(y_test_nb, y_pred_nb) * 100
print(f"Classification Accuracy: {accuracy_nb:.2f}%")

# ---------- Confusion Matrix Heatmap ----------
cm = confusion_matrix(y_test_nb, y_pred_nb)
plt.figure(figsize=(8, 6))
sns.heatmap(cm, annot=True, fmt='d', cmap='Blues', cbar=True,
            xticklabels=['Late', 'On Time'], yticklabels=['Late', 'On Time'])
plt.xlabel('Predicted Class')
plt.ylabel('Actual Class')
plt.title(f'Naive Bayes: Confusion Matrix (Accuracy = {accuracy_nb:.2f}%)')
plt.tight_layout()
plt.savefig('confusion_matrix.png', dpi=300, bbox_inches='tight')
plt.close()
print("Saved: confusion_matrix.png")
print()

# ==========================================
# ANALYSIS 5: Granger Causality Test
# ==========================================
print("=" * 60)
print("ANALYSIS 5: Granger Causality Test")
print("=" * 60)

# ---------- Aggregate Daily Spend and Revenue ----------
daily_marketing = marketing_df.groupby('date').agg({
    'spend': 'sum',
    'revenue_generated': 'sum'
}).reset_index()
daily_marketing = daily_marketing.sort_values('date').reset_index(drop=True)
daily_marketing.columns = ['date', 'daily_spend', 'daily_revenue']

# ---------- Prepare Time Series for Granger Test ----------
granger_data = daily_marketing[['daily_spend', 'daily_revenue']].values

# ---------- Perform Granger Causality Test (lag=5) ----------
print("Testing if daily_spend Granger-causes daily_revenue (lag=5)...")
gc_results = grangercausalitytests(granger_data[:, [1, 0]], maxlag=5, verbose=False)

# Extract F-statistic for lag 5
f_statistic = gc_results[5][0]['ssr_ftest'][0]
print(f"F-statistic (lag 5): {f_statistic:.2f}")

# ---------- Dual-Axis Line Plot ----------
fig, ax1 = plt.subplots(figsize=(12, 6))

color = 'tab:blue'
ax1.set_xlabel('Date')
ax1.set_ylabel('Daily Spend ($)', color=color)
ax1.plot(daily_marketing['date'], daily_marketing['daily_spend'], color=color, label='Spend')
ax1.tick_params(axis='y', labelcolor=color)
ax1.tick_params(axis='x', rotation=45)

ax2 = ax1.twinx()
color = 'tab:red'
ax2.set_ylabel('Daily Revenue ($)', color=color)
ax2.plot(daily_marketing['date'], daily_marketing['daily_revenue'], color=color, label='Revenue')
ax2.tick_params(axis='y', labelcolor=color)

plt.title(f'Granger Causality: Spend vs Revenue (F-stat = {f_statistic:.2f})')
fig.tight_layout()
plt.savefig('granger_causality.png', dpi=300, bbox_inches='tight')
plt.close()
print("Saved: granger_causality.png")
print()

# ==========================================
# ANALYSIS 6: Principal Component Regression
# ==========================================
print("=" * 60)
print("ANALYSIS 6: Principal Component Regression")
print("=" * 60)

# ---------- Prepare Product Data for PCR ----------
products_pcr = products_df[['mrp', 'price', 'margin_percentage', 'shelf_life_days',
                             'min_stock_level', 'max_stock_level']].copy()
products_pcr = products_pcr.dropna()

# ---------- Features and Target ----------
X_pcr = products_pcr[['price', 'margin_percentage', 'shelf_life_days', 'min_stock_level', 'max_stock_level']]
y_pcr = products_pcr['mrp']

# ---------- Standardize Features ----------
scaler_pcr = StandardScaler()
X_pcr_scaled = scaler_pcr.fit_transform(X_pcr)

# ---------- Fit PCA (3 components) ----------
print("Fitting PCA with 3 components...")
pca = PCA(n_components=3)
X_pca = pca.fit_transform(X_pcr_scaled)

# ---------- Fit Linear Regression on PC Scores ----------
lr_pcr = LinearRegression()
lr_pcr.fit(X_pca, y_pcr)

# ---------- Extract PC1 Variance Explained ----------
pc1_variance = pca.explained_variance_ratio_[0] * 100
print(f"PC1 Variance Explained: {pc1_variance:.2f}%")

# ---------- PCA Biplot ----------
plt.figure(figsize=(10, 8))
plt.scatter(X_pca[:, 0], X_pca[:, 1], alpha=0.5, c='blue', s=30)

# Plot loadings as arrows
loadings = pca.components_.T * np.sqrt(pca.explained_variance_)
feature_names_pcr = X_pcr.columns
for i, feature in enumerate(feature_names_pcr):
    plt.arrow(0, 0, loadings[i, 0]*3, loadings[i, 1]*3,
              head_width=0.3, head_length=0.3, fc='red', ec='red', alpha=0.7)
    plt.text(loadings[i, 0]*3.2, loadings[i, 1]*3.2, feature,
             fontsize=10, ha='center', va='center', bbox=dict(boxstyle='round', facecolor='wheat', alpha=0.5))

plt.xlabel(f'First Principal Component ({pca.explained_variance_ratio_[0]*100:.1f}%)')
plt.ylabel(f'Second Principal Component ({pca.explained_variance_ratio_[1]*100:.1f}%)')
plt.title('PCA Biplot: Product Features')
plt.grid(True, alpha=0.3)
plt.tight_layout()
plt.savefig('pca_biplot.png', dpi=300, bbox_inches='tight')
plt.close()
print("Saved: pca_biplot.png")
print()

# ==========================================
# ANALYSIS 7: Partial Least Squares Regression
# ==========================================
print("=" * 60)
print("ANALYSIS 7: Partial Least Squares Regression")
print("=" * 60)

# ---------- Merge Orders with Marketing ----------
orders_df['order_date'] = pd.to_datetime(orders_df['order_date'])
marketing_df['date'] = pd.to_datetime(marketing_df['date'])

# Aggregate marketing by date
marketing_agg = marketing_df.groupby('date').agg({
    'impressions': 'sum',
    'clicks': 'sum',
    'conversions': 'sum',
    'roas': 'mean'
}).reset_index()

# Ensure both dataframes have normalized dates
marketing_agg['date'] = pd.to_datetime(marketing_agg['date']).dt.date
orders_date_df = orders_df.copy()
orders_date_df['order_date'] = pd.to_datetime(orders_date_df['order_date']).dt.date

# Merge on date
orders_marketing = orders_date_df.merge(marketing_agg, left_on='order_date', right_on='date', how='inner')

# If no direct matches, use aggregated approach
if len(orders_marketing) == 0:
    print("No direct date matches found. Using aggregated time-based approach...")
    # Aggregate orders by date
    orders_agg = orders_date_df.groupby('order_date').agg({
        'order_total': 'mean'
    }).reset_index()

    # Merge aggregated data
    orders_marketing = orders_agg.merge(marketing_agg, left_on='order_date', right_on='date', how='inner')

# ---------- Prepare PLS Data ----------
if len(orders_marketing) > 0:
    pls_data = orders_marketing[['order_total', 'impressions', 'clicks', 'conversions', 'roas']].dropna()
else:
    # Fallback: create synthetic relationship using available data
    print("Using alternative approach: sampling marketing data with order totals...")
    np.random.seed(42)
    sample_size = min(1000, len(marketing_df))
    marketing_sample = marketing_df.sample(n=sample_size, random_state=42)
    orders_sample = orders_df.sample(n=sample_size, random_state=42)

    pls_data = pd.DataFrame({
        'order_total': orders_sample['order_total'].values,
        'impressions': marketing_sample['impressions'].values,
        'clicks': marketing_sample['clicks'].values,
        'conversions': marketing_sample['conversions'].values,
        'roas': marketing_sample['roas'].values
    })

X_pls = pls_data[['impressions', 'clicks', 'conversions', 'roas']]
y_pls = pls_data['order_total']

# ---------- Fit PLS Regression (2 components) ----------
print("Fitting PLS regression with 2 components...")
pls = PLSRegression(n_components=2)
pls.fit(X_pls, y_pls)

# ---------- Extract Coefficient for Impressions ----------
impressions_coef = pls.coef_[0, 0]
print(f"Impressions coefficient: {impressions_coef:.4f}")

# ---------- PLS Component Scores Plot ----------
X_pls_scores = pls.transform(X_pls)
plt.figure(figsize=(10, 6))
plt.scatter(X_pls_scores[:, 0], y_pls, alpha=0.5, color='purple')
plt.xlabel('First PLS Component Scores')
plt.ylabel('Order Total ($)')
plt.title(f'PLS Regression: Component Scores vs Order Total')
plt.tight_layout()
plt.savefig('pls_components.png', dpi=300, bbox_inches='tight')
plt.close()
print("Saved: pls_components.png")
print()

# ==========================================
# ANALYSIS 8: Bootstrap Confidence Interval
# ==========================================
print("=" * 60)
print("ANALYSIS 8: Bootstrap Confidence Interval")
print("=" * 60)

# ---------- Extract Margin Percentage ----------
margins = products_df['margin_percentage'].dropna().values

# ---------- Generate 1000 Bootstrap Samples ----------
print("Generating 1000 bootstrap samples...")
n_bootstrap = 1000
bootstrap_means = []

np.random.seed(42)
for i in range(n_bootstrap):
    sample = np.random.choice(margins, size=len(margins), replace=True)
    bootstrap_means.append(np.mean(sample))

# ---------- Construct 95% CI using Percentile Method ----------
ci_lower = np.percentile(bootstrap_means, 2.5)
ci_upper = np.percentile(bootstrap_means, 97.5)
ci_width = ci_upper - ci_lower
print(f"95% CI: [{ci_lower:.2f}, {ci_upper:.2f}]")
print(f"Confidence Interval Width: {ci_width:.2f}")

# ---------- Histogram of Bootstrap Means ----------
plt.figure(figsize=(10, 6))
plt.hist(bootstrap_means, bins=50, color='skyblue', edgecolor='black', alpha=0.7)
plt.axvline(ci_lower, color='red', linestyle='--', linewidth=2, label=f'2.5% ({ci_lower:.2f})')
plt.axvline(ci_upper, color='red', linestyle='--', linewidth=2, label=f'97.5% ({ci_upper:.2f})')
plt.xlabel('Bootstrap Sample Means')
plt.ylabel('Frequency Count')
plt.title(f'Bootstrap Distribution: Mean Margin Percentage (95% CI width = {ci_width:.2f})')
plt.legend()
plt.tight_layout()
plt.savefig('bootstrap_histogram.png', dpi=300, bbox_inches='tight')
plt.close()
print("Saved: bootstrap_histogram.png")
print()

# ==========================================
# ANALYSIS 9: CTR Variance by Channel
# ==========================================
print("=" * 60)
print("ANALYSIS 9: Click-Through Rate Variance Comparison")
print("=" * 60)

# ---------- Calculate CTR for Each Campaign ----------
marketing_df['ctr'] = (marketing_df['clicks'] / marketing_df['impressions']) * 100

# ---------- Group by Channel and Calculate Variance ----------
channel_ctr_variance = marketing_df.groupby('channel')['ctr'].var().reset_index()
channel_ctr_variance.columns = ['channel', 'ctr_variance']

# ---------- Identify Channel with Maximum Variance ----------
max_variance_row = channel_ctr_variance.loc[channel_ctr_variance['ctr_variance'].idxmax()]
max_variance_channel = max_variance_row['channel']
max_variance_value = max_variance_row['ctr_variance']

print(f"Channel with maximum CTR variance: {max_variance_channel}")
print(f"Maximum CTR variance: {max_variance_value:.2f} (percentage squared)")
print()

# ==========================================
# ANALYSIS 10: Delivery Time Deviation
# ==========================================
print("=" * 60)
print("ANALYSIS 10: Delivery Time Deviation Analysis")
print("=" * 60)

# ---------- Calculate Delivery Deviation in Minutes ----------
orders_df['promised_delivery_time'] = pd.to_datetime(orders_df['promised_delivery_time'])
orders_df['actual_delivery_time'] = pd.to_datetime(orders_df['actual_delivery_time'])
orders_df['delivery_deviation'] = (orders_df['actual_delivery_time'] -
                                    orders_df['promised_delivery_time']).dt.total_seconds() / 60

# ---------- Group by Delivery Status ----------
status_deviation = orders_df.groupby('delivery_status')['delivery_deviation'].apply(
    lambda x: np.mean(np.abs(x))
).reset_index()
status_deviation.columns = ['delivery_status', 'mean_absolute_deviation']

# ---------- Calculate Range ----------
deviation_range = status_deviation['mean_absolute_deviation'].max() - status_deviation['mean_absolute_deviation'].min()
print("Mean Absolute Deviation by Status:")
print(status_deviation)
print(f"Range of mean deviations: {deviation_range:.2f} minutes")
print()

# ==========================================
# ANALYSIS 11: Product Turnover Rate
# ==========================================
print("=" * 60)
print("ANALYSIS 11: Product Turnover Rate by Category")
print("=" * 60)

# ---------- Aggregate Orders by Category ----------
# First, we need to link orders to products (simplified approach using aggregation)
# Since we don't have product_id in orders, we'll use a proxy approach
# Aggregate product data by category
category_stock = products_df.groupby('category').agg({
    'max_stock_level': 'mean',
    'product_id': 'count'
}).reset_index()
category_stock.columns = ['category', 'avg_max_stock', 'product_count']

# Assume uniform distribution of orders across products (simplified)
# Calculate turnover as total orders / average stock
total_orders = len(orders_df)
category_stock['order_proxy'] = total_orders / len(products_df) * category_stock['product_count']
category_stock['turnover_ratio'] = category_stock['order_proxy'] / category_stock['avg_max_stock']

# ---------- Identify Category with Maximum Turnover ----------
max_turnover_row = category_stock.loc[category_stock['turnover_ratio'].idxmax()]
max_turnover_category = max_turnover_row['category']
max_turnover_value = max_turnover_row['turnover_ratio']

print("Turnover Ratio by Category:")
print(category_stock[['category', 'turnover_ratio']])
print(f"\nCategory with maximum turnover: {max_turnover_category}")
print(f"Maximum turnover ratio: {max_turnover_value:.2f}")
print()

# ==========================================
# KEY OUTPUTS SUMMARY
# ==========================================
print("\n" + "=" * 60)
print("KEY OUTPUTS SUMMARY")
print("=" * 60)
print(f"1. ARIMA MAE: ${mae:.2f}")
print(f"2. KNN R-squared: {r2_knn:.4f}")
print(f"3. Most Important Feature: {most_important_feature}")
print(f"4. Naive Bayes Accuracy: {accuracy_nb:.2f}%")
print(f"5. Granger F-statistic: {f_statistic:.2f}")
print(f"6. PC1 Variance Explained: {pc1_variance:.2f}%")
print(f"7. PLS Impressions Coefficient: {impressions_coef:.4f}")
print(f"8. Bootstrap CI Width: {ci_width:.2f}")
print(f"9. Max CTR Variance: {max_variance_value:.2f}")
print(f"10. Delivery Deviation Range: {deviation_range:.2f} minutes")
print(f"11. Max Turnover Ratio: {max_turnover_value:.2f}")
print("=" * 60)
print("\nAll visualizations saved successfully!")
print("Generated files:")
print("  - arima_forecast.png")
print("  - knn_predictions.png")
print("  - feature_importance.png")
print("  - confusion_matrix.png")
print("  - granger_causality.png")
print("  - pca_biplot.png")
print("  - pls_components.png")
print("  - bootstrap_histogram.png")
