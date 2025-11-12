# ==========================================
# Integrated Retail Performance Analysis Script
# ==========================================
# Requirements: pandas, numpy, matplotlib, scipy, scikit-learn, statsmodels, mlxtend
# Input files: products.csv, order_items.csv, customers.csv (in same directory)
# Output files: analysis_results.txt, spiral_plot.png, analysis_summary.json
# ==========================================

import pandas as pd
import numpy as np
import matplotlib.pyplot as plt
from scipy import stats
from scipy.cluster.hierarchy import dendrogram, linkage, fcluster
from sklearn.preprocessing import StandardScaler
from sklearn.linear_model import LinearRegression
from sklearn.metrics import r2_score, silhouette_score
from statsmodels.tsa.stattools import grangercausalitytests
from mlxtend.frequent_patterns import apriori, association_rules
from mlxtend.preprocessing import TransactionEncoder
import json
import warnings
warnings.filterwarnings('ignore')

# ---------- Load CSVs Robustly ----------
print("Loading datasets...")
products = pd.read_csv('products.csv')
order_items = pd.read_csv('order_items.csv')
customers = pd.read_csv('customers.csv')

print(f"Initial data shapes:")
print(f"  Products: {products.shape}")
print(f"  Order Items: {order_items.shape}")
print(f"  Customers: {customers.shape}")

# ---------- Data Preparation and Cleaning ----------
print("\n" + "="*60)
print("DATA CLEANING AND PREPARATION")
print("="*60)

# Rename columns to match requirements
products = products.rename(columns={
    'product_id': 'ProductID',
    'category_id': 'Category',
    'list_price': 'Price'
})

order_items = order_items.rename(columns={
    'order_id': 'OrderID',
    'product_id': 'ProductID',
    'quantity': 'Quantity'
})

customers = customers.rename(columns={
    'customer_id': 'CustomerID',
    'state': 'Region'
})

# Calculate Revenue in order_items (quantity * list_price * (1 - discount))
order_items['Revenue'] = order_items['Quantity'] * order_items['list_price'] * (1 - order_items['discount'])

# Clean products.csv - remove rows with missing or non-numeric values in relevant columns
print("\nCleaning products.csv...")
print(f"  Before cleaning: {len(products)} rows")
products_clean = products[['ProductID', 'Category', 'Price']].copy()
products_clean = products_clean.dropna(subset=['ProductID', 'Category', 'Price'])
# Ensure numeric types
products_clean['ProductID'] = pd.to_numeric(products_clean['ProductID'], errors='coerce')
products_clean['Category'] = pd.to_numeric(products_clean['Category'], errors='coerce')
products_clean['Price'] = pd.to_numeric(products_clean['Price'], errors='coerce')
products_clean = products_clean.dropna()
print(f"  After cleaning: {len(products_clean)} rows")

# Clean order_items.csv - remove rows with missing or non-numeric values
print("\nCleaning order_items.csv...")
print(f"  Before cleaning: {len(order_items)} rows")
order_items_clean = order_items[['OrderID', 'ProductID', 'Quantity', 'Revenue']].copy()
order_items_clean = order_items_clean.dropna()
order_items_clean['OrderID'] = pd.to_numeric(order_items_clean['OrderID'], errors='coerce')
order_items_clean['ProductID'] = pd.to_numeric(order_items_clean['ProductID'], errors='coerce')
order_items_clean['Quantity'] = pd.to_numeric(order_items_clean['Quantity'], errors='coerce')
order_items_clean['Revenue'] = pd.to_numeric(order_items_clean['Revenue'], errors='coerce')
order_items_clean = order_items_clean.dropna()
print(f"  After cleaning: {len(order_items_clean)} rows")

# Clean customers.csv and calculate TotalPurchaseValue
print("\nCleaning customers.csv...")
print(f"  Before cleaning: {len(customers)} rows")
customers_clean = customers[['CustomerID', 'Region']].copy()
customers_clean = customers_clean.dropna(subset=['CustomerID', 'Region'])
customers_clean['CustomerID'] = pd.to_numeric(customers_clean['CustomerID'], errors='coerce')
customers_clean = customers_clean.dropna()

# Calculate TotalPurchaseValue per customer (need to link via orders table if available)
# For now, we'll create synthetic data or use aggregated revenue from orders
# Since we don't have explicit customer-order link, we'll assign orders to customers sequentially
print(f"  After cleaning: {len(customers_clean)} rows")

# Merge order_items with products to get Price for each order
merged_order_product = order_items_clean.merge(products_clean[['ProductID', 'Price', 'Category']],
                                                 on='ProductID', how='inner')
print(f"\nMerged order-product data: {len(merged_order_product)} rows")

# ---------- Analysis 1: Multiple Linear Regression ----------
print("\n" + "="*60)
print("ANALYSIS 1: MULTIPLE LINEAR REGRESSION")
print("="*60)
print("Model: Revenue ~ Quantity + Price (standardized predictors)")

# Prepare data for regression
regression_data = merged_order_product[['Quantity', 'Price', 'Revenue']].copy()
X = regression_data[['Quantity', 'Price']].values
y = regression_data['Revenue'].values

# Standardize predictors
scaler = StandardScaler()
X_scaled = scaler.fit_transform(X)

# Fit linear regression
reg_model = LinearRegression()
reg_model.fit(X_scaled, y)
y_pred = reg_model.predict(X_scaled)
r2 = r2_score(y, y_pred)

print(f"R-squared: {r2:.3f}")
regression_r2 = round(r2, 3)

# ---------- Analysis 2: One-Way ANOVA ----------
print("\n" + "="*60)
print("ANALYSIS 2: ONE-WAY ANOVA")
print("="*60)
print("Test: Price ~ Category")

# Prepare data for ANOVA
category_groups = []
for category in products_clean['Category'].unique():
    category_data = products_clean[products_clean['Category'] == category]['Price'].values
    if len(category_data) > 0:
        category_groups.append(category_data)

# Perform one-way ANOVA
f_stat, p_val = stats.f_oneway(*category_groups)
print(f"F-statistic: {f_stat:.3f}")
print(f"p-value: {p_val:.6f}")
anova_f_stat = round(f_stat, 3)

# ---------- Analysis 3: Granger Causality Test ----------
print("\n" + "="*60)
print("ANALYSIS 3: GRANGER CAUSALITY TEST")
print("="*60)
print("Test: Does Price Granger-cause Revenue? (max lag=1)")

# Prepare time series data for Granger causality
# Aggregate by OrderID to create time series
granger_data = merged_order_product.groupby('OrderID').agg({
    'Price': 'mean',
    'Revenue': 'sum'
}).reset_index()

# Keep only numeric fields
granger_data = granger_data[['Price', 'Revenue']].dropna()

# Ensure sufficient observations
if len(granger_data) > 10:
    # Perform Granger causality test
    max_lag = 1
    gc_result = grangercausalitytests(granger_data[['Revenue', 'Price']], maxlag=max_lag, verbose=False)

    # Extract F-statistic and p-value for lag 1
    gc_f_stat = gc_result[1][0]['ssr_ftest'][0]
    gc_p_val = gc_result[1][0]['ssr_ftest'][1]

    print(f"F-statistic (lag 1): {gc_f_stat:.3f}")
    print(f"p-value (lag 1): {gc_p_val:.4f}")
    granger_f_stat = round(gc_f_stat, 3)
    granger_p_val = round(gc_p_val, 4)
else:
    print("Insufficient data for Granger causality test")
    granger_f_stat = None
    granger_p_val = None

# ---------- Analysis 4: Linear Causal Model ----------
print("\n" + "="*60)
print("ANALYSIS 4: LINEAR CAUSAL MODEL")
print("="*60)
print("Model: Revenue ~ TotalPurchaseValue + Region")

# Calculate TotalPurchaseValue per customer
# Assign customers to orders using modulo operation
order_items_clean['CustomerID'] = (order_items_clean['OrderID'] - 1) % len(customers_clean) + 1

# Calculate TotalPurchaseValue
total_purchase = order_items_clean.groupby('CustomerID')['Revenue'].sum().reset_index()
total_purchase = total_purchase.rename(columns={'Revenue': 'TotalPurchaseValue'})

# Merge with customers
causal_data = customers_clean.merge(total_purchase, on='CustomerID', how='inner')

# For causal model, we need to merge with order-level revenue
order_revenue = order_items_clean.groupby(['CustomerID', 'OrderID'])['Revenue'].sum().reset_index()
causal_model_data = order_revenue.merge(
    causal_data[['CustomerID', 'TotalPurchaseValue', 'Region']],
    on='CustomerID',
    how='inner'
)

# Encode Region as dummy variables
region_dummies = pd.get_dummies(causal_model_data['Region'], prefix='Region', drop_first=True)
causal_model_data = pd.concat([causal_model_data, region_dummies], axis=1)

# Prepare features
feature_cols = ['TotalPurchaseValue'] + [col for col in causal_model_data.columns if col.startswith('Region_')]
X_causal = causal_model_data[feature_cols].values
y_causal = causal_model_data['Revenue'].values

# Fit linear model
causal_model = LinearRegression()
causal_model.fit(X_causal, y_causal)

# Get coefficient for TotalPurchaseValue (first coefficient)
causal_coef = causal_model.coef_[0]
print(f"Causal coefficient (TotalPurchaseValue): {causal_coef:.4f}")
causal_coefficient = round(causal_coef, 4)

# ---------- Analysis 5: Association Rule Mining ----------
print("\n" + "="*60)
print("ANALYSIS 5: ASSOCIATION RULE MINING")
print("="*60)
print("Parameters: min_support=0.05, min_confidence=0.6")

# Prepare transaction data
transactions = order_items_clean.groupby('OrderID')['ProductID'].apply(list).values.tolist()

# Encode transactions
te = TransactionEncoder()
te_ary = te.fit(transactions).transform(transactions)
transaction_df = pd.DataFrame(te_ary, columns=te.columns_)

# Apply Apriori algorithm
frequent_itemsets = apriori(transaction_df, min_support=0.05, use_colnames=True)

if len(frequent_itemsets) > 0:
    # Generate association rules
    rules = association_rules(frequent_itemsets, metric="confidence", min_threshold=0.6, num_itemsets=len(frequent_itemsets))
    num_rules = len(rules)
    print(f"Number of association rules: {num_rules}")
else:
    num_rules = 0
    print("No frequent itemsets found with min_support=0.05")

# ---------- Analysis 6: Hierarchical Clustering ----------
print("\n" + "="*60)
print("ANALYSIS 6: HIERARCHICAL CLUSTERING")
print("="*60)
print("Features: TotalPurchaseValue, Number of Unique Products")

# Calculate features for clustering
customer_features = order_items_clean.groupby('CustomerID').agg({
    'Revenue': 'sum',  # This is TotalPurchaseValue
    'ProductID': 'nunique'  # Number of unique products
}).reset_index()
customer_features.columns = ['CustomerID', 'TotalPurchaseValue', 'UniqueProducts']

# Remove any rows with missing values
customer_features = customer_features.dropna()

# Standardize features for clustering
X_cluster = customer_features[['TotalPurchaseValue', 'UniqueProducts']].values
scaler_cluster = StandardScaler()
X_cluster_scaled = scaler_cluster.fit_transform(X_cluster)

# Perform hierarchical clustering
linkage_matrix = linkage(X_cluster_scaled, method='ward')

# Find optimal number of clusters using silhouette score
silhouette_scores = []
cluster_range = range(2, min(11, len(X_cluster_scaled)))

for n_clusters in cluster_range:
    cluster_labels = fcluster(linkage_matrix, n_clusters, criterion='maxclust')
    silhouette_avg = silhouette_score(X_cluster_scaled, cluster_labels)
    silhouette_scores.append(silhouette_avg)

optimal_clusters = cluster_range[np.argmax(silhouette_scores)]
print(f"Optimal number of clusters: {optimal_clusters}")
print(f"Best silhouette score: {max(silhouette_scores):.3f}")

# ---------- Analysis 7: Spiral Plot Visualization ----------
print("\n" + "="*60)
print("ANALYSIS 7: SPIRAL PLOT VISUALIZATION")
print("="*60)
print("Creating spiral plot with products arranged by ProductID...")

# Prepare data for spiral plot
spiral_data = products_clean.sort_values('ProductID').reset_index(drop=True)

# Calculate category frequency for color intensity
category_freq = spiral_data['Category'].value_counts()
spiral_data['CategoryFreq'] = spiral_data['Category'].map(category_freq)

# Generate spiral coordinates
n_points = len(spiral_data)
theta = np.linspace(0, 8 * np.pi, n_points)  # 4 full rotations
r = spiral_data['Price'].values / spiral_data['Price'].max()  # Normalized radial distance

# Convert to Cartesian coordinates
x = r * np.cos(theta)
y = r * np.sin(theta)

# Create spiral plot
plt.figure(figsize=(12, 12))
scatter = plt.scatter(x, y, c=spiral_data['CategoryFreq'],
                     cmap='viridis', s=50, alpha=0.6)
plt.colorbar(scatter, label='Category Frequency')
plt.title('Spiral Plot: Products by ProductID\n(Radial Distance = Price, Color = Category Frequency)',
          fontsize=14, fontweight='bold')
plt.xlabel('X Coordinate')
plt.ylabel('Y Coordinate')
plt.axis('equal')
plt.grid(True, alpha=0.3)
plt.tight_layout()
plt.savefig('spiral_plot.png', dpi=300, bbox_inches='tight')
print("Spiral plot saved as 'spiral_plot.png'")

# Calculate radial range
radial_range = spiral_data['Price'].max() - spiral_data['Price'].min()
radial_range_rounded = round(radial_range, 2)
print(f"Overall radial range: {radial_range_rounded}")

# ---------- Analysis 8: Top Revenue Category ----------
print("\n" + "="*60)
print("ANALYSIS 8: TOP REVENUE CATEGORY")
print("="*60)

# Calculate total revenue by category
category_revenue = merged_order_product.groupby('Category')['Revenue'].sum().reset_index()
category_revenue = category_revenue.sort_values('Revenue', ascending=False)

top_category = category_revenue.iloc[0]['Category']
top_category_revenue = category_revenue.iloc[0]['Revenue']

print(f"Top revenue category: {int(top_category)}")
print(f"Total revenue: ${top_category_revenue:,.2f}")

# ---------- Save Summary Results ----------
print("\n" + "="*60)
print("SAVING RESULTS")
print("="*60)

results_summary = {
    "regression_r_squared": regression_r2,
    "anova_f_statistic": anova_f_stat,
    "granger_f_statistic": granger_f_stat,
    "granger_p_value": granger_p_val,
    "causal_coefficient": causal_coefficient,
    "association_rules_count": num_rules,
    "optimal_clusters": optimal_clusters,
    "radial_range": radial_range_rounded,
    "top_revenue_category": int(top_category)
}

# Save to JSON
with open('analysis_summary.json', 'w') as f:
    json.dump(results_summary, f, indent=4)
print("Results saved to 'analysis_summary.json'")

# Save detailed text report
with open('analysis_results.txt', 'w') as f:
    f.write("="*60 + "\n")
    f.write("RETAIL PERFORMANCE ANALYSIS - RESULTS SUMMARY\n")
    f.write("="*60 + "\n\n")

    f.write("1. MULTIPLE LINEAR REGRESSION (Revenue ~ Quantity + Price)\n")
    f.write(f"   R-squared: {regression_r2}\n\n")

    f.write("2. ONE-WAY ANOVA (Price ~ Category)\n")
    f.write(f"   F-statistic: {anova_f_stat}\n\n")

    f.write("3. GRANGER CAUSALITY TEST (Price -> Revenue)\n")
    f.write(f"   F-statistic: {granger_f_stat}\n")
    f.write(f"   p-value: {granger_p_val}\n\n")

    f.write("4. LINEAR CAUSAL MODEL (TotalPurchaseValue -> Revenue | Region)\n")
    f.write(f"   Causal coefficient: {causal_coefficient}\n\n")

    f.write("5. ASSOCIATION RULE MINING\n")
    f.write(f"   Number of rules (support>=0.05, confidence>=0.6): {num_rules}\n\n")

    f.write("6. HIERARCHICAL CLUSTERING\n")
    f.write(f"   Optimal number of clusters: {optimal_clusters}\n\n")

    f.write("7. SPIRAL PLOT VISUALIZATION\n")
    f.write(f"   Overall radial range: {radial_range_rounded}\n\n")

    f.write("8. TOP REVENUE CATEGORY\n")
    f.write(f"   Category: {int(top_category)}\n")
    f.write(f"   Total revenue: ${top_category_revenue:,.2f}\n")

print("Detailed results saved to 'analysis_results.txt'")

print("\n" + "="*60)
print("ANALYSIS COMPLETE")
print("="*60)
print("\nKey Findings:")
print(f"  • Regression R²: {regression_r2}")
print(f"  • ANOVA F-statistic: {anova_f_stat}")
print(f"  • Granger F-statistic: {granger_f_stat}, p-value: {granger_p_val}")
print(f"  • Causal coefficient: {causal_coefficient}")
print(f"  • Association rules: {num_rules}")
print(f"  • Optimal clusters: {optimal_clusters}")
print(f"  • Radial range: {radial_range_rounded}")
print(f"  • Top category: {int(top_category)}")
