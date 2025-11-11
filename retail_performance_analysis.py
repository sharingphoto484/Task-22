# ==========================================
# Retail Performance Dynamics Analysis Script
# ==========================================
# Requirements: pandas, numpy, matplotlib, scipy, scikit-learn, mlxtend, statsmodels
# Input files: products.csv, order_items.csv, customers.csv (in same directory)
# Output files: merged_dataset.csv, spiral_plot.png, analysis_summary.json
# ==========================================

import pandas as pd
import numpy as np
import matplotlib.pyplot as plt
from scipy import stats
from scipy.cluster.hierarchy import dendrogram, linkage, fcluster
from sklearn.preprocessing import StandardScaler, LabelEncoder
from sklearn.linear_model import LinearRegression
from sklearn.metrics import r2_score, silhouette_score
from statsmodels.tsa.stattools import grangercausalitytests
from mlxtend.frequent_patterns import apriori, association_rules
from mlxtend.preprocessing import TransactionEncoder
import json
import warnings
warnings.filterwarnings('ignore')

# ==========================================
# Results Dictionary
# ==========================================
results = {}

# ==========================================
# ---------- Load CSVs Robustly ----------
# ==========================================
print("Loading datasets...")

# Load products
products_df = pd.read_csv('products.csv')
products_df.columns = products_df.columns.str.strip()

# Load order items
order_items_df = pd.read_csv('order_items.csv')
order_items_df.columns = order_items_df.columns.str.strip()

# Load customers
customers_df = pd.read_csv('customers.csv')
customers_df.columns = customers_df.columns.str.strip()

print(f"Loaded {len(products_df)} products, {len(order_items_df)} order items, {len(customers_df)} customers")

# ==========================================
# ---------- Data Cleaning ----------
# ==========================================
print("\nCleaning datasets...")

# Clean products: keep product_id, category_id, list_price
products_clean = products_df[['product_id', 'category_id', 'list_price']].copy()
products_clean.columns = ['ProductID', 'Category', 'Price']

# Remove missing and non-numeric values
products_clean = products_clean.dropna()
products_clean = products_clean[pd.to_numeric(products_clean['ProductID'], errors='coerce').notna()]
products_clean = products_clean[pd.to_numeric(products_clean['Category'], errors='coerce').notna()]
products_clean = products_clean[pd.to_numeric(products_clean['Price'], errors='coerce').notna()]
products_clean['ProductID'] = products_clean['ProductID'].astype(int)
products_clean['Category'] = products_clean['Category'].astype(int)
products_clean['Price'] = products_clean['Price'].astype(float)

print(f"Products after cleaning: {len(products_clean)}")

# Clean order_items: calculate Revenue = quantity * list_price * (1 - discount)
order_items_clean = order_items_df[['order_id', 'product_id', 'quantity', 'list_price', 'discount']].copy()
order_items_clean.columns = ['OrderID', 'ProductID', 'Quantity', 'ListPrice', 'Discount']

# Remove missing values
order_items_clean = order_items_clean.dropna()

# Ensure numeric
order_items_clean = order_items_clean[pd.to_numeric(order_items_clean['OrderID'], errors='coerce').notna()]
order_items_clean = order_items_clean[pd.to_numeric(order_items_clean['ProductID'], errors='coerce').notna()]
order_items_clean = order_items_clean[pd.to_numeric(order_items_clean['Quantity'], errors='coerce').notna()]
order_items_clean = order_items_clean[pd.to_numeric(order_items_clean['ListPrice'], errors='coerce').notna()]
order_items_clean = order_items_clean[pd.to_numeric(order_items_clean['Discount'], errors='coerce').notna()]

order_items_clean['OrderID'] = order_items_clean['OrderID'].astype(int)
order_items_clean['ProductID'] = order_items_clean['ProductID'].astype(int)
order_items_clean['Quantity'] = order_items_clean['Quantity'].astype(int)
order_items_clean['ListPrice'] = order_items_clean['ListPrice'].astype(float)
order_items_clean['Discount'] = order_items_clean['Discount'].astype(float)

# Calculate Revenue
order_items_clean['Revenue'] = order_items_clean['Quantity'] * order_items_clean['ListPrice'] * (1 - order_items_clean['Discount'])

print(f"Order items after cleaning: {len(order_items_clean)}")

# Clean customers: use state as Region
customers_clean = customers_df[['customer_id', 'state']].copy()
customers_clean.columns = ['CustomerID', 'Region']
customers_clean = customers_clean.dropna()
customers_clean = customers_clean[pd.to_numeric(customers_clean['CustomerID'], errors='coerce').notna()]
customers_clean['CustomerID'] = customers_clean['CustomerID'].astype(int)

print(f"Customers after cleaning: {len(customers_clean)}")

# ==========================================
# ---------- Merge Datasets ----------
# ==========================================
print("\nMerging datasets...")

# Merge order_items with products on ProductID
merged_orders = order_items_clean.merge(products_clean, on='ProductID', how='inner')
print(f"After merging order_items with products: {len(merged_orders)}")

# Assign CustomerID to orders (create synthetic relationship)
# Each unique order_id gets assigned to a random customer
np.random.seed(42)
unique_orders = merged_orders['OrderID'].unique()
customer_ids = customers_clean['CustomerID'].values
order_to_customer = pd.DataFrame({
    'OrderID': unique_orders,
    'CustomerID': np.random.choice(customer_ids, size=len(unique_orders), replace=True)
})

# Merge with customer assignments
merged_orders = merged_orders.merge(order_to_customer, on='OrderID', how='left')

# Calculate TotalPurchaseValue per customer
customer_purchase = merged_orders.groupby('CustomerID')['Revenue'].sum().reset_index()
customer_purchase.columns = ['CustomerID', 'TotalPurchaseValue']

# Merge with customers to get Region
customers_enriched = customers_clean.merge(customer_purchase, on='CustomerID', how='left')
customers_enriched['TotalPurchaseValue'] = customers_enriched['TotalPurchaseValue'].fillna(0)

# Final merged dataset
merged_full = merged_orders.merge(customers_enriched, on='CustomerID', how='inner')
print(f"Final merged dataset: {len(merged_full)}")

# Save merged dataset
merged_full.to_csv('merged_dataset.csv', index=False)
print("Saved merged_dataset.csv")

# ==========================================
# ---------- 1. Multiple Linear Regression ----------
# ==========================================
print("\n" + "="*50)
print("1. Multiple Linear Regression: Revenue ~ Quantity + Price")
print("="*50)

# Prepare data for regression
regression_data = merged_full[['Revenue', 'Quantity', 'Price']].copy()
regression_data = regression_data.dropna()

X = regression_data[['Quantity', 'Price']].values
y = regression_data['Revenue'].values

# Standardize predictors
scaler = StandardScaler()
X_scaled = scaler.fit_transform(X)

# Fit regression
reg_model = LinearRegression()
reg_model.fit(X_scaled, y)
y_pred = reg_model.predict(X_scaled)

# Calculate R-squared
r_squared = r2_score(y, y_pred)
results['regression_r_squared'] = round(r_squared, 3)

print(f"R-squared: {results['regression_r_squared']}")
print(f"Coefficients - Quantity: {reg_model.coef_[0]:.4f}, Price: {reg_model.coef_[1]:.4f}")
print(f"Intercept: {reg_model.intercept_:.4f}")

# ==========================================
# ---------- 2. One-Way ANOVA ----------
# ==========================================
print("\n" + "="*50)
print("2. One-Way ANOVA: Price ~ Category")
print("="*50)

# Prepare data for ANOVA
anova_data = products_clean[['Price', 'Category']].copy()
anova_data = anova_data.dropna()

# Group prices by category
categories = anova_data['Category'].unique()
price_groups = [anova_data[anova_data['Category'] == cat]['Price'].values for cat in categories]

# Perform one-way ANOVA
f_statistic, p_value = stats.f_oneway(*price_groups)
results['anova_f_statistic'] = round(f_statistic, 3)
results['anova_p_value'] = round(p_value, 4)

print(f"F-statistic: {results['anova_f_statistic']}")
print(f"p-value: {results['anova_p_value']}")

# ==========================================
# ---------- 3. Granger Causality Test ----------
# ==========================================
print("\n" + "="*50)
print("3. Granger Causality Test: Price -> Revenue")
print("="*50)

# Prepare time series data (aggregate by OrderID)
granger_data = merged_full.groupby('OrderID').agg({
    'Revenue': 'sum',
    'Price': 'mean'
}).reset_index()

# Sort by OrderID to create temporal sequence
granger_data = granger_data.sort_values('OrderID').reset_index(drop=True)

# Keep only Revenue and Price columns
granger_series = granger_data[['Revenue', 'Price']].copy()

# Ensure no missing values
granger_series = granger_series.dropna()

# Perform Granger causality test (max lag = 1)
try:
    granger_result = grangercausalitytests(granger_series[['Revenue', 'Price']], maxlag=1, verbose=False)

    # Extract F-statistic and p-value for lag 1
    f_stat = granger_result[1][0]['ssr_ftest'][0]
    p_val = granger_result[1][0]['ssr_ftest'][1]

    results['granger_f_statistic'] = round(f_stat, 3)
    results['granger_p_value'] = round(p_val, 4)

    print(f"F-statistic: {results['granger_f_statistic']}")
    print(f"p-value: {results['granger_p_value']}")
except Exception as e:
    print(f"Error in Granger test: {e}")
    results['granger_f_statistic'] = None
    results['granger_p_value'] = None

# ==========================================
# ---------- 4. Linear Causal Model ----------
# ==========================================
print("\n" + "="*50)
print("4. Linear Causal Model: TotalPurchaseValue -> Revenue (controlling for Region)")
print("="*50)

# Prepare data for causal model
causal_data = merged_full[['Revenue', 'TotalPurchaseValue', 'Region']].copy()
causal_data = causal_data.dropna()

# Encode Region as dummy variables
region_dummies = pd.get_dummies(causal_data['Region'], prefix='Region', drop_first=True)
X_causal = pd.concat([causal_data[['TotalPurchaseValue']], region_dummies], axis=1)
y_causal = causal_data['Revenue'].values

# Fit linear model
causal_model = LinearRegression()
causal_model.fit(X_causal, y_causal)

# Extract causal coefficient (first coefficient = TotalPurchaseValue)
causal_coefficient = causal_model.coef_[0]
results['causal_coefficient'] = round(causal_coefficient, 4)

print(f"Causal Coefficient (TotalPurchaseValue): {results['causal_coefficient']}")

# ==========================================
# ---------- 5. Association Rule Mining ----------
# ==========================================
print("\n" + "="*50)
print("5. Association Rule Mining: Product Pairs")
print("="*50)

# Create transaction data (products per order)
transactions = merged_full.groupby('OrderID')['ProductID'].apply(list).values

# Use TransactionEncoder
te = TransactionEncoder()
te_ary = te.fit(transactions).transform(transactions)
transactions_df = pd.DataFrame(te_ary, columns=te.columns_)

# Apply Apriori algorithm
frequent_itemsets = apriori(transactions_df, min_support=0.05, use_colnames=True)

if len(frequent_itemsets) > 0:
    # Generate association rules
    rules = association_rules(frequent_itemsets, metric="confidence", min_threshold=0.6, num_itemsets=len(frequent_itemsets))

    # Filter for product pairs (antecedents and consequents with length 1)
    rules_pairs = rules[(rules['antecedents'].apply(len) == 1) & (rules['consequents'].apply(len) == 1)]

    results['association_rules_count'] = len(rules_pairs)
    print(f"Number of rules (support >= 0.05, confidence >= 0.6): {results['association_rules_count']}")

    if len(rules_pairs) > 0:
        print(f"\nTop 5 rules:")
        print(rules_pairs[['antecedents', 'consequents', 'support', 'confidence']].head())
else:
    results['association_rules_count'] = 0
    print("No frequent itemsets found with min_support=0.05")

# ==========================================
# ---------- 6. Hierarchical Clustering ----------
# ==========================================
print("\n" + "="*50)
print("6. Hierarchical Clustering: Customer Segmentation")
print("="*50)

# Prepare clustering data
clustering_data = merged_full.groupby('CustomerID').agg({
    'TotalPurchaseValue': 'first',
    'ProductID': 'nunique'
}).reset_index()
clustering_data.columns = ['CustomerID', 'TotalPurchaseValue', 'UniqueProducts']
clustering_data = clustering_data.dropna()

# Features for clustering
X_cluster = clustering_data[['TotalPurchaseValue', 'UniqueProducts']].values

# Standardize features
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

# Find optimal k
optimal_k = cluster_range[np.argmax(silhouette_scores)]
optimal_silhouette = max(silhouette_scores)

results['optimal_clusters'] = optimal_k
results['optimal_silhouette'] = round(optimal_silhouette, 4)

print(f"Optimal number of clusters: {results['optimal_clusters']}")
print(f"Optimal silhouette score: {results['optimal_silhouette']}")

# ==========================================
# ---------- 7. Spiral Plot Visualization ----------
# ==========================================
print("\n" + "="*50)
print("7. Spiral Plot: Products by ProductID, Price, and Category")
print("="*50)

# Prepare spiral plot data
spiral_data = products_clean.sort_values('ProductID').reset_index(drop=True)

# Calculate category frequency
category_freq = spiral_data['Category'].value_counts().to_dict()
spiral_data['CategoryFrequency'] = spiral_data['Category'].map(category_freq)

# Generate spiral coordinates
n_products = len(spiral_data)
theta = np.linspace(0, 8 * np.pi, n_products)  # 4 complete rotations
r = spiral_data['Price'].values  # Radial distance = Price

# Convert to Cartesian coordinates
x = r * np.cos(theta)
y = r * np.sin(theta)

# Color intensity based on category frequency
colors = spiral_data['CategoryFrequency'].values

# Create spiral plot
plt.figure(figsize=(12, 12))
scatter = plt.scatter(x, y, c=colors, cmap='viridis', s=20, alpha=0.7)
plt.colorbar(scatter, label='Category Frequency')
plt.title('Spiral Plot: Products by ProductID\n(Radial Distance = Price, Color = Category Frequency)', fontsize=14)
plt.xlabel('X Coordinate', fontsize=12)
plt.ylabel('Y Coordinate', fontsize=12)
plt.grid(True, alpha=0.3)
plt.axis('equal')
plt.tight_layout()
plt.savefig('spiral_plot.png', dpi=300, bbox_inches='tight')
print("Saved spiral_plot.png")

# Calculate radial range
radial_range = r.max() - r.min()
results['radial_range'] = round(radial_range, 2)
print(f"Radial range: {results['radial_range']}")

# ==========================================
# ---------- 8. Top Revenue Category ----------
# ==========================================
print("\n" + "="*50)
print("8. Top Revenue-Contributing Category")
print("="*50)

# Calculate total revenue by category
category_revenue = merged_full.groupby('Category')['Revenue'].sum().reset_index()
category_revenue = category_revenue.sort_values('Revenue', ascending=False)

top_category = int(category_revenue.iloc[0]['Category'])
top_revenue = category_revenue.iloc[0]['Revenue']
total_revenue = category_revenue['Revenue'].sum()
top_percentage = (top_revenue / total_revenue) * 100

results['top_revenue_category'] = top_category
results['top_revenue_amount'] = round(top_revenue, 2)
results['top_revenue_percentage'] = round(top_percentage, 2)

print(f"Top category: {results['top_revenue_category']}")
print(f"Total revenue: ${results['top_revenue_amount']:,.2f}")
print(f"Percentage of total: {results['top_revenue_percentage']:.2f}%")

# ==========================================
# ---------- Save Analysis Summary ----------
# ==========================================
print("\n" + "="*50)
print("Saving Analysis Summary")
print("="*50)

# Save results to JSON
with open('analysis_summary.json', 'w') as f:
    json.dump(results, f, indent=4)

print("Saved analysis_summary.json")

# ==========================================
# ---------- Final Summary Report ----------
# ==========================================
print("\n" + "="*60)
print("FINAL ANALYSIS SUMMARY")
print("="*60)
print(f"1. Multiple Linear Regression R-squared: {results['regression_r_squared']}")
print(f"2. ANOVA F-statistic: {results['anova_f_statistic']}")
print(f"3. Granger Causality F-statistic: {results['granger_f_statistic']}")
print(f"4. Granger Causality p-value: {results['granger_p_value']}")
print(f"5. Causal Coefficient: {results['causal_coefficient']}")
print(f"6. Association Rules Count: {results['association_rules_count']}")
print(f"7. Optimal Clusters: {results['optimal_clusters']}")
print(f"8. Radial Range: {results['radial_range']}")
print(f"9. Top Revenue Category: {results['top_revenue_category']}")
print("="*60)
print("\nAnalysis complete! All outputs saved.")
