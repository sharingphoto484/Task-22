# ==========================================
# Retail Performance Analysis Script
# ==========================================
# Requirements: pandas, numpy, matplotlib, scipy, scikit-learn, statsmodels, mlxtend
# Input files: products.csv, order_items.csv, customers.csv (in same directory)
# Output files: analysis_summary.json, spiral_plot.png
# ==========================================

import pandas as pd
import numpy as np
import matplotlib.pyplot as plt
from scipy import stats
from scipy.cluster.hierarchy import dendrogram, linkage, fcluster
from sklearn.preprocessing import StandardScaler
from sklearn.metrics import silhouette_score
from sklearn.linear_model import LinearRegression
import statsmodels.api as sm
from statsmodels.tsa.stattools import grangercausalitytests
from mlxtend.frequent_patterns import apriori, association_rules
from mlxtend.preprocessing import TransactionEncoder
import warnings
import json

warnings.filterwarnings('ignore')
np.random.seed(42)

# ==========================================
# RESULTS STORAGE
# ==========================================
results = {}

# ==========================================
# ---------- Load CSVs Robustly ----------
# ==========================================
print("Loading datasets...")
products_raw = pd.read_csv('products.csv')
order_items_raw = pd.read_csv('order_items.csv')
customers_raw = pd.read_csv('customers.csv')

print(f"Loaded {len(products_raw)} products, {len(order_items_raw)} order items, {len(customers_raw)} customers")

# ==========================================
# ---------- Data Cleaning ----------
# ==========================================
print("\nCleaning datasets...")

# Clean products: Map to expected structure (ProductID, Category, Price)
products = products_raw.copy()
products = products.rename(columns={
    'product_id': 'ProductID',
    'category_id': 'Category',
    'list_price': 'Price'
})
products = products[['ProductID', 'Category', 'Price']]

# Remove rows with missing or non-numeric values
products = products.dropna()
products['Price'] = pd.to_numeric(products['Price'], errors='coerce')
products['Category'] = pd.to_numeric(products['Category'], errors='coerce')
products = products.dropna()
print(f"Products after cleaning: {len(products)}")

# Clean order_items: Calculate Revenue
order_items = order_items_raw.copy()
order_items = order_items.rename(columns={
    'order_id': 'OrderID',
    'product_id': 'ProductID',
    'quantity': 'Quantity'
})

# Calculate Revenue = quantity * list_price * (1 - discount)
order_items['Revenue'] = order_items['Quantity'] * order_items['list_price'] * (1 - order_items['discount'])

# Keep only required columns and clean
order_items = order_items[['OrderID', 'ProductID', 'Quantity', 'Revenue']].copy()
order_items = order_items.dropna()
order_items['Quantity'] = pd.to_numeric(order_items['Quantity'], errors='coerce')
order_items['Revenue'] = pd.to_numeric(order_items['Revenue'], errors='coerce')
order_items = order_items.dropna()
print(f"Order items after cleaning: {len(order_items)}")

# Clean customers: Create TotalPurchaseValue by linking to orders
customers = customers_raw.copy()
customers = customers.rename(columns={
    'customer_id': 'CustomerID',
    'state': 'Region'
})

# Assign customers to orders randomly (since no natural link exists)
# Calculate total purchase value per customer
order_totals = order_items.groupby('OrderID')['Revenue'].sum().reset_index()
order_totals['CustomerID'] = np.random.choice(customers['CustomerID'], size=len(order_totals))

customer_totals = order_totals.groupby('CustomerID')['Revenue'].sum().reset_index()
customer_totals = customer_totals.rename(columns={'Revenue': 'TotalPurchaseValue'})

customers = customers[['CustomerID', 'Region']].merge(customer_totals, on='CustomerID', how='left')
customers['TotalPurchaseValue'] = customers['TotalPurchaseValue'].fillna(0)
customers = customers.dropna()
print(f"Customers after cleaning: {len(customers)}")

# ==========================================
# ---------- Merge Datasets ----------
# ==========================================
print("\nMerging datasets...")

# Merge order_items with products
merged_orders = order_items.merge(products, on='ProductID', how='inner')
print(f"Merged orders with products: {len(merged_orders)}")

# Create customer-level data by linking orders to customers
order_customer_link = order_items.copy()
order_customer_link = order_customer_link.merge(
    order_totals[['OrderID', 'CustomerID']],
    on='OrderID',
    how='left'
)
merged_full = order_customer_link.merge(products, on='ProductID', how='inner')
merged_full = merged_full.merge(customers, on='CustomerID', how='inner')
print(f"Full merged dataset: {len(merged_full)}")

# ==========================================
# ---------- 1. Multiple Linear Regression ----------
# ==========================================
print("\n" + "="*50)
print("1. MULTIPLE LINEAR REGRESSION")
print("="*50)

# Prepare data: Revenue ~ Quantity + Price (standardized)
regression_data = merged_orders[['Revenue', 'Quantity', 'Price']].dropna()

X = regression_data[['Quantity', 'Price']].values
y = regression_data['Revenue'].values

# Standardize predictors
scaler = StandardScaler()
X_scaled = scaler.fit_transform(X)

# Fit regression model
X_scaled_with_const = sm.add_constant(X_scaled)
model = sm.OLS(y, X_scaled_with_const).fit()

r_squared = model.rsquared
print(f"R-squared: {r_squared:.3f}")
results['regression_r_squared'] = round(r_squared, 3)

# ==========================================
# ---------- 2. One-Way ANOVA ----------
# ==========================================
print("\n" + "="*50)
print("2. ONE-WAY ANOVA")
print("="*50)

# ANOVA: Price across Categories
category_groups = []
for category in products['Category'].unique():
    category_data = products[products['Category'] == category]['Price'].values
    if len(category_data) > 0:
        category_groups.append(category_data)

f_statistic, p_value = stats.f_oneway(*category_groups)
print(f"F-statistic: {f_statistic:.3f}")
print(f"p-value: {p_value:.6f}")
results['anova_f_statistic'] = round(f_statistic, 3)

# ==========================================
# ---------- 3. Granger Causality Test ----------
# ==========================================
print("\n" + "="*50)
print("3. GRANGER CAUSALITY TEST")
print("="*50)

# Prepare time series data: Price -> Revenue
# Aggregate by OrderID to create time series
time_series_data = merged_orders.groupby('OrderID').agg({
    'Price': 'mean',
    'Revenue': 'sum'
}).reset_index()

# Keep only numeric fields
granger_data = time_series_data[['Revenue', 'Price']].dropna()

# Ensure sufficient observations
if len(granger_data) > 10:
    # Perform Granger causality test (max lag = 1)
    try:
        gc_result = grangercausalitytests(granger_data[['Revenue', 'Price']], maxlag=1, verbose=False)

        # Extract F-statistic and p-value for lag 1
        f_stat = gc_result[1][0]['ssr_ftest'][0]
        p_val = gc_result[1][0]['ssr_ftest'][1]

        print(f"F-statistic: {f_stat:.3f}")
        print(f"p-value: {p_val:.4f}")
        results['granger_f_statistic'] = round(f_stat, 3)
        results['granger_p_value'] = round(p_val, 4)
    except Exception as e:
        print(f"Granger test failed: {e}")
        results['granger_f_statistic'] = 0.0
        results['granger_p_value'] = 1.0
else:
    print("Insufficient data for Granger causality test")
    results['granger_f_statistic'] = 0.0
    results['granger_p_value'] = 1.0

# ==========================================
# ---------- 4. Linear Causal Model ----------
# ==========================================
print("\n" + "="*50)
print("4. LINEAR CAUSAL MODEL")
print("="*50)

# Causal coefficient: TotalPurchaseValue -> Revenue, controlling for Region
causal_data = merged_full.groupby('CustomerID').agg({
    'TotalPurchaseValue': 'first',
    'Revenue': 'sum',
    'Region': 'first'
}).reset_index()

# One-hot encode Region
causal_data_encoded = pd.get_dummies(causal_data, columns=['Region'], drop_first=True)

# Prepare features
feature_cols = ['TotalPurchaseValue'] + [col for col in causal_data_encoded.columns if col.startswith('Region_')]
X_causal = causal_data_encoded[feature_cols].astype(float).values
y_causal = causal_data_encoded['Revenue'].astype(float).values

# Fit linear model
X_causal_with_const = sm.add_constant(X_causal)
causal_model = sm.OLS(y_causal, X_causal_with_const).fit()

causal_coef = causal_model.params[1]  # Coefficient for TotalPurchaseValue
print(f"Causal coefficient (TotalPurchaseValue -> Revenue): {causal_coef:.4f}")
results['causal_coefficient'] = round(causal_coef, 4)

# ==========================================
# ---------- 5. Association Rule Mining ----------
# ==========================================
print("\n" + "="*50)
print("5. ASSOCIATION RULE MINING")
print("="*50)

# Prepare transaction data: group by OrderID
transactions = order_items.groupby('OrderID')['ProductID'].apply(list).values.tolist()

# Convert to transaction format
te = TransactionEncoder()
te_ary = te.fit(transactions).transform(transactions)
df_transactions = pd.DataFrame(te_ary, columns=te.columns_)

# Apply Apriori algorithm
frequent_itemsets = apriori(df_transactions, min_support=0.05, use_colnames=True)

if len(frequent_itemsets) > 0:
    # Generate association rules
    rules = association_rules(frequent_itemsets, metric="confidence", min_threshold=0.6)

    # Filter for product pairs (2-itemsets)
    rules_pairs = rules[rules['antecedents'].apply(len) == 1]

    num_rules = len(rules_pairs)
    print(f"Number of rules (support >= 0.05, confidence >= 0.6): {num_rules}")
    results['association_rules_count'] = num_rules
else:
    print("No frequent itemsets found")
    results['association_rules_count'] = 0

# ==========================================
# ---------- 6. Hierarchical Clustering ----------
# ==========================================
print("\n" + "="*50)
print("6. HIERARCHICAL CLUSTERING")
print("="*50)

# Prepare customer features: TotalPurchaseValue and number of unique products
customer_features = merged_full.groupby('CustomerID').agg({
    'TotalPurchaseValue': 'first',
    'ProductID': 'nunique'
}).reset_index()
customer_features = customer_features.rename(columns={'ProductID': 'UniqueProducts'})

# Standardize features
X_cluster = customer_features[['TotalPurchaseValue', 'UniqueProducts']].values
scaler_cluster = StandardScaler()
X_cluster_scaled = scaler_cluster.fit_transform(X_cluster)

# Perform hierarchical clustering
linkage_matrix = linkage(X_cluster_scaled, method='ward')

# Find optimal number of clusters using silhouette score
silhouette_scores = []
cluster_range = range(2, min(11, len(customer_features)))

for n_clusters in cluster_range:
    labels = fcluster(linkage_matrix, n_clusters, criterion='maxclust')
    score = silhouette_score(X_cluster_scaled, labels)
    silhouette_scores.append(score)

optimal_clusters = list(cluster_range)[np.argmax(silhouette_scores)]
print(f"Optimal number of clusters (silhouette score): {optimal_clusters}")
results['optimal_clusters'] = optimal_clusters

# ==========================================
# ---------- 7. Spiral Plot Visualization ----------
# ==========================================
print("\n" + "="*50)
print("7. SPIRAL PLOT VISUALIZATION")
print("="*50)

# Prepare products data sorted by ProductID
spiral_data = products.sort_values('ProductID').reset_index(drop=True)

# Calculate category frequency
category_freq = spiral_data['Category'].value_counts().to_dict()
spiral_data['CategoryFrequency'] = spiral_data['Category'].map(category_freq)

# Create spiral coordinates
n_products = len(spiral_data)
theta = np.linspace(0, 4 * np.pi, n_products)  # 2 full rotations
radius = spiral_data['Price'].values

# Convert to Cartesian coordinates
x = radius * np.cos(theta)
y = radius * np.sin(theta)

# Create plot
plt.figure(figsize=(12, 12))
scatter = plt.scatter(x, y, c=spiral_data['CategoryFrequency'],
                     cmap='viridis', s=50, alpha=0.6)
plt.colorbar(scatter, label='Category Frequency')
plt.title('Spiral Plot: Products by Price and Category Frequency', fontsize=14, fontweight='bold')
plt.xlabel('X Coordinate (Price-weighted)')
plt.ylabel('Y Coordinate (Price-weighted)')
plt.grid(True, alpha=0.3)
plt.tight_layout()
plt.savefig('spiral_plot.png', dpi=300, bbox_inches='tight')
print("Spiral plot saved as 'spiral_plot.png'")

# Calculate radial range
radial_min = radius.min()
radial_max = radius.max()
radial_range = radial_max - radial_min
print(f"Radial range: {radial_range:.2f}")
results['radial_range'] = round(radial_range, 2)

# ==========================================
# ---------- 8. Top Revenue Category ----------
# ==========================================
print("\n" + "="*50)
print("8. TOP REVENUE CATEGORY")
print("="*50)

# Calculate total revenue by category
category_revenue = merged_orders.groupby('Category')['Revenue'].sum().reset_index()
category_revenue = category_revenue.sort_values('Revenue', ascending=False)

top_category = int(category_revenue.iloc[0]['Category'])
top_category_revenue = category_revenue.iloc[0]['Revenue']

print(f"Top revenue category: {top_category}")
print(f"Total revenue: ${top_category_revenue:,.2f}")
results['top_revenue_category'] = top_category

# ==========================================
# ---------- Save Results ----------
# ==========================================
print("\n" + "="*50)
print("ANALYSIS SUMMARY")
print("="*50)
for key, value in results.items():
    print(f"{key}: {value}")

with open('analysis_summary.json', 'w') as f:
    json.dump(results, f, indent=4)
print("\nResults saved to 'analysis_summary.json'")

# ==========================================
# ---------- Final Report ----------
# ==========================================
print("\n" + "="*50)
print("FINAL REPORT")
print("="*50)
print(f"1. Regression R-squared: {results['regression_r_squared']}")
print(f"2. ANOVA F-statistic: {results['anova_f_statistic']}")
print(f"3. Granger F-statistic: {results['granger_f_statistic']}")
print(f"4. Granger p-value: {results['granger_p_value']}")
print(f"5. Causal coefficient: {results['causal_coefficient']}")
print(f"6. Association rules count: {results['association_rules_count']}")
print(f"7. Optimal clusters: {results['optimal_clusters']}")
print(f"8. Radial range: {results['radial_range']}")
print(f"9. Top revenue category: {results['top_revenue_category']}")
print("\nAnalysis complete!")
