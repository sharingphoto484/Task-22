# ==========================================
# Delivery Logistics Optimization Analysis Script
# ==========================================
# Requirements: pandas, numpy, matplotlib, scipy, scikit-learn, statsmodels
# Input files: updated.xlsx, encoded_cleaned_test.xlsx, cleaned_test.xlsx (in same directory)
# Output files: quantile_histogram.png, ks_ecdf_plot.png, elastic_net_predictions.png,
#               correlation_comparison.png, poisson_distribution.png
# ==========================================

import pandas as pd
import numpy as np
import matplotlib.pyplot as plt
from scipy import stats
from sklearn.preprocessing import StandardScaler
from sklearn.linear_model import ElasticNetCV, PoissonRegressor
import warnings
warnings.filterwarnings('ignore')

# Try importing statsmodels for quantile regression
try:
    import statsmodels.api as sm
    import statsmodels.formula.api as smf
    HAS_STATSMODELS = True
except ImportError:
    HAS_STATSMODELS = False
    print("Warning: statsmodels not available, using sklearn quantile regression")
    from sklearn.linear_model import QuantileRegressor

# ---------- Load Data Files ----------
print("Loading datasets...")
df_train = pd.read_excel('updated.xlsx')
df_test_encoded = pd.read_excel('encoded_cleaned_test.xlsx')
df_test_cleaned = pd.read_excel('cleaned_test.xlsx')

print(f"Training data shape: {df_train.shape}")
print(f"Encoded test data shape: {df_test_encoded.shape}")
print(f"Cleaned test data shape: {df_test_cleaned.shape}")

# ==========================================
# ANALYSIS 1: Quantile Regression (75th Percentile)
# ==========================================
print("\n" + "="*50)
print("ANALYSIS 1: Quantile Regression (75th Percentile)")
print("="*50)

# ---------- Prepare Training Data for Quantile Regression ----------
df_qr = df_train.dropna(subset=['Delivery_person_Ratings', 'Vehicle_condition',
                                 'Road_traffic_density', 'Weatherconditions',
                                 'Time_taken(min)'])

print(f"Records after removing missing values: {len(df_qr)}")

# ---------- Fit Quantile Regression Model ----------
X_train_qr = df_qr[['Delivery_person_Ratings', 'Vehicle_condition',
                     'Road_traffic_density', 'Weatherconditions']]
y_train_qr = df_qr['Time_taken(min)']

if HAS_STATSMODELS:
    # Add constant for statsmodels
    X_train_qr_sm = sm.add_constant(X_train_qr)
    qr_model = sm.QuantReg(y_train_qr, X_train_qr_sm)
    qr_result = qr_model.fit(q=0.75)

    # Predict on test set (first 100 records)
    df_test_100 = df_test_encoded.head(100).copy()
    # Rename 'Weather' to 'Weatherconditions' for consistency
    if 'Weather' in df_test_100.columns:
        df_test_100['Weatherconditions'] = df_test_100['Weather']
    X_test_qr = df_test_100[['Delivery_person_Ratings', 'Vehicle_condition',
                              'Road_traffic_density', 'Weatherconditions']].fillna(
                                  df_test_100[['Delivery_person_Ratings', 'Vehicle_condition',
                                              'Road_traffic_density', 'Weatherconditions']].median())
    X_test_qr_sm = sm.add_constant(X_test_qr)
    predictions_75th = qr_result.predict(X_test_qr_sm)
else:
    # Use sklearn QuantileRegressor
    qr_model = QuantileRegressor(quantile=0.75, alpha=0.0, solver='highs')
    qr_model.fit(X_train_qr, y_train_qr)

    # Predict on test set (first 100 records)
    df_test_100 = df_test_encoded.head(100).copy()
    # Rename 'Weather' to 'Weatherconditions' for consistency
    if 'Weather' in df_test_100.columns:
        df_test_100['Weatherconditions'] = df_test_100['Weather']
    X_test_qr = df_test_100[['Delivery_person_Ratings', 'Vehicle_condition',
                              'Road_traffic_density', 'Weatherconditions']].fillna(
                                  df_test_100[['Delivery_person_Ratings', 'Vehicle_condition',
                                              'Road_traffic_density', 'Weatherconditions']].median())
    predictions_75th = qr_model.predict(X_test_qr)

# ---------- Compute Mean of Predicted 75th Percentiles ----------
mean_predicted_75th = np.mean(predictions_75th)
print(f"\nMean Predicted 75th Percentile: {mean_predicted_75th:.2f}")

# ---------- Generate Histogram ----------
plt.figure(figsize=(10, 6))
plt.hist(predictions_75th, bins=20, edgecolor='black', alpha=0.7)
plt.xlabel('Predicted 75th Percentile Values')
plt.ylabel('Frequency Count')
plt.title('Distribution of Predicted 75th Percentile Delivery Times\n(First 100 Test Records)')
plt.grid(True, alpha=0.3)
plt.tight_layout()
plt.savefig('quantile_histogram.png', dpi=300)
plt.close()
print("Saved: quantile_histogram.png")

# ==========================================
# ANALYSIS 2: Kolmogorov-Smirnov Test for Delivery_person_Age
# ==========================================
print("\n" + "="*50)
print("ANALYSIS 2: Kolmogorov-Smirnov Test")
print("="*50)

# ---------- Extract Age Distributions ----------
age_train = df_train['Delivery_person_Age'].dropna()
age_test = df_test_encoded['Delivery_person_Age'].dropna()

print(f"Training age samples: {len(age_train)}")
print(f"Test age samples: {len(age_test)}")

# ---------- Compute KS Test Statistic ----------
ks_statistic, ks_pvalue = stats.ks_2samp(age_train, age_test)
print(f"\nKS Test Statistic: {ks_statistic:.4f}")
print(f"KS Test p-value: {ks_pvalue:.4f}")

# ---------- Generate ECDF Plot ----------
plt.figure(figsize=(12, 6))
sorted_train = np.sort(age_train)
sorted_test = np.sort(age_test)
ecdf_train = np.arange(1, len(sorted_train) + 1) / len(sorted_train)
ecdf_test = np.arange(1, len(sorted_test) + 1) / len(sorted_test)

plt.plot(sorted_train, ecdf_train, label='Training Dataset', color='blue', linewidth=2)
plt.plot(sorted_test, ecdf_test, label='Test Dataset', color='orange', linewidth=2)
plt.xlabel('Delivery_person_Age')
plt.ylabel('Cumulative Probability')
plt.title('Empirical Cumulative Distribution Functions - Delivery Person Age')
plt.legend()
plt.grid(True, alpha=0.3)
plt.tight_layout()
plt.savefig('ks_ecdf_plot.png', dpi=300)
plt.close()
print("Saved: ks_ecdf_plot.png")

# ==========================================
# ANALYSIS 3: Elastic Net Regression
# ==========================================
print("\n" + "="*50)
print("ANALYSIS 3: Elastic Net Regression")
print("="*50)

# ---------- Prepare Data for Elastic Net ----------
predictors = ['Delivery_person_Age', 'Delivery_person_Ratings',
              'Vehicle_condition', 'Road_traffic_density', 'Weatherconditions']
df_en = df_train[predictors + ['Time_taken(min)']].dropna()

X_train_en = df_en[predictors]
y_train_en = df_en['Time_taken(min)']

print(f"Records for Elastic Net: {len(df_en)}")

# ---------- Standardize Predictors ----------
scaler = StandardScaler()
X_train_en_scaled = scaler.fit_transform(X_train_en)

# ---------- Fit Elastic Net with CV ----------
elastic_net = ElasticNetCV(l1_ratio=0.5, cv=5, random_state=42, max_iter=10000)
elastic_net.fit(X_train_en_scaled, y_train_en)

print(f"Best alpha: {elastic_net.alpha_:.4f}")

# ---------- Predict on First 50 Test Records ----------
df_test_50 = df_test_encoded.head(50).copy()
# Rename 'Weather' to 'Weatherconditions' for consistency
if 'Weather' in df_test_50.columns:
    df_test_50['Weatherconditions'] = df_test_50['Weather']
X_test_en = df_test_50[predictors].fillna(df_test_50[predictors].median())
X_test_en_scaled = scaler.transform(X_test_en)
predictions_en = elastic_net.predict(X_test_en_scaled)

# ---------- Compute RMSV (Root Mean Squared Value) ----------
rmsv = np.sqrt(np.mean(predictions_en**2))
print(f"\nRoot Mean Squared Value of Predictions: {rmsv:.2f}")

# ---------- Generate Scatter Plot ----------
plt.figure(figsize=(12, 6))
plt.scatter(range(1, 51), predictions_en, color='darkblue', s=100, alpha=0.6, edgecolor='black')
plt.plot(range(1, 51), predictions_en, color='lightblue', alpha=0.5, linestyle='--')
plt.xlabel('Record Index')
plt.ylabel('Predicted Delivery Time (min)')
plt.title('Elastic Net Predictions for First 50 Test Records')
plt.grid(True, alpha=0.3)
plt.tight_layout()
plt.savefig('elastic_net_predictions.png', dpi=300)
plt.close()
print("Saved: elastic_net_predictions.png")

# ==========================================
# ANALYSIS 4: Cross-Dataset Pearson Correlation
# ==========================================
print("\n" + "="*50)
print("ANALYSIS 4: Cross-Dataset Pearson Correlation")
print("="*50)

# ---------- Compute Correlations ----------
# Training dataset
df_corr_train = df_train[['Delivery_person_Ratings', 'Vehicle_condition']].dropna()
corr_train = df_corr_train['Delivery_person_Ratings'].corr(df_corr_train['Vehicle_condition'])

# Test dataset (cleaned)
df_corr_test = df_test_cleaned[['Delivery_person_Ratings', 'Vehicle_condition']].dropna()
corr_test = df_corr_test['Delivery_person_Ratings'].corr(df_corr_test['Vehicle_condition'])

# Correlation difference
corr_difference = corr_train - corr_test

print(f"Training Correlation: {corr_train:.4f}")
print(f"Test Correlation: {corr_test:.4f}")
print(f"\nCorrelation Difference (Train - Test): {corr_difference:.4f}")

# ---------- Generate Side-by-Side Scatter Plots ----------
fig, axes = plt.subplots(1, 2, figsize=(14, 6))

# Training data
axes[0].scatter(df_corr_train['Delivery_person_Ratings'],
                df_corr_train['Vehicle_condition'],
                alpha=0.5, color='blue', edgecolor='black', s=30)
axes[0].set_xlabel('Delivery_person_Ratings')
axes[0].set_ylabel('Vehicle_condition')
axes[0].set_title(f'Training Dataset (updated.xlsx)\nCorrelation: {corr_train:.4f}')
axes[0].grid(True, alpha=0.3)

# Test data
axes[1].scatter(df_corr_test['Delivery_person_Ratings'],
                df_corr_test['Vehicle_condition'],
                alpha=0.5, color='orange', edgecolor='black', s=30)
axes[1].set_xlabel('Delivery_person_Ratings')
axes[1].set_ylabel('Vehicle_condition')
axes[1].set_title(f'Test Dataset (cleaned_test.xlsx)\nCorrelation: {corr_test:.4f}')
axes[1].grid(True, alpha=0.3)

plt.tight_layout()
plt.savefig('correlation_comparison.png', dpi=300)
plt.close()
print("Saved: correlation_comparison.png")

# ==========================================
# ANALYSIS 5: Poisson Regression
# ==========================================
print("\n" + "="*50)
print("ANALYSIS 5: Poisson Regression")
print("="*50)

# ---------- Create Binary Predictors for Training ----------
df_poisson = df_train.copy()
df_poisson['high_traffic'] = (df_poisson['Road_traffic_density'] >= 2).astype(int)
df_poisson['festival_indicator'] = df_poisson['Festival'].astype(int)
df_poisson['urban_indicator'] = (df_poisson['City'] == 1).astype(int)

# Drop missing values
df_poisson = df_poisson[['multiple_deliveries', 'high_traffic',
                         'festival_indicator', 'urban_indicator']].dropna()

print(f"Records for Poisson regression: {len(df_poisson)}")

# ---------- Fit Poisson Regression ----------
X_train_poisson = df_poisson[['high_traffic', 'festival_indicator', 'urban_indicator']]
y_train_poisson = df_poisson['multiple_deliveries']

poisson_model = PoissonRegressor(max_iter=1000)
poisson_model.fit(X_train_poisson, y_train_poisson)

# ---------- Create Binary Predictors for Test ----------
df_test_200 = df_test_encoded.head(200).copy()
df_test_200['high_traffic'] = (df_test_200['Road_traffic_density'] >= 2).astype(int)
df_test_200['festival_indicator'] = df_test_200['Festival'].fillna(0).astype(int)
df_test_200['urban_indicator'] = (df_test_200['City'] == 1).astype(int)

X_test_poisson = df_test_200[['high_traffic', 'festival_indicator', 'urban_indicator']]

# ---------- Predict and Count ----------
predictions_poisson = poisson_model.predict(X_test_poisson)
count_ge_1_5 = np.sum(predictions_poisson >= 1.5)

print(f"\nCount of predictions >= 1.5: {count_ge_1_5}")

# ---------- Generate Bar Chart ----------
# Categorize predictions
pred_categories = np.round(predictions_poisson).astype(int)
unique, counts = np.unique(pred_categories, return_counts=True)

plt.figure(figsize=(10, 6))
plt.bar(unique, counts, edgecolor='black', alpha=0.7, color='steelblue')
plt.xlabel('Predicted multiple_deliveries Categories')
plt.ylabel('Frequency Count')
plt.title('Distribution of Poisson Predictions for Test Set (First 200 Records)')
plt.xticks(range(int(pred_categories.min()), int(pred_categories.max()) + 1))
plt.grid(True, alpha=0.3, axis='y')
plt.tight_layout()
plt.savefig('poisson_distribution.png', dpi=300)
plt.close()
print("Saved: poisson_distribution.png")

# ==========================================
# ANALYSIS 6: Median Delivery_person_Ratings Ratio
# ==========================================
print("\n" + "="*50)
print("ANALYSIS 6: Median Delivery_person_Ratings Ratio")
print("="*50)

# ---------- Compute Medians ----------
median_ratings_train = df_train['Delivery_person_Ratings'].dropna().median()
median_ratings_test = df_test_encoded['Delivery_person_Ratings'].dropna().median()

# ---------- Calculate Ratio ----------
median_ratio = median_ratings_train / median_ratings_test

print(f"Training Median Ratings: {median_ratings_train:.4f}")
print(f"Test Median Ratings: {median_ratings_test:.4f}")
print(f"\nMedian Ratio (Train/Test): {median_ratio:.4f}")

# ==========================================
# ANALYSIS 7: 90th Percentile Road_traffic_density Difference
# ==========================================
print("\n" + "="*50)
print("ANALYSIS 7: 90th Percentile Road_traffic_density")
print("="*50)

# ---------- Compute 90th Percentiles ----------
# Note: Using encoded_cleaned_test for numeric comparison since cleaned_test has categorical strings
p90_traffic_train = np.percentile(df_train['Road_traffic_density'].dropna(), 90)
p90_traffic_test = np.percentile(df_test_encoded['Road_traffic_density'].dropna(), 90)

# ---------- Calculate Difference ----------
p90_difference = p90_traffic_train - p90_traffic_test

print(f"Training 90th Percentile: {p90_traffic_train:.4f}")
print(f"Test 90th Percentile (from encoded): {p90_traffic_test:.4f}")
print(f"\n90th Percentile Difference (Train - Test): {p90_difference:.2f}")

# ==========================================
# ANALYSIS 8: Mean Vehicle_condition Absolute Difference
# ==========================================
print("\n" + "="*50)
print("ANALYSIS 8: Mean Vehicle_condition Comparison")
print("="*50)

# ---------- Compute Means ----------
mean_vehicle_encoded = df_test_encoded['Vehicle_condition'].dropna().mean()
mean_vehicle_cleaned = df_test_cleaned['Vehicle_condition'].dropna().mean()

# ---------- Calculate Absolute Difference ----------
mean_abs_difference = abs(mean_vehicle_encoded - mean_vehicle_cleaned)

print(f"Encoded Test Mean Vehicle_condition: {mean_vehicle_encoded:.4f}")
print(f"Cleaned Test Mean Vehicle_condition: {mean_vehicle_cleaned:.4f}")
print(f"\nAbsolute Mean Difference: {mean_abs_difference:.4f}")

# ==========================================
# SUMMARY OF KEY OUTPUTS
# ==========================================
print("\n" + "="*70)
print("KEY OUTPUTS SUMMARY")
print("="*70)
print(f"1. Mean Predicted 75th Percentile:                    {mean_predicted_75th:.2f}")
print(f"2. KS Test Statistic:                                 {ks_statistic:.4f}")
print(f"3. RMSV of Elastic Net Predictions:                   {rmsv:.2f}")
print(f"4. Correlation Difference (Train - Test):             {corr_difference:.4f}")
print(f"5. Count of Poisson Predictions >= 1.5:               {count_ge_1_5}")
print(f"6. Median Ratings Ratio (Train/Test):                 {median_ratio:.4f}")
print(f"7. 90th Percentile Traffic Difference (Train - Test): {p90_difference:.2f}")
print(f"8. Absolute Mean Vehicle_condition Difference:        {mean_abs_difference:.4f}")
print("="*70)
print("\nAnalysis Complete!")
