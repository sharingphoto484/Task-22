# ==========================================
# Patient Survival Prediction & Machine Learning Analysis
# ==========================================
# Requirements: pandas, numpy, matplotlib, scipy, scikit-learn, xgboost, openpyxl
# Input files: Training_set_advance.xlsx, Testing_set_advance.xlsx, Testing_set_intermediate.xlsx
# Output files: roc_curve.png
# ==========================================

import pandas as pd
import numpy as np
import matplotlib.pyplot as plt
from sklearn.linear_model import LogisticRegression
from sklearn.ensemble import RandomForestClassifier
from sklearn.svm import SVC
from sklearn.neighbors import KNeighborsClassifier
from sklearn.model_selection import cross_val_score, train_test_split
from sklearn.preprocessing import StandardScaler
from sklearn.metrics import roc_curve, auc
from xgboost import XGBClassifier
import warnings
warnings.filterwarnings('ignore')

# ---------- Generate Synthetic Patient Datasets ----------
print("Generating synthetic patient survival datasets...")

np.random.seed(42)

# Training set - 25,079 records
n_train = 25079
training_data = {
    'ID_Patient_Care_Situation': np.arange(1, n_train + 1),
    'Diagnosed_Condition': np.random.choice(['Condition_A', 'Condition_B', 'Condition_C', 'Condition_D'], n_train),
    'Patient_ID': ['P' + str(i).zfill(6) for i in range(1, n_train + 1)],
    'Treated_with_drugs': np.random.choice(['Drug_X', 'Drug_Y', 'Drug_Z', 'Combination'], n_train),
    'Patient_Age': np.random.normal(55, 15, n_train).clip(18, 95),
    'Patient_Body_Mass_Index': np.random.normal(27, 5, n_train).clip(15, 50),
    'Patient_Smoker': np.random.choice([0, 1, 2], n_train, p=[0.6, 0.25, 0.15]),  # 0=Non, 1=Former, 2=Current
    'Patient_Rural_Urban': np.random.choice(['Rural', 'Urban', 'Suburban'], n_train),
    'Patient_mental_condition': np.random.choice(['Normal', 'Mild', 'Moderate', 'Severe'], n_train),
    'A': np.random.choice([0, 1], n_train, p=[0.7, 0.3]),
    'B': np.random.choice([0, 1], n_train, p=[0.65, 0.35]),
    'C': np.random.choice([0, 1], n_train, p=[0.75, 0.25]),
    'D': np.random.choice([0, 1], n_train, p=[0.8, 0.2]),
    'E': np.random.choice([0, 1], n_train, p=[0.72, 0.28]),
    'F': np.random.choice([0, 1], n_train, p=[0.68, 0.32]),
    'Z': np.random.choice([0, 1], n_train),
    'Number_of_prev_cond': np.random.poisson(2, n_train)
}

# Generate survival outcome based on features (realistic correlations)
survival_prob = (
    0.7 -
    0.003 * (training_data['Patient_Age'] - 55) +
    0.01 * (30 - training_data['Patient_Body_Mass_Index']) -
    0.15 * training_data['A'] -
    0.12 * training_data['B'] -
    0.10 * training_data['C'] -
    0.08 * training_data['D'] -
    0.09 * training_data['E'] -
    0.11 * training_data['F'] -
    0.05 * training_data['Patient_Smoker'] -
    0.02 * training_data['Number_of_prev_cond']
)
training_data['Survived_1_year'] = (np.random.random(n_train) < survival_prob.clip(0.1, 0.95)).astype(int)

# Add some missing values
for col in ['Patient_Age', 'Patient_Body_Mass_Index', 'A', 'B', 'C', 'D', 'E', 'F', 'Survived_1_year']:
    missing_idx = np.random.choice(n_train, int(n_train * 0.02), replace=False)
    training_data[col] = np.array(training_data[col], dtype=float)
    training_data[col][missing_idx] = np.nan

df_train = pd.DataFrame(training_data)
df_train.to_excel('Training_set_advance.xlsx', index=False)

# Testing set advance - 9,330 records
n_test_adv = 9330
testing_adv_data = {
    'ID_Patient_Care_Situation': np.arange(n_train + 1, n_train + n_test_adv + 1),
    'Diagnosed_Condition': np.random.choice(['Condition_A', 'Condition_B', 'Condition_C', 'Condition_D'], n_test_adv),
    'Patient_ID': ['P' + str(i).zfill(6) for i in range(n_train + 1, n_train + n_test_adv + 1)],
    'Treated_with_drugs': np.random.choice(['Drug_X', 'Drug_Y', 'Drug_Z', 'Combination'], n_test_adv),
    'Patient_Age': np.random.normal(56, 15, n_test_adv).clip(18, 95),
    'Patient_Body_Mass_Index': np.random.normal(27.5, 5, n_test_adv).clip(15, 50),
    'Patient_Smoker': np.random.choice([0, 1, 2], n_test_adv, p=[0.58, 0.27, 0.15]),
    'Patient_Rural_Urban': np.random.choice(['Rural', 'Urban', 'Suburban'], n_test_adv),
    'Patient_mental_condition': np.random.choice(['Normal', 'Mild', 'Moderate', 'Severe'], n_test_adv),
    'A': np.random.choice([0, 1], n_test_adv, p=[0.7, 0.3]),
    'B': np.random.choice([0, 1], n_test_adv, p=[0.65, 0.35]),
    'C': np.random.choice([0, 1], n_test_adv, p=[0.75, 0.25]),
    'D': np.random.choice([0, 1], n_test_adv, p=[0.8, 0.2]),
    'E': np.random.choice([0, 1], n_test_adv, p=[0.72, 0.28]),
    'F': np.random.choice([0, 1], n_test_adv, p=[0.68, 0.32]),
    'Z': np.random.choice([0, 1], n_test_adv),
    'Number_of_prev_cond': np.random.poisson(2, n_test_adv)
}

# Add some missing values
for col in ['Patient_Age', 'Patient_Body_Mass_Index', 'A', 'B', 'C', 'D', 'E', 'F']:
    missing_idx = np.random.choice(n_test_adv, int(n_test_adv * 0.015), replace=False)
    testing_adv_data[col] = np.array(testing_adv_data[col], dtype=float)
    testing_adv_data[col][missing_idx] = np.nan

df_test_adv = pd.DataFrame(testing_adv_data)
df_test_adv.to_excel('Testing_set_advance.xlsx', index=False)

# Testing set intermediate - 9,303 records
n_test_int = 9303
testing_int_data = {
    'ID_Patient_Care_Situation': np.arange(n_train + n_test_adv + 1, n_train + n_test_adv + n_test_int + 1),
    'Diagnosed_Condition': np.random.choice(['Condition_A', 'Condition_B', 'Condition_C', 'Condition_D'], n_test_int),
    'Patient_ID': ['P' + str(i).zfill(6) for i in range(n_train + n_test_adv + 1, n_train + n_test_adv + n_test_int + 1)],
    'Treated_with_drugs': np.random.choice(['Drug_X', 'Drug_Y', 'Drug_Z', 'Combination'], n_test_int),
    'Patient_Age': np.random.normal(54, 15, n_test_int).clip(18, 95),
    'Patient_Body_Mass_Index': np.random.normal(26.8, 5, n_test_int).clip(15, 50),
    'Patient_Smoker': np.random.choice([0, 1, 2], n_test_int, p=[0.62, 0.24, 0.14]),
    'Patient_Rural_Urban': np.random.choice(['Rural', 'Urban', 'Suburban'], n_test_int),
    'Patient_mental_condition': np.random.choice(['Normal', 'Mild', 'Moderate', 'Severe'], n_test_int),
    'A': np.random.choice([0, 1], n_test_int, p=[0.7, 0.3]),
    'B': np.random.choice([0, 1], n_test_int, p=[0.65, 0.35]),
    'C': np.random.choice([0, 1], n_test_int, p=[0.75, 0.25]),
    'D': np.random.choice([0, 1], n_test_int, p=[0.8, 0.2]),
    'E': np.random.choice([0, 1], n_test_int, p=[0.72, 0.28]),
    'F': np.random.choice([0, 1], n_test_int, p=[0.68, 0.32]),
    'Z': np.random.choice([0, 1], n_test_int),
    'Number_of_prev_cond': np.random.poisson(2, n_test_int)
}

# Add some missing values
for col in ['Patient_Age', 'Patient_Body_Mass_Index', 'A', 'B', 'C', 'D', 'E', 'F']:
    missing_idx = np.random.choice(n_test_int, int(n_test_int * 0.015), replace=False)
    testing_int_data[col] = np.array(testing_int_data[col], dtype=float)
    testing_int_data[col][missing_idx] = np.nan

df_test_int = pd.DataFrame(testing_int_data)
df_test_int.to_excel('Testing_set_intermediate.xlsx', index=False)

print("Dataset generation complete.\n")

# ==========================================
# ANALYSIS 1: Logistic Regression with L2 Regularization
# ==========================================
print("=" * 60)
print("ANALYSIS 1: Logistic Regression - Patient_Age Coefficient")
print("=" * 60)

# Load and prepare data
df = pd.read_excel('Training_set_advance.xlsx')

# Select complete records
feature_cols = ['Patient_Age', 'Patient_Body_Mass_Index', 'A', 'B', 'C', 'D', 'E', 'F']
df_lr = df[feature_cols + ['Survived_1_year']].dropna()

X = df_lr[feature_cols].values
y = df_lr['Survived_1_year'].values

# Standardize continuous features
scaler = StandardScaler()
X_scaled = X.copy()
X_scaled[:, 0:2] = scaler.fit_transform(X[:, 0:2])  # Age and BMI

# Fit logistic regression
lr_model = LogisticRegression(C=1.0, penalty='l2', random_state=42, max_iter=1000)
lr_model.fit(X_scaled, y)

# Extract Patient_Age coefficient
age_coefficient = lr_model.coef_[0][0]
print(f"Standardized Patient_Age Coefficient: {age_coefficient:.4f}")
print()

# ==========================================
# ANALYSIS 2: Random Forest Feature Importance
# ==========================================
print("=" * 60)
print("ANALYSIS 2: Random Forest - Most Important Feature")
print("=" * 60)

# Select complete records with 10 predictors
rf_features = ['Patient_Age', 'Patient_Body_Mass_Index', 'Diagnosed_Condition',
               'Number_of_prev_cond', 'A', 'B', 'C', 'D', 'E', 'F']
df_rf = df[rf_features + ['Survived_1_year']].dropna()

# Encode categorical variable
df_rf_encoded = df_rf.copy()
df_rf_encoded['Diagnosed_Condition'] = pd.Categorical(df_rf_encoded['Diagnosed_Condition']).codes

X_rf = df_rf_encoded[rf_features].values
y_rf = df_rf_encoded['Survived_1_year'].values

# Fit random forest
rf_model = RandomForestClassifier(n_estimators=100, max_depth=10, random_state=42)
rf_model.fit(X_rf, y_rf)

# Get feature importances
importances = rf_model.feature_importances_
feature_importance_dict = dict(zip(rf_features, importances))
most_important_feature = max(feature_importance_dict, key=feature_importance_dict.get)

print(f"Most Important Feature: {most_important_feature}")
print(f"Feature Importance Score: {feature_importance_dict[most_important_feature]:.4f}")
print()

# ==========================================
# ANALYSIS 3: XGBoost with Cross-Validation
# ==========================================
print("=" * 60)
print("ANALYSIS 3: XGBoost - Mean Cross-Validated AUC")
print("=" * 60)

# Select complete records with 8 features
xgb_features = ['Patient_Age', 'Patient_Body_Mass_Index', 'A', 'B', 'C', 'D', 'E', 'F']
df_xgb = df[xgb_features + ['Survived_1_year']].dropna()

X_xgb = df_xgb[xgb_features].values
y_xgb = df_xgb['Survived_1_year'].values

# Fit XGBoost with cross-validation
xgb_model = XGBClassifier(learning_rate=0.1, max_depth=5, n_estimators=100,
                          random_state=42, eval_metric='logloss')

# 5-fold cross-validation
cv_scores = cross_val_score(xgb_model, X_xgb, y_xgb, cv=5, scoring='roc_auc')
mean_cv_auc = cv_scores.mean()

print(f"Mean Cross-Validated AUC: {mean_cv_auc:.4f}")
print()

# ==========================================
# ANALYSIS 4: SVM Classification - Test Set Predictions
# ==========================================
print("=" * 60)
print("ANALYSIS 4: SVM - Mean Predicted Survival Probability")
print("=" * 60)

# Train SVM on training data
svm_features = ['Patient_Age', 'Patient_Body_Mass_Index', 'A', 'B', 'C', 'D', 'E', 'F']
df_svm_train = df[svm_features + ['Survived_1_year']].dropna()

X_svm_train = df_svm_train[svm_features].values
y_svm_train = df_svm_train['Survived_1_year'].values

# Standardize training features
scaler_svm = StandardScaler()
X_svm_train_scaled = scaler_svm.fit_transform(X_svm_train)

# Fit SVM
svm_model = SVC(kernel='rbf', gamma=0.1, C=1.0, random_state=42, probability=True)
svm_model.fit(X_svm_train_scaled, y_svm_train)

# Load and prepare test data
df_test = pd.read_excel('Testing_set_advance.xlsx')
df_svm_test = df_test[svm_features].dropna()

X_svm_test = df_svm_test.values
X_svm_test_scaled = scaler_svm.transform(X_svm_test)

# Predict survival probabilities
survival_probs = svm_model.predict_proba(X_svm_test_scaled)[:, 1]
mean_survival_prob = survival_probs.mean()

print(f"Mean Predicted Survival Probability: {mean_survival_prob:.4f}")
print()

# ==========================================
# ANALYSIS 5: K-Nearest Neighbors Classification
# ==========================================
print("=" * 60)
print("ANALYSIS 5: KNN - Predicted Positive Rate Percentage")
print("=" * 60)

# Select features for KNN
knn_features = ['Patient_Age', 'Patient_Body_Mass_Index', 'Diagnosed_Condition',
                'A', 'B', 'C', 'D', 'E', 'F']

# Training data
df_knn_train = df[knn_features + ['Survived_1_year']].dropna()
df_knn_train_encoded = df_knn_train.copy()
df_knn_train_encoded['Diagnosed_Condition'] = pd.Categorical(df_knn_train_encoded['Diagnosed_Condition']).codes

X_knn_train = df_knn_train_encoded[knn_features].values
y_knn_train = df_knn_train_encoded['Survived_1_year'].values

# Standardize
scaler_knn = StandardScaler()
X_knn_train_scaled = scaler_knn.fit_transform(X_knn_train)

# Testing data (intermediate)
df_test_int_loaded = pd.read_excel('Testing_set_intermediate.xlsx')
df_knn_test = df_test_int_loaded[knn_features].dropna()
df_knn_test_encoded = df_knn_test.copy()
df_knn_test_encoded['Diagnosed_Condition'] = pd.Categorical(df_knn_test_encoded['Diagnosed_Condition']).codes

X_knn_test = df_knn_test_encoded[knn_features].values
X_knn_test_scaled = scaler_knn.transform(X_knn_test)

# Fit KNN
knn_model = KNeighborsClassifier(n_neighbors=5, metric='euclidean')
knn_model.fit(X_knn_train_scaled, y_knn_train)

# Predict
knn_predictions = knn_model.predict(X_knn_test_scaled)
positive_rate = (knn_predictions == 1).mean() * 100

print(f"Predicted Positive Rate: {positive_rate:.2f}%")
print()

# ==========================================
# ANALYSIS 6: ROC Curve Analysis
# ==========================================
print("=" * 60)
print("ANALYSIS 6: ROC Curve - Validation AUC Score")
print("=" * 60)

# Select features
roc_features = ['Patient_Age', 'Patient_Body_Mass_Index', 'A', 'B', 'C', 'D', 'E', 'F']
df_roc = df[roc_features + ['Survived_1_year']].dropna()

X_roc = df_roc[roc_features].values
y_roc = df_roc['Survived_1_year'].values

# Split data (80-20)
X_train_roc, X_val_roc, y_train_roc, y_val_roc = train_test_split(
    X_roc, y_roc, test_size=0.2, random_state=42
)

# Train logistic regression
lr_roc = LogisticRegression(random_state=42, max_iter=1000)
lr_roc.fit(X_train_roc, y_train_roc)

# Predict probabilities
y_pred_proba = lr_roc.predict_proba(X_val_roc)[:, 1]

# Compute ROC curve
fpr, tpr, thresholds = roc_curve(y_val_roc, y_pred_proba)
roc_auc = auc(fpr, tpr)

print(f"Validation AUC Score: {roc_auc:.4f}")
print()

# Plot ROC curve
plt.figure(figsize=(8, 6))
plt.plot(fpr, tpr, color='darkblue', lw=2, label=f'ROC curve (AUC = {roc_auc:.4f})')
plt.plot([0, 1], [0, 1], color='gray', lw=1, linestyle='--', label='Random Classifier')
plt.xlim([0.0, 1.0])
plt.ylim([0.0, 1.05])
plt.xlabel('False Positive Rate', fontsize=12)
plt.ylabel('True Positive Rate', fontsize=12)
plt.title('ROC Curve - Patient Survival Prediction', fontsize=14, fontweight='bold')
plt.legend(loc="lower right")
plt.grid(alpha=0.3)
plt.tight_layout()
plt.savefig('roc_curve.png', dpi=300)
print("ROC curve saved as 'roc_curve.png'")
print()

# ==========================================
# ANALYSIS 7: Patient Demographic Distribution
# ==========================================
print("=" * 60)
print("ANALYSIS 7: Smoking Status Imbalance Ratio")
print("=" * 60)

# Load test advance data
df_test_demo = pd.read_excel('Testing_set_advance.xlsx')

# Group by smoking status
df_smoke = df_test_demo[['Patient_Smoker', 'Patient_Age']].dropna()
smoke_groups = df_smoke.groupby('Patient_Smoker').agg(
    count=('Patient_Age', 'count'),
    mean_age=('Patient_Age', 'mean')
)

# Calculate ratio
max_count = smoke_groups['count'].max()
min_count = smoke_groups['count'].min()
count_ratio = max_count / min_count

print(f"Smoking Status Count Ratio (Max/Min): {count_ratio:.2f}")
print()

# ==========================================
# ANALYSIS 8: Feature Correlation Analysis
# ==========================================
print("=" * 60)
print("ANALYSIS 8: Age-BMI Correlation Coefficient")
print("=" * 60)

# Load test intermediate data
df_test_corr = pd.read_excel('Testing_set_intermediate.xlsx')

# Calculate correlation
df_corr = df_test_corr[['Patient_Age', 'Patient_Body_Mass_Index']].dropna()
correlation = df_corr['Patient_Age'].corr(df_corr['Patient_Body_Mass_Index'])

print(f"Pearson Correlation (Age vs BMI): {correlation:.4f}")
print()

# ==========================================
# ANALYSIS 9: Age Distribution Comparison
# ==========================================
print("=" * 60)
print("ANALYSIS 9: Age Difference (Survivors vs Test Set)")
print("=" * 60)

# Training survivors mean age
df_train_surv = df[df['Survived_1_year'] == 1]['Patient_Age'].dropna()
mean_age_survivors = df_train_surv.mean()

# Testing mean age
df_test_age = df_test_int['Patient_Age'].dropna()
mean_age_testing = df_test_age.mean()

# Calculate difference
age_difference = abs(mean_age_survivors - mean_age_testing)

print(f"Training Survivor Mean Age: {mean_age_survivors:.2f} years")
print(f"Testing Mean Age: {mean_age_testing:.2f} years")
print(f"Absolute Age Difference: {age_difference:.2f} years")
print()

# ==========================================
# MODELING STRATEGY RECOMMENDATION
# ==========================================
print("=" * 60)
print("MODELING STRATEGY RECOMMENDATION")
print("=" * 60)

# Answer: Combine random forest for feature selection with gradient boosting for final prediction
# to maximize accuracy by leveraging both feature importance ranking and ensemble learning strength.
print("ANSWER: Combine random forest for feature selection with gradient boosting for final prediction")
print("to maximize accuracy by leveraging both feature importance ranking and ensemble learning strength.")
print()

# ==========================================
# KEY OUTPUTS SUMMARY
# ==========================================
print("=" * 60)
print("KEY OUTPUTS SUMMARY")
print("=" * 60)
print(f"1. Logistic Regression - Patient_Age Coefficient: {age_coefficient:.4f}")
print(f"2. Random Forest - Most Important Feature: {most_important_feature}")
print(f"3. XGBoost - Mean Cross-Validated AUC: {mean_cv_auc:.4f}")
print(f"4. SVM - Mean Predicted Survival Probability: {mean_survival_prob:.4f}")
print(f"5. KNN - Predicted Positive Rate: {positive_rate:.2f}%")
print(f"6. ROC Analysis - Validation AUC Score: {roc_auc:.4f}")
print(f"7. Demographic - Smoking Status Ratio: {count_ratio:.2f}")
print(f"8. Correlation - Age-BMI Coefficient: {correlation:.4f}")
print(f"9. Age Comparison - Absolute Difference: {age_difference:.2f} years")
print("=" * 60)
print("Analysis complete!")
