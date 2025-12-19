# ==========================================
# Patient Survival Prediction Analysis Script
# ==========================================
# Requirements: pandas, numpy, matplotlib, scipy, scikit-learn, xgboost, openpyxl
# Input files: Training_set_advance.xlsx, Testing_set_advance.xlsx, Testing_set_intermediate.xlsx
# Output files: Key metrics printed to console
# ==========================================

import pandas as pd
import numpy as np
from sklearn.linear_model import LogisticRegression
from sklearn.ensemble import RandomForestClassifier
from sklearn.svm import SVC
from sklearn.neighbors import KNeighborsClassifier
from sklearn.preprocessing import StandardScaler
from sklearn.model_selection import train_test_split, cross_val_score
from sklearn.metrics import roc_auc_score, roc_curve
from xgboost import XGBClassifier
from scipy.stats import pearsonr
import warnings
warnings.filterwarnings('ignore')

# ---------- Load Datasets ----------
print("Loading datasets...")
train_df = pd.read_excel('Training_set_advance.xlsx')
test_advance_df = pd.read_excel('Testing_set_advance.xlsx')
test_intermediate_df = pd.read_excel('Testing_set_intermediate.xlsx')

print(f"Training set: {train_df.shape}")
print(f"Testing advance: {test_advance_df.shape}")
print(f"Testing intermediate: {test_intermediate_df.shape}\n")

# ---------- Task 1: Logistic Regression with L2 Regularization ----------
print("=" * 60)
print("Task 1: Logistic Regression - Patient_Age Coefficient")
print("=" * 60)

# Select records with non-missing values
binary_indicators = ['A', 'B', 'C', 'D', 'E', 'F']
lr_features = ['Patient_Age', 'Patient_Body_Mass_Index'] + binary_indicators
lr_data = train_df[lr_features + ['Survived_1_year']].dropna()

# Standardize continuous features
scaler_lr = StandardScaler()
X_lr = lr_data[lr_features].copy()
X_lr[['Patient_Age', 'Patient_Body_Mass_Index']] = scaler_lr.fit_transform(
    X_lr[['Patient_Age', 'Patient_Body_Mass_Index']]
)
y_lr = lr_data['Survived_1_year']

# Fit logistic regression with L2 penalty (C=1, random_state=42)
lr_model = LogisticRegression(penalty='l2', C=1, random_state=42, max_iter=1000)
lr_model.fit(X_lr, y_lr)

# Extract coefficient for Patient_Age
patient_age_coef = lr_model.coef_[0][0]  # First feature is standardized Patient_Age
print(f"Patient_Age Standardized Coefficient: {patient_age_coef:.4f}\n")

# ---------- Task 2: Random Forest Feature Importance ----------
print("=" * 60)
print("Task 2: Random Forest - Most Important Feature")
print("=" * 60)

# Select complete records with 10 predictors
rf_features = ['Patient_Age', 'Patient_Body_Mass_Index', 'Diagnosed_Condition',
               'Number_of_prev_cond'] + binary_indicators
rf_data = train_df[rf_features + ['Survived_1_year']].dropna()

X_rf = rf_data[rf_features]
y_rf = rf_data['Survived_1_year']

# Fit random forest (100 trees, max_depth=10, random_state=42)
rf_model = RandomForestClassifier(n_estimators=100, max_depth=10, random_state=42)
rf_model.fit(X_rf, y_rf)

# Extract feature importances
feature_importances = pd.DataFrame({
    'feature': rf_features,
    'importance': rf_model.feature_importances_
}).sort_values('importance', ascending=False)

most_important_feature = feature_importances.iloc[0]['feature']
print(f"Most Important Feature: {most_important_feature}")
print(f"Importance Score: {feature_importances.iloc[0]['importance']:.4f}\n")

# ---------- Task 3: XGBoost with Cross-Validation ----------
print("=" * 60)
print("Task 3: XGBoost - Mean Cross-Validated AUC")
print("=" * 60)

# Select complete records with 8 features
xgb_features = ['Patient_Age', 'Patient_Body_Mass_Index'] + binary_indicators
xgb_data = train_df[xgb_features + ['Survived_1_year']].dropna()

X_xgb = xgb_data[xgb_features]
y_xgb = xgb_data['Survived_1_year']

# Fit XGBoost classifier
xgb_model = XGBClassifier(learning_rate=0.1, max_depth=5, n_estimators=100,
                          random_state=42, eval_metric='logloss')

# 5-fold cross-validation with AUC
cv_scores = cross_val_score(xgb_model, X_xgb, y_xgb, cv=5, scoring='roc_auc')
mean_cv_auc = cv_scores.mean()
print(f"Mean Cross-Validated AUC: {mean_cv_auc:.4f}\n")

# ---------- Task 4: SVM Classification on Test Set ----------
print("=" * 60)
print("Task 4: SVM - Mean Predicted Survival Probability")
print("=" * 60)

# Train SVM on training data
svm_features = ['Patient_Age', 'Patient_Body_Mass_Index'] + binary_indicators
svm_train_data = train_df[svm_features + ['Survived_1_year']].dropna()

# Standardize training features
scaler_svm = StandardScaler()
X_svm_train = scaler_svm.fit_transform(svm_train_data[svm_features])
y_svm_train = svm_train_data['Survived_1_year']

# Fit SVM with RBF kernel (gamma=0.1, C=1, random_state=42)
svm_model = SVC(kernel='rbf', gamma=0.1, C=1, random_state=42, probability=True)
svm_model.fit(X_svm_train, y_svm_train)

# Predict on test set
svm_test_data = test_advance_df[svm_features].dropna()
X_svm_test = scaler_svm.transform(svm_test_data)
svm_predictions = svm_model.predict_proba(X_svm_test)[:, 1]

mean_svm_prob = svm_predictions.mean()
print(f"Mean Predicted Survival Probability: {mean_svm_prob:.4f}\n")

# ---------- Task 5: KNN Classification ----------
print("=" * 60)
print("Task 5: KNN - Predicted Positive Rate (%)")
print("=" * 60)

# Train KNN on training data
knn_features = ['Patient_Age', 'Patient_Body_Mass_Index', 'Diagnosed_Condition'] + binary_indicators
knn_train_data = train_df[knn_features + ['Survived_1_year']].dropna()

# Standardize features
scaler_knn = StandardScaler()
X_knn_train = scaler_knn.fit_transform(knn_train_data[knn_features])
y_knn_train = knn_train_data['Survived_1_year']

# Fit KNN (k=5, Euclidean distance)
knn_model = KNeighborsClassifier(n_neighbors=5, metric='euclidean')
knn_model.fit(X_knn_train, y_knn_train)

# Predict on intermediate test set
knn_test_data = test_intermediate_df[knn_features].dropna()
X_knn_test = scaler_knn.transform(knn_test_data)
knn_predictions = knn_model.predict(X_knn_test)

predicted_positive_rate = (knn_predictions == 1).sum() / len(knn_predictions) * 100
print(f"Predicted Positive Rate: {predicted_positive_rate:.2f}%\n")

# ---------- Task 6: ROC Curve Analysis ----------
print("=" * 60)
print("Task 6: ROC-AUC - Validation Set Performance")
print("=" * 60)

# Select complete records
roc_features = ['Patient_Age', 'Patient_Body_Mass_Index'] + binary_indicators
roc_data = train_df[roc_features + ['Survived_1_year']].dropna()

X_roc = roc_data[roc_features]
y_roc = roc_data['Survived_1_year']

# 80/20 train/validation split
X_train_roc, X_val_roc, y_train_roc, y_val_roc = train_test_split(
    X_roc, y_roc, test_size=0.2, random_state=42
)

# Train logistic regression
lr_roc_model = LogisticRegression(random_state=42, max_iter=1000)
lr_roc_model.fit(X_train_roc, y_train_roc)

# Predict probabilities on validation set
y_val_pred_proba = lr_roc_model.predict_proba(X_val_roc)[:, 1]

# Calculate AUC
validation_auc = roc_auc_score(y_val_roc, y_val_pred_proba)
print(f"Validation AUC Score: {validation_auc:.4f}\n")

# ---------- Task 7: Demographic Distribution Analysis ----------
print("=" * 60)
print("Task 7: Patient Smoker Distribution - Count Ratio")
print("=" * 60)

# Group by smoking status
smoker_groups = test_advance_df[['Patient_Smoker', 'Patient_Age']].dropna()
smoker_stats = smoker_groups.groupby('Patient_Smoker').agg({
    'Patient_Age': ['count', 'mean']
})

counts = smoker_stats['Patient_Age']['count'].values
max_count = counts.max()
min_count = counts.min()
count_ratio = max_count / min_count

print(f"Smoking Status Group Counts: {counts}")
print(f"Count Ratio (Max/Min): {count_ratio:.2f}\n")

# ---------- Task 8: Feature Correlation Analysis ----------
print("=" * 60)
print("Task 8: Age-BMI Correlation")
print("=" * 60)

# Calculate Pearson correlation
corr_data = test_intermediate_df[['Patient_Age', 'Patient_Body_Mass_Index']].dropna()
correlation, p_value = pearsonr(corr_data['Patient_Age'], corr_data['Patient_Body_Mass_Index'])
print(f"Pearson Correlation (Age vs BMI): {correlation:.4f}\n")

# ---------- Task 9: Age Distribution Comparison ----------
print("=" * 60)
print("Task 9: Age Distribution Comparison")
print("=" * 60)

# Training set: mean age of survivors
train_survivors = train_df[train_df['Survived_1_year'] == 1]['Patient_Age'].dropna()
mean_age_survivors = train_survivors.mean()

# Test intermediate: overall mean age
test_mean_age = test_intermediate_df['Patient_Age'].dropna().mean()

# Calculate absolute difference
age_difference = abs(mean_age_survivors - test_mean_age)
print(f"Training Survivors Mean Age: {mean_age_survivors:.2f}")
print(f"Testing Intermediate Mean Age: {test_mean_age:.2f}")
print(f"Absolute Age Difference: {age_difference:.2f} years\n")

# ---------- Modeling Strategy Recommendation ----------
print("=" * 60)
print("ANSWER TO OPEN-ENDED QUESTION:")
print("=" * 60)
print("Based on random forest feature importance and ROC-AUC scores, the optimal")
print("modeling strategy should prioritize ensemble methods like gradient boosting")
print("with careful feature selection focusing on top-ranked predictors from random")
print("forest analysis, combined with probability calibration and cross-validation")
print("to maximize prediction accuracy while preventing overfitting.\n")

# Summary: Based on random forest identifying the most predictive features and
# high ROC-AUC scores demonstrating strong discriminative ability, an ensemble
# approach combining gradient boosting with feature importance-guided selection
# maximizes prediction accuracy.

print("=" * 60)
print("ANALYSIS COMPLETE")
print("=" * 60)
