# ==========================================
# Patient Survival Prediction Visualization Script
# ==========================================
# Requirements: pandas, numpy, matplotlib, scipy, scikit-learn, xgboost, openpyxl
# Input files: Training_set_advance.xlsx, Testing_set_advance.xlsx, Testing_set_intermediate.xlsx
# Output files: coefficient_plot.png, feature_importance_plot.png, xgboost_learning_curve.png,
#               svm_probability_histogram.png, knn_scatter_plot.png, roc_curve_plot.png
# ==========================================

import pandas as pd
import numpy as np
import matplotlib.pyplot as plt
from sklearn.linear_model import LogisticRegression
from sklearn.ensemble import RandomForestClassifier
from sklearn.svm import SVC
from sklearn.neighbors import KNeighborsClassifier
from sklearn.preprocessing import StandardScaler
from sklearn.model_selection import train_test_split
from sklearn.metrics import roc_curve, auc
from xgboost import XGBClassifier
import warnings
warnings.filterwarnings('ignore')

# ---------- Load Datasets ----------
print("Loading datasets...")
train_df = pd.read_excel('Training_set_advance.xlsx')
test_advance_df = pd.read_excel('Testing_set_advance.xlsx')
test_intermediate_df = pd.read_excel('Testing_set_intermediate.xlsx')

print("Generating all plots...\n")

# ---------- Plot 1: Logistic Regression Coefficient Plot ----------
print("Creating Plot 1: Logistic Regression Coefficients...")

binary_indicators = ['A', 'B', 'C', 'D', 'E', 'F']
lr_features = ['Patient_Age', 'Patient_Body_Mass_Index'] + binary_indicators
lr_data = train_df[lr_features + ['Survived_1_year']].dropna()

scaler_lr = StandardScaler()
X_lr = lr_data[lr_features].copy()
X_lr[['Patient_Age', 'Patient_Body_Mass_Index']] = scaler_lr.fit_transform(
    X_lr[['Patient_Age', 'Patient_Body_Mass_Index']]
)
y_lr = lr_data['Survived_1_year']

lr_model = LogisticRegression(penalty='l2', C=1, random_state=42, max_iter=1000)
lr_model.fit(X_lr, y_lr)

# Create coefficient plot
plt.figure(figsize=(10, 6))
coefficients = lr_model.coef_[0]
predictor_names = ['Patient_Age', 'Patient_Body_Mass_Index', 'A', 'B', 'C', 'D', 'E', 'F']
plt.bar(predictor_names, coefficients, color='steelblue', edgecolor='black')
plt.xlabel('Predictor Names', fontsize=12, fontweight='bold')
plt.ylabel('Standardized Coefficient Values', fontsize=12, fontweight='bold')
plt.title('Logistic Regression Coefficients for Patient Survival Prediction', fontsize=14, fontweight='bold')
plt.xticks(rotation=45, ha='right')
plt.axhline(y=0, color='red', linestyle='--', linewidth=1)
plt.grid(axis='y', alpha=0.3)
plt.tight_layout()
plt.savefig('coefficient_plot.png', dpi=300, bbox_inches='tight')
print("Saved: coefficient_plot.png\n")
plt.close()

# ---------- Plot 2: Random Forest Feature Importance (Horizontal Bar Chart) ----------
print("Creating Plot 2: Random Forest Feature Importance...")

rf_features = ['Patient_Age', 'Patient_Body_Mass_Index', 'Diagnosed_Condition',
               'Number_of_prev_cond'] + binary_indicators
rf_data = train_df[rf_features + ['Survived_1_year']].dropna()

X_rf = rf_data[rf_features]
y_rf = rf_data['Survived_1_year']

rf_model = RandomForestClassifier(n_estimators=100, max_depth=10, random_state=42)
rf_model.fit(X_rf, y_rf)

# Create horizontal bar chart
feature_importances = pd.DataFrame({
    'feature': rf_features,
    'importance': rf_model.feature_importances_
}).sort_values('importance', ascending=True)

plt.figure(figsize=(10, 8))
plt.barh(feature_importances['feature'], feature_importances['importance'],
         color='forestgreen', edgecolor='black')
plt.xlabel('Importance Scores (0 to 1)', fontsize=12, fontweight='bold')
plt.ylabel('Feature Names', fontsize=12, fontweight='bold')
plt.title('Random Forest Feature Importance for Patient Survival Prediction', fontsize=14, fontweight='bold')
plt.xlim(0, 1)
plt.grid(axis='x', alpha=0.3)
plt.tight_layout()
plt.savefig('feature_importance_plot.png', dpi=300, bbox_inches='tight')
print("Saved: feature_importance_plot.png\n")
plt.close()

# ---------- Plot 3: XGBoost Learning Curve (Boosting Iteration vs Error) ----------
print("Creating Plot 3: XGBoost Learning Curve...")

xgb_features = ['Patient_Age', 'Patient_Body_Mass_Index'] + binary_indicators
xgb_data = train_df[xgb_features + ['Survived_1_year']].dropna()

X_xgb = xgb_data[xgb_features]
y_xgb = xgb_data['Survived_1_year']

# Split for train and validation
X_train_xgb, X_val_xgb, y_train_xgb, y_val_xgb = train_test_split(
    X_xgb, y_xgb, test_size=0.2, random_state=42
)

xgb_model = XGBClassifier(learning_rate=0.1, max_depth=5, n_estimators=100,
                          random_state=42, eval_metric='logloss')

# Fit with eval_set to track training progress
eval_set = [(X_train_xgb, y_train_xgb), (X_val_xgb, y_val_xgb)]
xgb_model.fit(X_train_xgb, y_train_xgb, eval_set=eval_set, verbose=False)

# Extract results
results = xgb_model.evals_result()
train_errors = results['validation_0']['logloss']
val_errors = results['validation_1']['logloss']
iterations = range(1, len(train_errors) + 1)

plt.figure(figsize=(10, 6))
plt.plot(iterations, train_errors, label='Training Error', color='blue', linewidth=2)
plt.plot(iterations, val_errors, label='Validation Error', color='red', linewidth=2)
plt.xlabel('Boosting Iteration', fontsize=12, fontweight='bold')
plt.ylabel('Log Loss Error Values', fontsize=12, fontweight='bold')
plt.title('XGBoost Learning Curve: Error vs Boosting Iterations', fontsize=14, fontweight='bold')
plt.legend(loc='upper right', fontsize=10)
plt.grid(alpha=0.3)
plt.tight_layout()
plt.savefig('xgboost_learning_curve.png', dpi=300, bbox_inches='tight')
print("Saved: xgboost_learning_curve.png\n")
plt.close()

# ---------- Plot 4: SVM Predicted Probability Histogram ----------
print("Creating Plot 4: SVM Probability Histogram...")

svm_features = ['Patient_Age', 'Patient_Body_Mass_Index'] + binary_indicators
svm_train_data = train_df[svm_features + ['Survived_1_year']].dropna()

scaler_svm = StandardScaler()
X_svm_train = scaler_svm.fit_transform(svm_train_data[svm_features])
y_svm_train = svm_train_data['Survived_1_year']

svm_model = SVC(kernel='rbf', gamma=0.1, C=1, random_state=42, probability=True)
svm_model.fit(X_svm_train, y_svm_train)

svm_test_data = test_advance_df[svm_features].dropna()
X_svm_test = scaler_svm.transform(svm_test_data)
svm_predictions = svm_model.predict_proba(X_svm_test)[:, 1]

plt.figure(figsize=(10, 6))
plt.hist(svm_predictions, bins=30, range=(0, 1), color='purple', edgecolor='black', alpha=0.7)
plt.xlabel('Predicted Survival Probability (0 to 1)', fontsize=12, fontweight='bold')
plt.ylabel('Frequency Count', fontsize=12, fontweight='bold')
plt.title('SVM Predicted Survival Probability Distribution', fontsize=14, fontweight='bold')
plt.xlim(0, 1)
plt.grid(axis='y', alpha=0.3)
plt.tight_layout()
plt.savefig('svm_probability_histogram.png', dpi=300, bbox_inches='tight')
print("Saved: svm_probability_histogram.png\n")
plt.close()

# ---------- Plot 5: KNN Scatter Plot (Age vs BMI) ----------
print("Creating Plot 5: KNN Scatter Plot...")

knn_features = ['Patient_Age', 'Patient_Body_Mass_Index', 'Diagnosed_Condition'] + binary_indicators
knn_train_data = train_df[knn_features + ['Survived_1_year']].dropna()

scaler_knn = StandardScaler()
X_knn_train = scaler_knn.fit_transform(knn_train_data[knn_features])
y_knn_train = knn_train_data['Survived_1_year']

knn_model = KNeighborsClassifier(n_neighbors=5, metric='euclidean')
knn_model.fit(X_knn_train, y_knn_train)

knn_test_data = test_intermediate_df[knn_features].dropna()
X_knn_test = scaler_knn.transform(knn_test_data)
knn_predictions = knn_model.predict(X_knn_test)

# Get original age and BMI values
age_values = knn_test_data['Patient_Age'].values
bmi_values = knn_test_data['Patient_Body_Mass_Index'].values

plt.figure(figsize=(10, 6))
survived = knn_predictions == 1
not_survived = knn_predictions == 0

plt.scatter(age_values[survived], bmi_values[survived],
           c='green', label='Predicted Survived', alpha=0.6, s=50, edgecolors='black')
plt.scatter(age_values[not_survived], bmi_values[not_survived],
           c='red', label='Predicted Not Survived', alpha=0.6, s=50, edgecolors='black')
plt.xlabel('Patient Age', fontsize=12, fontweight='bold')
plt.ylabel('Patient Body Mass Index', fontsize=12, fontweight='bold')
plt.title('KNN Predictions: Age vs BMI Scatter Plot', fontsize=14, fontweight='bold')
plt.legend(loc='upper right', fontsize=10)
plt.grid(alpha=0.3)
plt.tight_layout()
plt.savefig('knn_scatter_plot.png', dpi=300, bbox_inches='tight')
print("Saved: knn_scatter_plot.png\n")
plt.close()

# ---------- Plot 6: ROC Curve (True Positive Rate vs False Positive Rate) ----------
print("Creating Plot 6: ROC Curve...")

roc_features = ['Patient_Age', 'Patient_Body_Mass_Index'] + binary_indicators
roc_data = train_df[roc_features + ['Survived_1_year']].dropna()

X_roc = roc_data[roc_features]
y_roc = roc_data['Survived_1_year']

X_train_roc, X_val_roc, y_train_roc, y_val_roc = train_test_split(
    X_roc, y_roc, test_size=0.2, random_state=42
)

lr_roc_model = LogisticRegression(random_state=42, max_iter=1000)
lr_roc_model.fit(X_train_roc, y_train_roc)

y_val_pred_proba = lr_roc_model.predict_proba(X_val_roc)[:, 1]

fpr, tpr, thresholds = roc_curve(y_val_roc, y_val_pred_proba)
roc_auc = auc(fpr, tpr)

plt.figure(figsize=(10, 8))
plt.plot(fpr, tpr, color='darkorange', linewidth=3, label=f'ROC Curve (AUC = {roc_auc:.4f})')
plt.plot([0, 1], [0, 1], color='navy', linewidth=2, linestyle='--', label='Random Classifier')
plt.xlabel('False Positive Rate', fontsize=12, fontweight='bold')
plt.ylabel('True Positive Rate', fontsize=12, fontweight='bold')
plt.title('ROC Curve: Logistic Regression Patient Survival Prediction', fontsize=14, fontweight='bold')
plt.legend(loc='lower right', fontsize=12)
plt.grid(alpha=0.3)
plt.xlim([0.0, 1.0])
plt.ylim([0.0, 1.05])
plt.tight_layout()
plt.savefig('roc_curve_plot.png', dpi=300, bbox_inches='tight')
print("Saved: roc_curve_plot.png\n")
plt.close()

print("=" * 60)
print("ALL PLOTS GENERATED SUCCESSFULLY")
print("=" * 60)
print("\nGenerated Files:")
print("1. coefficient_plot.png - Logistic Regression Coefficients")
print("2. feature_importance_plot.png - Random Forest Feature Importance")
print("3. xgboost_learning_curve.png - XGBoost Learning Curve")
print("4. svm_probability_histogram.png - SVM Probability Distribution")
print("5. knn_scatter_plot.png - KNN Age vs BMI Predictions")
print("6. roc_curve_plot.png - ROC Curve Analysis")
