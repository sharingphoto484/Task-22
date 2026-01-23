# ==========================================
# Integrated Airline Operations Analysis Script
# Requirements: pandas, numpy, matplotlib, scipy, scikit-learn
# Input files: customer_airways_data.xlsx, filtered_customer_booking.xlsx, airline_data.xlsx
# Output files: confusion_matrix_svm.png, feature_importance_dt.png, model_comparison.png,
#               roc_curves_knn.png, mediation_path_diagram.png, bootstrap_histogram.png,
#               scree_plot.png, residual_plot.png
# ==========================================

import pandas as pd
import numpy as np
import matplotlib.pyplot as plt
import seaborn as sns
from sklearn.svm import SVC
from sklearn.tree import DecisionTreeClassifier
from sklearn.neighbors import KNeighborsClassifier
from sklearn.multiclass import OneVsRestClassifier
from sklearn.model_selection import train_test_split
from sklearn.metrics import accuracy_score, f1_score, confusion_matrix, roc_auc_score, roc_curve
from sklearn.linear_model import LinearRegression, LogisticRegression
from sklearn.decomposition import PCA
from sklearn.preprocessing import StandardScaler, label_binarize
from sklearn.metrics import mean_squared_error
from scipy import stats
import warnings
warnings.filterwarnings('ignore')

# ---------- Load Excel Files ----------
print("Loading data files...")
customer_airways = pd.read_excel('customer_airways_data.xlsx')
filtered_booking = pd.read_excel('filtered_customer_booking.xlsx')
airline_reviews = pd.read_excel('airline_data.xlsx')
print(f"Loaded {len(customer_airways)} customer airways records")
print(f"Loaded {len(filtered_booking)} booking records")
print(f"Loaded {len(airline_reviews)} review records")

# ==========================================
# ANALYSIS 1: SVM CLASSIFICATION FOR BOOKING COMPLETION
# ==========================================
print("\n" + "="*50)
print("ANALYSIS 1: SVM Classification")
print("="*50)

# ---------- Prepare SVM Dataset ----------
svm_data = filtered_booking[['flight_duration', 'num_passengers', 'length_of_stay', 'booking_complete']].dropna()
X_svm = svm_data[['flight_duration', 'num_passengers', 'length_of_stay']]
y_svm = svm_data['booking_complete']

# ---------- Train-Test Split (80/20) ----------
X_train_svm, X_test_svm, y_train_svm, y_test_svm = train_test_split(
    X_svm, y_svm, test_size=0.2, random_state=42
)

# ---------- Train SVM Model ----------
svm_model = SVC(kernel='rbf', C=1.0, gamma='scale', random_state=42)
svm_model.fit(X_train_svm, y_train_svm)

# ---------- Evaluate SVM Model ----------
y_pred_svm = svm_model.predict(X_test_svm)
svm_accuracy = accuracy_score(y_test_svm, y_pred_svm) * 100
svm_f1 = f1_score(y_test_svm, y_pred_svm, average='macro')

print(f"SVM Accuracy: {svm_accuracy:.2f}%")
print(f"SVM Macro F1: {svm_f1:.4f}")

# ---------- Confusion Matrix Visualization ----------
cm_svm = confusion_matrix(y_test_svm, y_pred_svm)
plt.figure(figsize=(8, 6))
sns.heatmap(cm_svm, annot=True, fmt='d', cmap='Blues', cbar=True)
plt.title('SVM Confusion Matrix')
plt.ylabel('Actual Booking Outcome')
plt.xlabel('Predicted Booking Outcome')
plt.tight_layout()
plt.savefig('confusion_matrix_svm.png', dpi=300, bbox_inches='tight')
plt.close()
print("Saved: confusion_matrix_svm.png")

# ==========================================
# ANALYSIS 2: DECISION TREE CLASSIFICATION
# ==========================================
print("\n" + "="*50)
print("ANALYSIS 2: Decision Tree Classification")
print("="*50)

# ---------- Prepare Decision Tree Dataset ----------
dt_features = ['num_passengers', 'purchase_lead', 'length_of_stay', 'flight_hour',
               'wants_extra_baggage', 'wants_preferred_seat', 'wants_in_flight_meals']
dt_data = filtered_booking[dt_features + ['booking_complete']].dropna()
X_dt = dt_data[dt_features]
y_dt = dt_data['booking_complete']

# ---------- Train-Test Split (80/20, Same Random State) ----------
X_train_dt, X_test_dt, y_train_dt, y_test_dt = train_test_split(
    X_dt, y_dt, test_size=0.2, random_state=42
)

# ---------- Train Decision Tree Model ----------
dt_model = DecisionTreeClassifier(max_depth=5, min_samples_split=100, random_state=42)
dt_model.fit(X_train_dt, y_train_dt)

# ---------- Evaluate Decision Tree Model ----------
y_pred_dt = dt_model.predict(X_test_dt)
dt_accuracy = accuracy_score(y_test_dt, y_pred_dt) * 100
dt_f1 = f1_score(y_test_dt, y_pred_dt, average='macro')

print(f"Decision Tree Accuracy: {dt_accuracy:.2f}%")
print(f"Decision Tree Macro F1: {dt_f1:.4f}")

# ---------- Feature Importance Visualization ----------
feature_importances = dt_model.feature_importances_
plt.figure(figsize=(10, 6))
plt.barh(dt_features, feature_importances)
plt.xlabel('Feature Contribution Magnitude')
plt.ylabel('Attribute Names')
plt.title('Decision Tree Feature Importance')
plt.tight_layout()
plt.savefig('feature_importance_dt.png', dpi=300, bbox_inches='tight')
plt.close()
print("Saved: feature_importance_dt.png")

# ---------- Model Comparison Visualization ----------
models = ['SVM', 'Decision Tree']
macro_f1_scores = [svm_f1, dt_f1]
accuracy_scores = [svm_accuracy, dt_accuracy]

x = np.arange(2)
width = 0.35

plt.figure(figsize=(10, 6))
plt.bar(x - width/2, macro_f1_scores, width, label='Macro F1')
plt.bar(x + width/2, [a/100 for a in accuracy_scores], width, label='Accuracy (scaled)')
plt.xlabel('Models')
plt.ylabel('Score')
plt.title('SVM vs Decision Tree Performance Comparison')
plt.xticks(x, models)
plt.legend()
plt.tight_layout()
plt.savefig('model_comparison.png', dpi=300, bbox_inches='tight')
plt.close()
print("Saved: model_comparison.png")

# ---------- Business Insight ----------
higher_f1_model = "Decision Tree" if dt_f1 > svm_f1 else "SVM"
insight = f"# Business Insight: {higher_f1_model} achieves higher macro F1 because it better handles class imbalance by learning decision boundaries that equally weight both booking completion and non-completion classes through hierarchical splits, while SVM optimizes global margin which may favor the majority class."
print(f"\n{insight}")

# ==========================================
# ANALYSIS 3: K-NN CLASSIFICATION WITH ONEVSREST
# ==========================================
print("\n" + "="*50)
print("ANALYSIS 3: K-NN Classification with OneVsRestClassifier")
print("="*50)

# ---------- Prepare K-NN Dataset ----------
knn_data = customer_airways[['purchase_lead', 'flight_duration', 'trip_type']].dropna()
X_knn = knn_data[['purchase_lead', 'flight_duration']]
y_knn = knn_data['trip_type']

# ---------- Train-Test Split (70/30) ----------
X_train_knn, X_test_knn, y_train_knn, y_test_knn = train_test_split(
    X_knn, y_knn, test_size=0.3, random_state=42
)

# ---------- Train K-NN Model with OneVsRestClassifier ----------
knn_base = KNeighborsClassifier(n_neighbors=5, weights='distance')
knn_model = OneVsRestClassifier(knn_base)
knn_model.fit(X_train_knn, y_train_knn)

# ---------- Evaluate K-NN Model ----------
y_pred_knn = knn_model.predict(X_test_knn)
knn_accuracy = accuracy_score(y_test_knn, y_pred_knn) * 100
print(f"K-NN Accuracy: {knn_accuracy:.2f}%")

# ---------- Calculate Per-Class ROC AUC ----------
classes = sorted(y_knn.unique())
y_test_binarized = label_binarize(y_test_knn, classes=classes)
y_proba_knn = knn_model.predict_proba(X_test_knn)

per_class_auc = {}
for i, cls in enumerate(classes):
    auc = roc_auc_score(y_test_binarized[:, i], y_proba_knn[:, i])
    per_class_auc[cls] = auc
    print(f"Trip Type {cls} AUC: {auc:.4f}")

# ---------- Micro-Average ROC AUC ----------
micro_auc = roc_auc_score(y_test_binarized, y_proba_knn, average='micro')
print(f"Micro-Average AUC: {micro_auc:.4f}")

# ---------- Identify Highest AUC Trip Type ----------
max_auc_value = max(per_class_auc.values())
candidates = [cls for cls, auc in per_class_auc.items() if abs(auc - max_auc_value) < 0.001]
highest_auc_trip = sorted(candidates)[0]  # Alphabetically first if tie
print(f"Highest AUC Trip Type: {highest_auc_trip}")

# ---------- ROC Curves Visualization ----------
plt.figure(figsize=(10, 8))
colors = ['blue', 'orange', 'green']
for i, (cls, color) in enumerate(zip(classes, colors)):
    fpr, tpr, _ = roc_curve(y_test_binarized[:, i], y_proba_knn[:, i])
    plt.plot(fpr, tpr, color=color, label=f'Trip Type {cls} (AUC = {per_class_auc[cls]:.4f})')

# Micro-average ROC curve
fpr_micro, tpr_micro, _ = roc_curve(y_test_binarized.ravel(), y_proba_knn.ravel())
plt.plot(fpr_micro, tpr_micro, color='red', linestyle='--',
         label=f'Micro-average (AUC = {micro_auc:.4f})')

plt.plot([0, 1], [0, 1], 'k--', linewidth=1)
plt.xlabel('False Positive Proportion')
plt.ylabel('True Positive Proportion')
plt.title('ROC Curves - K-NN Trip Type Classification')
plt.legend(loc='lower right')
plt.tight_layout()
plt.savefig('roc_curves_knn.png', dpi=300, bbox_inches='tight')
plt.close()
print("Saved: roc_curves_knn.png")

# ==========================================
# ANALYSIS 4: MEDIATION ANALYSIS
# ==========================================
print("\n" + "="*50)
print("ANALYSIS 4: Mediation Analysis")
print("="*50)

# ---------- Prepare Mediation Dataset ----------
mediation_data = filtered_booking[['purchase_lead', 'wants_in_flight_meals', 'booking_complete']].dropna()
X_med = mediation_data['purchase_lead'].values.reshape(-1, 1)
M_med = mediation_data['wants_in_flight_meals'].values
Y_med = mediation_data['booking_complete'].values

# ---------- Model 1: Total Effect (c) ----------
model_c = LogisticRegression(random_state=42, max_iter=1000)
model_c.fit(X_med, Y_med)
c = model_c.coef_[0][0]

# ---------- Model 2: Path a (X -> M) ----------
model_a = LogisticRegression(random_state=42, max_iter=1000)
model_a.fit(X_med, M_med)
a = model_a.coef_[0][0]

# ---------- Model 3: Direct Effect (c') and Path b ----------
X_med_with_M = np.column_stack([X_med, M_med])
model_b = LogisticRegression(random_state=42, max_iter=1000)
model_b.fit(X_med_with_M, Y_med)
c_prime = model_b.coef_[0][0]
b = model_b.coef_[0][1]

# ---------- Indirect Effect ----------
indirect_effect = a * b
print(f"Path a (X->M): {a:.4f}")
print(f"Path b (M->Y): {b:.4f}")
print(f"Direct effect c': {c_prime:.4f}")
print(f"Indirect effect (a*b): {indirect_effect:.4f}")

# ---------- Bootstrap Confidence Interval ----------
np.random.seed(42)
n_bootstrap = 1000
bootstrap_indirect = []

rng = np.random.RandomState(42)
for _ in range(n_bootstrap):
    indices = rng.choice(len(mediation_data), size=len(mediation_data), replace=True)
    X_boot = X_med[indices]
    M_boot = M_med[indices]
    Y_boot = Y_med[indices]

    # Path a
    model_a_boot = LogisticRegression(random_state=42, max_iter=1000)
    model_a_boot.fit(X_boot, M_boot)
    a_boot = model_a_boot.coef_[0][0]

    # Path b
    X_boot_with_M = np.column_stack([X_boot, M_boot])
    model_b_boot = LogisticRegression(random_state=42, max_iter=1000)
    model_b_boot.fit(X_boot_with_M, Y_boot)
    b_boot = model_b_boot.coef_[0][1]

    bootstrap_indirect.append(a_boot * b_boot)

# ---------- 95% Confidence Interval ----------
ci_lower = np.percentile(bootstrap_indirect, 2.5)
ci_upper = np.percentile(bootstrap_indirect, 97.5)
print(f"95% CI for indirect effect: [{ci_lower:.4f}, {ci_upper:.4f}]")

mediation_supported = "supported" if (ci_lower > 0 or ci_upper < 0) else "not supported"
print(f"Mediation is statistically {mediation_supported}")

# ---------- Path Diagram Visualization ----------
fig, ax = plt.subplots(figsize=(12, 6))
ax.set_xlim(0, 10)
ax.set_ylim(0, 10)
ax.axis('off')

# Nodes
ax.text(1, 5, 'purchase_lead', fontsize=12, ha='center', va='center',
        bbox=dict(boxstyle='round', facecolor='lightblue'))
ax.text(5, 8, 'wants_in_flight_meals', fontsize=12, ha='center', va='center',
        bbox=dict(boxstyle='round', facecolor='lightgreen'))
ax.text(9, 5, 'booking_complete', fontsize=12, ha='center', va='center',
        bbox=dict(boxstyle='round', facecolor='lightcoral'))

# Arrows
ax.annotate('', xy=(4.2, 7.7), xytext=(1.8, 5.3),
            arrowprops=dict(arrowstyle='->', lw=2, color='black'))
ax.text(2.8, 6.8, f'a = {a:.4f}', fontsize=10, ha='center')

ax.annotate('', xy=(8.2, 5.3), xytext=(5.8, 7.7),
            arrowprops=dict(arrowstyle='->', lw=2, color='black'))
ax.text(7.2, 6.8, f'b = {b:.4f}', fontsize=10, ha='center')

ax.annotate('', xy=(8, 5), xytext=(2, 5),
            arrowprops=dict(arrowstyle='->', lw=2, color='gray', linestyle='dashed'))
ax.text(5, 4.3, f"c' = {c_prime:.4f}", fontsize=10, ha='center')

plt.title('Mediation Path Diagram', fontsize=14, fontweight='bold')
plt.tight_layout()
plt.savefig('mediation_path_diagram.png', dpi=300, bbox_inches='tight')
plt.close()
print("Saved: mediation_path_diagram.png")

# ---------- Bootstrap Histogram ----------
plt.figure(figsize=(10, 6))
plt.hist(bootstrap_indirect, bins=30, edgecolor='black', alpha=0.7)
plt.axvline(indirect_effect, color='red', linestyle='--', linewidth=2, label='Point Estimate')
plt.xlabel('Bootstrapped Indirect Effect')
plt.ylabel('Frequency')
plt.title('Distribution of Bootstrapped Indirect Effects')
plt.legend()
plt.tight_layout()
plt.savefig('bootstrap_histogram.png', dpi=300, bbox_inches='tight')
plt.close()
print("Saved: bootstrap_histogram.png")

# ==========================================
# ANALYSIS 5: PRINCIPAL COMPONENT ANALYSIS
# ==========================================
print("\n" + "="*50)
print("ANALYSIS 5: Principal Component Analysis")
print("="*50)

# ---------- Prepare PCA Dataset ----------
pca_features = ['num_passengers', 'purchase_lead', 'length_of_stay', 'flight_hour',
                'flight_duration', 'wants_extra_baggage', 'wants_preferred_seat',
                'wants_in_flight_meals']
pca_data = filtered_booking[pca_features].dropna()

# ---------- Standardize Features ----------
scaler = StandardScaler()
pca_scaled = scaler.fit_transform(pca_data)

# ---------- Apply PCA ----------
pca = PCA(n_components=5, svd_solver='full')
pca.fit(pca_scaled)

# ---------- Extract Variance Metrics ----------
first_component_variance = pca.explained_variance_ratio_[0]
cumulative_variance = np.cumsum(pca.explained_variance_ratio_)
print(f"First Component Variance: {first_component_variance:.4f}")
print(f"Cumulative Variance (5 components): {cumulative_variance[-1]:.4f}")

# ---------- Components Needed for 80% Variance ----------
components_for_80 = np.argmax(cumulative_variance >= 0.80) + 1 if any(cumulative_variance >= 0.80) else None
if components_for_80:
    print(f"Components needed for 80% variance: {components_for_80}")
else:
    print("Components needed for 80% variance: none within first 5")

# ---------- Scree Plot ----------
fig, ax1 = plt.subplots(figsize=(10, 6))

component_indices = np.arange(1, 6)
variance_pct = pca.explained_variance_ratio_ * 100
cumulative_pct = cumulative_variance * 100

color = 'tab:blue'
ax1.set_xlabel('Component Index')
ax1.set_ylabel('Individual Variance Explained (%)', color=color)
ax1.plot(component_indices, variance_pct, marker='o', color=color, label='Individual Variance')
ax1.tick_params(axis='y', labelcolor=color)

ax2 = ax1.twinx()
color = 'tab:red'
ax2.set_ylabel('Cumulative Variance Explained (%)', color=color)
ax2.plot(component_indices, cumulative_pct, marker='s', color=color, label='Cumulative Variance')
ax2.tick_params(axis='y', labelcolor=color)

plt.title('PCA Scree Plot')
fig.tight_layout()
plt.savefig('scree_plot.png', dpi=300, bbox_inches='tight')
plt.close()
print("Saved: scree_plot.png")

# ==========================================
# ANALYSIS 6: MULTIPLE LINEAR REGRESSION
# ==========================================
print("\n" + "="*50)
print("ANALYSIS 6: Multiple Linear Regression")
print("="*50)

# ---------- Prepare Regression Dataset ----------
reg_data = customer_airways[['num_passengers', 'purchase_lead', 'length_of_stay', 'flight_duration']].dropna()
X_reg = reg_data[['num_passengers', 'purchase_lead', 'length_of_stay']]
y_reg = reg_data['flight_duration']

# ---------- Train-Test Split (80/20) ----------
X_train_reg, X_test_reg, y_train_reg, y_test_reg = train_test_split(
    X_reg, y_reg, test_size=0.2, random_state=42
)

# ---------- Train Linear Regression Model ----------
lr_model = LinearRegression()
lr_model.fit(X_train_reg, y_train_reg)

# ---------- Evaluate Model ----------
y_pred_reg = lr_model.predict(X_test_reg)
rmse = np.sqrt(mean_squared_error(y_test_reg, y_pred_reg))
print(f"RMSE: {rmse:.3f}")

# ---------- Residual Plot ----------
residuals = y_test_reg - y_pred_reg
plt.figure(figsize=(10, 6))
plt.scatter(y_pred_reg, residuals, alpha=0.5)
plt.axhline(y=0, color='red', linestyle='--', linewidth=2)
plt.xlabel('Predicted Flight Duration')
plt.ylabel('Residuals (Observed - Predicted)')
plt.title('Residual Plot - Linear Regression')
plt.tight_layout()
plt.savefig('residual_plot.png', dpi=300, bbox_inches='tight')
plt.close()
print("Saved: residual_plot.png")

# ==========================================
# ANALYSIS 7: ONE-WAY ANOVA
# ==========================================
print("\n" + "="*50)
print("ANALYSIS 7: One-Way ANOVA")
print("="*50)

# ---------- Prepare ANOVA Dataset ----------
anova_data = customer_airways[['purchase_lead', 'sales_channel']].dropna()
groups = [group['purchase_lead'].values for name, group in anova_data.groupby('sales_channel')]

# ---------- Perform ANOVA ----------
f_statistic, p_value = stats.f_oneway(*groups)
print(f"F-statistic: {f_statistic:.2f}")
print(f"P-value: {p_value:.6f}")

# ==========================================
# ANALYSIS 8: STANDARD ERROR BY TRIP TYPE
# ==========================================
print("\n" + "="*50)
print("ANALYSIS 8: Standard Error by Trip Type")
print("="*50)

# ---------- Prepare SEM Dataset ----------
sem_data = filtered_booking[['flight_duration', 'trip_type']].dropna()

# ---------- Calculate SEM for Each Trip Type ----------
sem_results = {}
for trip_type in sem_data['trip_type'].unique():
    trip_data = sem_data[sem_data['trip_type'] == trip_type]['flight_duration']
    std_dev = trip_data.std()
    n = len(trip_data)
    sem = std_dev / np.sqrt(n)
    sem_results[trip_type] = sem
    print(f"Trip Type {trip_type} SEM: {sem:.4f}")

# ---------- Identify Maximum SEM ----------
max_sem_trip = max(sem_results, key=sem_results.get)
max_sem_value = sem_results[max_sem_trip]
print(f"Maximum SEM: {max_sem_value:.4f} (Trip Type: {max_sem_trip})")

# ==========================================
# ANALYSIS 9: PERCENTILE RANK
# ==========================================
print("\n" + "="*50)
print("ANALYSIS 9: Percentile Rank")
print("="*50)

# ---------- Calculate Percentile Rank for flight_hour = 12 ----------
flight_hours = customer_airways['flight_hour'].dropna()
count_below = (flight_hours < 12).sum()
total_count = len(flight_hours)
percentile_rank = int((count_below / total_count) * 100)
print(f"Percentile rank for flight_hour = 12: {percentile_rank}%")

# ==========================================
# ANALYSIS 10: CORRELATION ANALYSIS
# ==========================================
print("\n" + "="*50)
print("ANALYSIS 10: Correlation Analysis")
print("="*50)

# ---------- Prepare Correlation Dataset ----------
corr_data = airline_reviews[['rates', 'reviews']].dropna()
corr_data['review_length'] = corr_data['reviews'].astype(str).str.len()

# ---------- Calculate Pearson Correlation ----------
correlation = corr_data[['rates', 'review_length']].corr().iloc[0, 1]
print(f"Correlation between satisfaction and review length: {correlation:.4f}")

# ==========================================
# FINAL SUMMARY
# ==========================================
print("\n" + "="*50)
print("ANALYSIS COMPLETE")
print("="*50)
print("\nKey Outputs Generated:")
print("1. SVM Accuracy: {:.2f}%, Macro F1: {:.4f}".format(svm_accuracy, svm_f1))
print("2. Decision Tree Accuracy: {:.2f}%, Macro F1: {:.4f}".format(dt_accuracy, dt_f1))
print("3. K-NN Accuracy: {:.2f}%, Highest AUC Trip Type: {}".format(knn_accuracy, highest_auc_trip))
print("4. Mediation Indirect Effect: {:.4f}, Status: {}".format(indirect_effect, mediation_supported))
print("5. PCA First Component Variance: {:.4f}, Cumulative: {:.4f}".format(first_component_variance, cumulative_variance[-1]))
print("6. Linear Regression RMSE: {:.3f}".format(rmse))
print("7. ANOVA F-statistic: {:.2f}".format(f_statistic))
print("8. Maximum SEM: {:.4f}".format(max_sem_value))
print("9. Percentile rank (flight_hour=12): {}%".format(percentile_rank))
print("10. Correlation (satisfaction vs review length): {:.4f}".format(correlation))

print("\n" + insight)
