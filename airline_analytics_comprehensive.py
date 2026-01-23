# ==========================================
# Integrated Airline Analytics Script
# Requirements: pandas, numpy, matplotlib, scipy, scikit-learn
# Input files: customer_airways_data.xlsx, filtered_customer_booking.xlsx, airline_data.xlsx
# Output files: feature_importance.png, confusion_matrix.png, roc_curve_multiclass.png,
#               cluster_scatter_matrix.png, scree_plot.png, residual_plot.png
# ==========================================

import pandas as pd
import numpy as np
import matplotlib.pyplot as plt
import seaborn as sns
from sklearn.tree import DecisionTreeClassifier
from sklearn.svm import SVC
from sklearn.neighbors import KNeighborsClassifier
from sklearn.mixture import GaussianMixture
from sklearn.decomposition import PCA
from sklearn.linear_model import LinearRegression
from sklearn.model_selection import train_test_split
from sklearn.preprocessing import StandardScaler, label_binarize
from sklearn.metrics import accuracy_score, f1_score, confusion_matrix, silhouette_score, mean_squared_error, roc_curve, auc
from scipy.stats import f_oneway
import warnings
warnings.filterwarnings('ignore')

# ==========================================
# PROCEDURE 1: Decision Tree Classification for Booking Completion
# ==========================================

print("=" * 60)
print("PROCEDURE 1: Decision Tree Classification - Booking Completion")
print("=" * 60)

# ---------- Load Filtered Booking Data ----------
df_booking = pd.read_excel('filtered_customer_booking.xlsx')

# ---------- Construct Target and Features ----------
target = 'booking_complete'
features = ['num_passengers', 'purchase_lead', 'length_of_stay', 'flight_hour',
            'wants_extra_baggage', 'wants_preferred_seat', 'wants_in_flight_meals']

# ---------- Remove Missing Observations ----------
df_booking_clean = df_booking[features + [target]].dropna()

X = df_booking_clean[features]
y = df_booking_clean[target]

# ---------- Train-Test Split (80-20) ----------
X_train, X_test, y_train, y_test = train_test_split(X, y, test_size=0.2, random_state=42)

# ---------- Train Decision Tree Classifier ----------
dt_classifier = DecisionTreeClassifier(max_depth=5, min_samples_split=100, random_state=42)
dt_classifier.fit(X_train, y_train)

# ---------- Evaluate Performance ----------
y_pred_dt = dt_classifier.predict(X_test)
dt_accuracy = accuracy_score(y_test, y_pred_dt) * 100

print(f"\nDecision Tree Accuracy: {dt_accuracy:.2f}%")

# ---------- Feature Importance Visualization ----------
feature_importances = dt_classifier.feature_importances_
importance_df = pd.DataFrame({'Feature': features, 'Importance': feature_importances})
importance_df = importance_df.sort_values('Importance')

plt.figure(figsize=(10, 6))
plt.barh(importance_df['Feature'], importance_df['Importance'], color='steelblue')
plt.xlabel('Importance', fontsize=12)
plt.ylabel('Feature', fontsize=12)
plt.title('Decision Tree Feature Importance', fontsize=14, fontweight='bold')
plt.tight_layout()
plt.savefig('feature_importance.png', dpi=300, bbox_inches='tight')
plt.close()
print("Feature importance chart saved: feature_importance.png")

# ==========================================
# PROCEDURE 2: SVM Classification for Extra Baggage Preference
# ==========================================

print("\n" + "=" * 60)
print("PROCEDURE 2: SVM Classification - Extra Baggage Preference")
print("=" * 60)

# ---------- Load Customer Airways Data ----------
df_customer = pd.read_excel('customer_airways_data.xlsx')

# ---------- Extract Target and Features ----------
target_svm = 'wants_extra_baggage'
features_svm = ['flight_duration', 'num_passengers', 'length_of_stay']

# ---------- Exclude Rows with Null Values ----------
df_svm_clean = df_customer[[target_svm] + features_svm].dropna()

X_svm = df_svm_clean[features_svm]
y_svm = df_svm_clean[target_svm]

# ---------- Train-Test Split (75-25) ----------
X_train_svm, X_test_svm, y_train_svm, y_test_svm = train_test_split(
    X_svm, y_svm, test_size=0.25, random_state=42
)

# ---------- Fit SVC Classifier ----------
svc_classifier = SVC(kernel='rbf', C=1.0, gamma='scale', random_state=42)
svc_classifier.fit(X_train_svm, y_train_svm)

# ---------- Calculate F1 Score ----------
y_pred_svm = svc_classifier.predict(X_test_svm)
svm_f1 = f1_score(y_test_svm, y_pred_svm)

print(f"\nSVM F1 Score: {svm_f1:.4f}")

# ---------- Confusion Matrix Heatmap ----------
cm = confusion_matrix(y_test_svm, y_pred_svm)

plt.figure(figsize=(8, 6))
sns.heatmap(cm, annot=True, fmt='d', cmap='Blues', cbar=True,
            xticklabels=['Predicted 0', 'Predicted 1'],
            yticklabels=['Actual 0', 'Actual 1'])
plt.xlabel('Predicted Class', fontsize=12)
plt.ylabel('Actual Class', fontsize=12)
plt.title('SVM Confusion Matrix - Extra Baggage Preference', fontsize=14, fontweight='bold')
plt.tight_layout()
plt.savefig('confusion_matrix.png', dpi=300, bbox_inches='tight')
plt.close()
print("Confusion matrix saved: confusion_matrix.png")

# ==========================================
# PROCEDURE 3: KNN Classification for Trip Type
# ==========================================

print("\n" + "=" * 60)
print("PROCEDURE 3: KNN Classification - Trip Type Categorization")
print("=" * 60)

# ---------- Load Customer Airways Data ----------
target_knn = 'trip_type'
features_knn = ['purchase_lead', 'flight_duration']

# ---------- Remove Missing Entries ----------
df_knn_clean = df_customer[[target_knn] + features_knn].dropna()

X_knn = df_knn_clean[features_knn]
y_knn = df_knn_clean[target_knn]

# ---------- Train-Test Split (70-30) ----------
X_train_knn, X_test_knn, y_train_knn, y_test_knn = train_test_split(
    X_knn, y_knn, test_size=0.3, random_state=42
)

# ---------- Train KNN Classifier ----------
knn_classifier = KNeighborsClassifier(n_neighbors=5, weights='distance')
knn_classifier.fit(X_train_knn, y_train_knn)

# ---------- Compute Test Accuracy ----------
y_pred_knn = knn_classifier.predict(X_test_knn)
knn_accuracy = accuracy_score(y_test_knn, y_pred_knn) * 100

print(f"\nKNN Accuracy: {knn_accuracy:.2f}%")

# ---------- ROC Curve for Multiclass (One-vs-Rest) ----------
classes = np.unique(y_knn)
y_test_bin = label_binarize(y_test_knn, classes=classes)
y_score = knn_classifier.predict_proba(X_test_knn)

plt.figure(figsize=(10, 7))
colors = ['blue', 'red', 'green']

for i, (cls, color) in enumerate(zip(classes, colors)):
    fpr, tpr, _ = roc_curve(y_test_bin[:, i], y_score[:, i])
    roc_auc = auc(fpr, tpr)
    plt.plot(fpr, tpr, color=color, lw=2,
             label=f'{cls} (AUC = {roc_auc:.2f})')

plt.plot([0, 1], [0, 1], 'k--', lw=1, label='Random Classifier')
plt.xlim([0.0, 1.0])
plt.ylim([0.0, 1.05])
plt.xlabel('False Positive Rate', fontsize=12)
plt.ylabel('True Positive Rate', fontsize=12)
plt.title('ROC Curve - Trip Type Classification (One-vs-Rest)', fontsize=14, fontweight='bold')
plt.legend(loc='lower right', fontsize=10)
plt.grid(alpha=0.3)
plt.tight_layout()
plt.savefig('roc_curve_multiclass.png', dpi=300, bbox_inches='tight')
plt.close()
print("ROC curve saved: roc_curve_multiclass.png")

# ==========================================
# PROCEDURE 4: Gaussian Mixture Modeling for Customer Segmentation
# ==========================================

print("\n" + "=" * 60)
print("PROCEDURE 4: Gaussian Mixture Modeling - Customer Segmentation")
print("=" * 60)

# ---------- Load and Merge Datasets ----------
df_customer_reload = pd.read_excel('customer_airways_data.xlsx')
df_booking_reload = pd.read_excel('filtered_customer_booking.xlsx')

# Extract Clustering Features from booking data (contains all needed features)
gmm_features = ['wants_extra_baggage', 'wants_preferred_seat',
                'wants_in_flight_meals', 'num_passengers']

df_gmm = df_booking_reload[gmm_features].dropna()

# ---------- Standardize Features ----------
scaler = StandardScaler()
X_scaled = scaler.fit_transform(df_gmm)

# ---------- Fit Gaussian Mixture Model ----------
gmm = GaussianMixture(n_components=3, covariance_type='full', random_state=42)
cluster_labels = gmm.fit_predict(X_scaled)

# ---------- Calculate Silhouette Score ----------
silhouette = silhouette_score(X_scaled, cluster_labels)

print(f"\nSilhouette Score: {silhouette:.4f}")

# ---------- Scatter Plot Matrix ----------
df_plot = df_gmm.copy()
df_plot['Cluster'] = cluster_labels

fig, axes = plt.subplots(4, 4, figsize=(16, 16))
feature_names = gmm_features

for i in range(4):
    for j in range(4):
        ax = axes[i, j]
        if i == j:
            # Diagonal: histogram
            for cluster in range(3):
                cluster_data = df_plot[df_plot['Cluster'] == cluster][feature_names[i]]
                ax.hist(cluster_data, alpha=0.5, label=f'Cluster {cluster}', bins=20)
            ax.set_ylabel('Frequency', fontsize=8)
            ax.set_xlabel(feature_names[i], fontsize=8)
        else:
            # Off-diagonal: scatter plot
            scatter = ax.scatter(df_plot[feature_names[j]], df_plot[feature_names[i]],
                               c=df_plot['Cluster'], cmap='viridis', alpha=0.6, s=5)
            ax.set_xlabel(feature_names[j], fontsize=8)
            ax.set_ylabel(feature_names[i], fontsize=8)
        ax.tick_params(labelsize=7)

plt.suptitle('Scatter Plot Matrix - Customer Segmentation by Cluster',
             fontsize=16, fontweight='bold', y=0.995)
plt.tight_layout()
plt.savefig('cluster_scatter_matrix.png', dpi=300, bbox_inches='tight')
plt.close()
print("Cluster scatter matrix saved: cluster_scatter_matrix.png")

# ==========================================
# PROCEDURE 5: PCA for Dimensionality Reduction
# ==========================================

print("\n" + "=" * 60)
print("PROCEDURE 5: PCA - Dimensionality Reduction")
print("=" * 60)

# ---------- Load Booking Data ----------
df_booking_pca = pd.read_excel('filtered_customer_booking.xlsx')

# ---------- Select Numerical Attributes ----------
pca_features = ['num_passengers', 'purchase_lead', 'length_of_stay', 'flight_hour',
                'flight_duration', 'wants_extra_baggage', 'wants_preferred_seat',
                'wants_in_flight_meals']

df_pca = df_booking_pca[pca_features].dropna()

# ---------- Standardize Features ----------
scaler_pca = StandardScaler()
X_pca_scaled = scaler_pca.fit_transform(df_pca)

# ---------- Apply PCA ----------
pca = PCA(n_components=5, random_state=42)
pca.fit(X_pca_scaled)

# ---------- Extract Explained Variance Ratio ----------
explained_variance = pca.explained_variance_ratio_
first_component_variance = explained_variance[0]

print(f"\nFirst Principal Component Explained Variance Ratio: {first_component_variance:.4f}")

# ---------- Scree Plot ----------
plt.figure(figsize=(10, 6))
plt.plot(range(1, 6), explained_variance * 100, marker='o', linestyle='-',
         color='darkblue', markersize=8, linewidth=2)
plt.xlabel('Principal Component Number', fontsize=12)
plt.ylabel('Explained Variance (%)', fontsize=12)
plt.title('Scree Plot - PCA Explained Variance', fontsize=14, fontweight='bold')
plt.xticks(range(1, 6))
plt.grid(alpha=0.3)
plt.tight_layout()
plt.savefig('scree_plot.png', dpi=300, bbox_inches='tight')
plt.close()
print("Scree plot saved: scree_plot.png")

# ==========================================
# PROCEDURE 6: Multiple Linear Regression for Flight Duration
# ==========================================

print("\n" + "=" * 60)
print("PROCEDURE 6: Multiple Linear Regression - Flight Duration Modeling")
print("=" * 60)

# ---------- Load Customer Airways Data ----------
df_customer_lr = pd.read_excel('customer_airways_data.xlsx')

# ---------- Define Dependent and Independent Variables ----------
dependent_var = 'flight_duration'
independent_vars = ['num_passengers', 'purchase_lead', 'length_of_stay']

# ---------- Exclude Missing Values ----------
df_lr = df_customer_lr[[dependent_var] + independent_vars].dropna()

X_lr = df_lr[independent_vars]
y_lr = df_lr[dependent_var]

# ---------- Train-Test Split (80-20) ----------
X_train_lr, X_test_lr, y_train_lr, y_test_lr = train_test_split(
    X_lr, y_lr, test_size=0.2, random_state=42
)

# ---------- Fit Linear Regression Model ----------
lr_model = LinearRegression()
lr_model.fit(X_train_lr, y_train_lr)

# ---------- Calculate RMSE ----------
y_pred_lr = lr_model.predict(X_test_lr)
rmse = np.sqrt(mean_squared_error(y_test_lr, y_pred_lr))

print(f"\nRoot Mean Squared Error (RMSE): {rmse:.3f}")

# ---------- Residual Plot ----------
residuals = y_test_lr - y_pred_lr

plt.figure(figsize=(10, 6))
plt.scatter(y_pred_lr, residuals, alpha=0.5, color='purple', s=20)
plt.axhline(y=0, color='red', linestyle='--', linewidth=2)
plt.xlabel('Predicted Flight Duration', fontsize=12)
plt.ylabel('Residuals (Actual - Predicted)', fontsize=12)
plt.title('Residual Plot - Flight Duration Prediction', fontsize=14, fontweight='bold')
plt.grid(alpha=0.3)
plt.tight_layout()
plt.savefig('residual_plot.png', dpi=300, bbox_inches='tight')
plt.close()
print("Residual plot saved: residual_plot.png")

# ==========================================
# PROCEDURE 7: One-Way ANOVA for Purchase Lead by Sales Channel
# ==========================================

print("\n" + "=" * 60)
print("PROCEDURE 7: One-Way ANOVA - Purchase Lead by Sales Channel")
print("=" * 60)

# ---------- Load Customer Airways Data ----------
df_customer_anova = pd.read_excel('customer_airways_data.xlsx')

# ---------- Extract Variables ----------
df_anova = df_customer_anova[['purchase_lead', 'sales_channel']].dropna()

# ---------- Group by Sales Channel ----------
groups = df_anova.groupby('sales_channel')['purchase_lead'].apply(list)

# ---------- Apply One-Way ANOVA ----------
f_statistic, p_value = f_oneway(*groups)

print(f"\nF-Statistic: {f_statistic:.2f}")

# ==========================================
# PROCEDURE 8: Standard Error of Mean for Flight Duration by Trip Type
# ==========================================

print("\n" + "=" * 60)
print("PROCEDURE 8: Standard Error of Mean - Flight Duration by Trip Type")
print("=" * 60)

# ---------- Load Booking Data ----------
df_booking_sem = pd.read_excel('filtered_customer_booking.xlsx')

# ---------- Extract Variables ----------
df_sem = df_booking_sem[['flight_duration', 'trip_type']].dropna()

# ---------- Calculate SEM for Each Trip Type ----------
sem_results = df_sem.groupby('trip_type')['flight_duration'].agg(
    lambda x: x.std() / np.sqrt(len(x))
)

max_sem_trip = sem_results.idxmax()
max_sem_value = sem_results.max()

print(f"\nTrip Type with Maximum SEM: {max_sem_trip}")
print(f"Maximum Standard Error of Mean: {max_sem_value:.4f}")

# ==========================================
# PROCEDURE 9: Percentile Rank for Flight Hour Value 12
# ==========================================

print("\n" + "=" * 60)
print("PROCEDURE 9: Percentile Rank - Flight Hour Value 12")
print("=" * 60)

# ---------- Load Customer Airways Data ----------
df_customer_percentile = pd.read_excel('customer_airways_data.xlsx')

# ---------- Calculate Percentile Rank ----------
flight_hours = df_customer_percentile['flight_hour'].dropna()
count_below = (flight_hours < 12).sum()
total_count = len(flight_hours)
percentile_rank = int((count_below / total_count) * 100)

print(f"\nPercentile Rank for Flight Hour 12: {percentile_rank}%")

# ==========================================
# PROCEDURE 10: Sentiment Correlation - Ratings vs Review Length
# ==========================================

print("\n" + "=" * 60)
print("PROCEDURE 10: Sentiment Correlation - Ratings vs Review Length")
print("=" * 60)

# ---------- Load Airline Review Data ----------
df_airline = pd.read_excel('airline_data.xlsx')

# ---------- Extract Rates and Calculate Review Length ----------
df_sentiment = df_airline[['rates', 'reviews']].dropna()
df_sentiment['review_length'] = df_sentiment['reviews'].astype(str).str.len()

# ---------- Compute Pearson Correlation ----------
correlation = df_sentiment[['rates', 'review_length']].corr().iloc[0, 1]

print(f"\nPearson Correlation (Rates vs Review Length): {correlation:.4f}")

# ==========================================
# COMPARATIVE ANALYSIS
# ==========================================

print("\n" + "=" * 60)
print("COMPARATIVE ANALYSIS: Decision Tree vs SVM")
print("=" * 60)

print(f"\nDecision Tree Accuracy: {dt_accuracy:.2f}%")
print(f"SVM F1 Score: {svm_f1:.4f}")

# Answer to comparison question
if dt_accuracy / 100 > svm_f1:
    comparison_answer = "Decision tree delivers superior booking prediction performance with higher accuracy compared to SVM's F1 score."
else:
    comparison_answer = "SVM delivers superior booking prediction performance with higher F1 score compared to decision tree's accuracy metric."

print(f"\n{comparison_answer}")

print("\n" + "=" * 60)
print("ANALYSIS COMPLETE")
print("=" * 60)
print("\nGenerated Output Files:")
print("  - feature_importance.png")
print("  - confusion_matrix.png")
print("  - roc_curve_multiclass.png")
print("  - cluster_scatter_matrix.png")
print("  - scree_plot.png")
print("  - residual_plot.png")
