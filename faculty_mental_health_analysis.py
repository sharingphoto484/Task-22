# ==========================================
# Faculty Mental Health & Achievement Analysis Script
# Requirements: pandas, numpy, matplotlib, scipy, scikit-learn, statsmodels, openpyxl
# Input files: bycountry.xlsx, survey_mental_health.xlsx, survey_mental_health_furthernonimized.xlsx
# Output files: svr_forecast_plot.png, mlp_confusion_matrix.png, birch_clusters_plot.png,
#               gmm_components_plot.png, ols_residuals_plot.png, sgd_loss_plot.png
# ==========================================

import pandas as pd
import numpy as np
import matplotlib.pyplot as plt
from scipy import stats
from sklearn.svm import SVR
from sklearn.neural_network import MLPClassifier
from sklearn.cluster import Birch
from sklearn.mixture import GaussianMixture
from sklearn.linear_model import SGDClassifier
from sklearn.model_selection import train_test_split
from sklearn.preprocessing import StandardScaler
from sklearn.metrics import r2_score, accuracy_score, f1_score, confusion_matrix, ConfusionMatrixDisplay
import statsmodels.api as sm
import warnings
warnings.filterwarnings('ignore')

# ---------- Operation 1: SVR Analysis (bycountry.xlsx) ----------
print("=" * 60)
print("OPERATION 1: Support Vector Regression Analysis")
print("=" * 60)

# Load data independently
df_svr = pd.read_excel('bycountry.xlsx')

# Remove entries where subjective.happyness is null or stress is null
df_svr_filtered = df_svr.dropna(subset=['subjective.happyness', 'stress'])

# Prepare features and target
X_svr = df_svr_filtered[['stress']].values
y_svr = df_svr_filtered['subjective.happyness'].values

# Split data: 75% training, 25% validation
X_train_svr, X_test_svr, y_train_svr, y_test_svr = train_test_split(
    X_svr, y_svr, test_size=0.25, random_state=42
)

# Configure and fit SVR
svr_model = SVR(kernel='rbf', C=1.0, epsilon=0.1, gamma='scale')
svr_model.fit(X_train_svr, y_train_svr)

# Predict and calculate R-squared
y_pred_svr = svr_model.predict(X_test_svr)
r2_svr = r2_score(y_test_svr, y_pred_svr)

print(f"R-squared: {r2_svr:.4f}")

# Create forecast assessment plot
plt.figure(figsize=(8, 6))
plt.scatter(y_test_svr, y_pred_svr, alpha=0.6, edgecolors='k', linewidths=0.5)
plt.xlabel('Actual Wellbeing Satisfaction Scores')
plt.ylabel('Estimated Wellbeing Satisfaction Scores')
plt.title('SVR Forecast Assessment: Actual vs Predicted Wellbeing')
plt.plot([y_test_svr.min(), y_test_svr.max()], [y_test_svr.min(), y_test_svr.max()], 'r--', lw=2)
plt.tight_layout()
plt.savefig('svr_forecast_plot.png', dpi=150)
plt.close()

# ---------- Operation 2: MLP Classification (survey_mental_health.xlsx) ----------
print("\n" + "=" * 60)
print("OPERATION 2: Multilayer Perceptron Classification")
print("=" * 60)

# Load data independently
df_mlp = pd.read_excel('survey_mental_health.xlsx')

# Generate permanence indicator
tenure_col = 'At what age did you got tenure? (type 0 if not yet)'
df_mlp['permanence'] = df_mlp[tenure_col].apply(
    lambda x: 1 if pd.notna(x) and x > 0 else (0 if pd.notna(x) and x <= 0 else np.nan)
)

# Filter entries where permanence is null or predictors are null
df_mlp_filtered = df_mlp.dropna(subset=['permanence', 'What is your h-index?', 'How stressed are you because of work?'])

# Prepare features and target
X_mlp = df_mlp_filtered[['What is your h-index?', 'How stressed are you because of work?']].values
y_mlp = df_mlp_filtered['permanence'].values.astype(int)

# Split data: 70% training, 30% holdout with stratified split
X_train_mlp, X_test_mlp, y_train_mlp, y_test_mlp = train_test_split(
    X_mlp, y_mlp, test_size=0.30, random_state=42, stratify=y_mlp
)

# Configure and fit MLP
mlp_model = MLPClassifier(hidden_layer_sizes=(50, 30), activation='relu', solver='adam', max_iter=500, random_state=42)
mlp_model.fit(X_train_mlp, y_train_mlp)

# Predict and calculate accuracy
y_pred_mlp = mlp_model.predict(X_test_mlp)
accuracy_mlp = accuracy_score(y_test_mlp, y_pred_mlp) * 100

print(f"Accuracy: {accuracy_mlp:.2f}%")

# Create confusion matrix heatmap
plt.figure(figsize=(8, 6))
cm = confusion_matrix(y_test_mlp, y_pred_mlp)
disp = ConfusionMatrixDisplay(confusion_matrix=cm, display_labels=['No Tenure', 'Tenured'])
disp.plot(cmap='Blues', values_format='d')
plt.xlabel('Estimated Permanence Categories')
plt.ylabel('True Permanence Categories')
plt.title('MLP Classification: Confusion Matrix')
plt.tight_layout()
plt.savefig('mlp_confusion_matrix.png', dpi=150)
plt.close()

# ---------- Operation 3: BIRCH Clustering (survey_mental_health_furthernonimized.xlsx) ----------
print("\n" + "=" * 60)
print("OPERATION 3: BIRCH Clustering Analysis")
print("=" * 60)

# Load data independently
df_birch = pd.read_excel('survey_mental_health_furthernonimized.xlsx')

# Remove entries where SubjectiveHappinessIndex or stress is null
df_birch_filtered = df_birch.dropna(subset=['SubjectiveHappinessIndex', 'How stressed are you because of work?'])

# Prepare features
X_birch = df_birch_filtered[['SubjectiveHappinessIndex', 'How stressed are you because of work?']].values

# Fit StandardScaler on remaining data
scaler_birch = StandardScaler()
X_birch_scaled = scaler_birch.fit_transform(X_birch)

# Run BIRCH
birch_model = Birch(n_clusters=4, threshold=0.5, branching_factor=50)
cluster_labels = birch_model.fit_predict(X_birch_scaled)

# Calculate Calinski-Harabasz score
from sklearn.metrics import calinski_harabasz_score
ch_score = calinski_harabasz_score(X_birch_scaled, cluster_labels)

print(f"Calinski-Harabasz: {ch_score:.2f}")

# Create cluster visualization plot
plt.figure(figsize=(10, 6))
scatter = plt.scatter(X_birch[:, 0], X_birch[:, 1], c=cluster_labels, cmap='viridis', alpha=0.6, edgecolors='k', linewidths=0.3)
plt.colorbar(scatter, label='Segment Identity')
plt.xlabel('SubjectiveHappinessIndex')
plt.ylabel('Workplace Strain Intensity')
plt.title('BIRCH Clustering: Mental Health Segmentation')
plt.tight_layout()
plt.savefig('birch_clusters_plot.png', dpi=150)
plt.close()

# ---------- Operation 4: Gaussian Mixture Model (bycountry.xlsx) ----------
print("\n" + "=" * 60)
print("OPERATION 4: Gaussian Mixture Probabilistic Modeling")
print("=" * 60)

# Load data independently
df_gmm = pd.read_excel('bycountry.xlsx')

# Filter entries where hindex or subjective.happyness is null
df_gmm_filtered = df_gmm.dropna(subset=['hindex', 'subjective.happyness'])

# Prepare features
X_gmm = df_gmm_filtered[['hindex', 'subjective.happyness']].values

# Run Gaussian Mixture
gmm_model = GaussianMixture(n_components=3, covariance_type='full', random_state=42)
gmm_model.fit(X_gmm)
gmm_labels = gmm_model.predict(X_gmm)

# Calculate BIC
bic_score = gmm_model.bic(X_gmm)

print(f"BIC: {bic_score:.2f}")

# Create component distribution plot
plt.figure(figsize=(10, 6))
scatter = plt.scatter(X_gmm[:, 0], X_gmm[:, 1], c=gmm_labels, cmap='plasma', alpha=0.6, edgecolors='k', linewidths=0.3)
plt.colorbar(scatter, label='Component Membership')
plt.xlabel('H-index Scores')
plt.ylabel('Wellbeing Satisfaction Scores')
plt.title('Gaussian Mixture: Faculty Subpopulation Distribution')
plt.tight_layout()
plt.savefig('gmm_components_plot.png', dpi=150)
plt.close()

# ---------- Operation 5: OLS Regression (survey_mental_health.xlsx) ----------
print("\n" + "=" * 60)
print("OPERATION 5: Ordinary Least Squares Regression")
print("=" * 60)

# Load data independently
df_ols = pd.read_excel('survey_mental_health.xlsx')

# Filter entries where any required variable is null
df_ols_filtered = df_ols.dropna(subset=['SubjectiveHappinessIndex', 'How stressed are you because of work?', 'What is your h-index?'])

# Prepare variables
y_ols = df_ols_filtered['SubjectiveHappinessIndex'].values
X_ols = df_ols_filtered[['How stressed are you because of work?', 'What is your h-index?']].values

# Add constant for OLS
X_ols_const = sm.add_constant(X_ols)

# Run OLS
ols_model = sm.OLS(y_ols, X_ols_const).fit()

# Get adjusted R-squared
adj_r2_ols = ols_model.rsquared_adj

print(f"Adjusted R-squared: {adj_r2_ols:.4f}")

# Create residual diagnostic plot
fitted_values = ols_model.fittedvalues
residuals = ols_model.resid

plt.figure(figsize=(8, 6))
plt.scatter(fitted_values, residuals, alpha=0.6, edgecolors='k', linewidths=0.5)
plt.axhline(y=0, color='r', linestyle='--', lw=2)
plt.xlabel('Fitted Outcome Values')
plt.ylabel('Error Terms (Residuals)')
plt.title('OLS Diagnostic: Residuals vs Fitted Values')
plt.tight_layout()
plt.savefig('ols_residuals_plot.png', dpi=150)
plt.close()

# ---------- Operation 6: SGD Classification (survey_mental_health_furthernonimized.xlsx) ----------
print("\n" + "=" * 60)
print("OPERATION 6: Stochastic Gradient Descent Classification")
print("=" * 60)

# Load data independently
df_sgd = pd.read_excel('survey_mental_health_furthernonimized.xlsx')

# Generate elevated wellbeing indicator
df_sgd['elevated_wellbeing'] = df_sgd['SubjectiveHappinessIndex'].apply(
    lambda x: 1 if pd.notna(x) and x > 5.0 else (0 if pd.notna(x) and x <= 5.0 else np.nan)
)

# Filter entries where any required variable is null
df_sgd_filtered = df_sgd.dropna(subset=['elevated_wellbeing', 'How stressed are you because of work?', 'What is your h-index?'])

# Prepare features and target
X_sgd = df_sgd_filtered[['How stressed are you because of work?', 'What is your h-index?']].values
y_sgd = df_sgd_filtered['elevated_wellbeing'].values.astype(int)

# Split data: 80% training, 20% testing with stratified split
X_train_sgd, X_test_sgd, y_train_sgd, y_test_sgd = train_test_split(
    X_sgd, y_sgd, test_size=0.20, random_state=42, stratify=y_sgd
)

# Standardize features for SGD
scaler_sgd = StandardScaler()
X_train_sgd_scaled = scaler_sgd.fit_transform(X_train_sgd)
X_test_sgd_scaled = scaler_sgd.transform(X_test_sgd)

# Configure SGD with partial_fit to track loss
sgd_model = SGDClassifier(loss='hinge', penalty='l2', max_iter=1000, random_state=42)
sgd_model.fit(X_train_sgd_scaled, y_train_sgd)

# Predict and calculate F1 score
y_pred_sgd = sgd_model.predict(X_test_sgd_scaled)
f1_sgd = f1_score(y_test_sgd, y_pred_sgd)

print(f"F1 Score: {f1_sgd:.4f}")

# Create optimization path plot by training incrementally
np.random.seed(42)
sgd_loss_tracker = SGDClassifier(loss='hinge', penalty='l2', max_iter=1, warm_start=True, random_state=42)
losses = []
iterations = list(range(1, 101))

# Convert labels from {0, 1} to {-1, 1} for correct hinge loss calculation
y_train_sgd_transformed = np.where(y_train_sgd == 0, -1, 1)

for i in iterations:
    sgd_loss_tracker.max_iter = i
    sgd_loss_tracker.fit(X_train_sgd_scaled, y_train_sgd)
    # Calculate hinge loss using correct formula: L(y, f(x)) = max(0, 1 - y * f(x))
    decision_values = sgd_loss_tracker.decision_function(X_train_sgd_scaled)
    hinge_loss = np.mean(np.maximum(0, 1 - y_train_sgd_transformed * decision_values))
    losses.append(hinge_loss)

plt.figure(figsize=(10, 6))
plt.plot(iterations, losses, 'b-', lw=2)
plt.xlabel('Iteration Number')
plt.ylabel('Iteration Loss')
plt.title('SGD Optimization Path: Training Loss over Iterations')
plt.grid(True, alpha=0.3)
plt.tight_layout()
plt.savefig('sgd_loss_plot.png', dpi=150)
plt.close()

# ---------- Operation 7: Workplace Strain-Wellbeing Association Index (bycountry.xlsx) ----------
print("\n" + "=" * 60)
print("OPERATION 7: Workplace Strain-Wellbeing Association Index")
print("=" * 60)

# Load data independently
df_assoc = pd.read_excel('bycountry.xlsx')

# Remove entries where stress or subjective.happyness is null
df_assoc_filtered = df_assoc.dropna(subset=['stress', 'subjective.happyness'])

# Calculate Pearson correlation
stress_vals = df_assoc_filtered['stress'].values
happiness_vals = df_assoc_filtered['subjective.happyness'].values

r, p_value = stats.pearsonr(stress_vals, happiness_vals)

# Calculate Association Index = |r| * (1 - p_value)
association_index = abs(r) * (1 - p_value)

print(f"Association Index: {association_index:.4f}")

# ---------- Operation 8: Faculty Permanence Achievement Rate (survey_mental_health.xlsx) ----------
print("\n" + "=" * 60)
print("OPERATION 8: Faculty Permanence Achievement Rate")
print("=" * 60)

# Load data independently
df_achieve = pd.read_excel('survey_mental_health.xlsx')

# Select entries where tenure age > 0
tenure_col = 'At what age did you got tenure? (type 0 if not yet)'
df_tenured = df_achieve[df_achieve[tenure_col] > 0].copy()

# Calculate mean h-index for tenured faculty (using non-null values)
mean_hindex = df_tenured['What is your h-index?'].dropna().mean()

# Calculate mean tenure age for tenured faculty (using non-null values)
mean_tenure_age = df_tenured[tenure_col].dropna().mean()

# Calculate Achievement Rate
achievement_rate = mean_hindex / mean_tenure_age

print(f"Achievement Rate: {achievement_rate:.3f}")

# ---------- Operation 9: Assessment Instrument Agreement Measure ----------
print("\n" + "=" * 60)
print("OPERATION 9: Assessment Instrument Agreement Measure")
print("=" * 60)

# Load survey_mental_health.xlsx independently
df_survey = pd.read_excel('survey_mental_health.xlsx')
mean_survey_happiness = df_survey['SubjectiveHappinessIndex'].dropna().mean()

# Load bycountry.xlsx independently
df_country = pd.read_excel('bycountry.xlsx')
mean_country_happiness = df_country['subjective.happyness'].dropna().mean()

# Calculate Agreement Measure as absolute difference
agreement_measure = abs(mean_survey_happiness - mean_country_happiness)

print(f"Agreement Measure: {agreement_measure:.4f}")

# ---------- Open-Ended Analysis Response ----------
print("\n" + "=" * 60)
print("ANALYTICAL INSIGHT: SVR vs OLS Performance Contrast")
print("=" * 60)

# Answer to why SVR and OLS produce contrasting performance:
# SVR with RBF kernel captures non-linear relationships between stress and wellbeing through
# kernel-based transformation into higher-dimensional feature space, while OLS assumes strictly
# linear relationships; thus when stress-wellbeing associations contain non-linear patterns,
# SVR outperforms, but when relationships are genuinely linear with multiple predictors
# explaining variance additively, OLS may show stronger adjusted R-squared due to its
# parsimony and interpretability without overfitting risk from kernel complexity.

print("SVR uses RBF kernel to capture non-linear stress-wellbeing relationships through")
print("implicit high-dimensional feature mapping, while OLS assumes linear additive effects")
print("of predictors; performance contrast arises because SVR can model complex curvature")
print("patterns but may overfit with limited features, whereas OLS with multiple linear")
print("predictors leverages additive variance explanation without kernel complexity overhead.")

print("\n" + "=" * 60)
print("KEY OUTPUT SUMMARY")
print("=" * 60)
print(f"1. SVR R-squared: {r2_svr:.4f}")
print(f"2. MLP Accuracy: {accuracy_mlp:.2f}%")
print(f"3. BIRCH Calinski-Harabasz: {ch_score:.2f}")
print(f"4. GMM BIC: {bic_score:.2f}")
print(f"5. OLS Adjusted R-squared: {adj_r2_ols:.4f}")
print(f"6. SGD F1 Score: {f1_sgd:.4f}")
print(f"7. Association Index: {association_index:.4f}")
print(f"8. Achievement Rate: {achievement_rate:.3f}")
print(f"9. Agreement Measure: {agreement_measure:.4f}")
print("=" * 60)

print("\nPlots saved: svr_forecast_plot.png, mlp_confusion_matrix.png, birch_clusters_plot.png,")
print("             gmm_components_plot.png, ols_residuals_plot.png, sgd_loss_plot.png")
