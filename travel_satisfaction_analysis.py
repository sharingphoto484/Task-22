# ==========================================
# Travel Satisfaction Statistical Analysis
# ==========================================
# Requirements: pandas, numpy, matplotlib, scipy, scikit-learn, statsmodels
# Input files: Final_Updated_Expanded_Reviews.csv, Expanded_Destinations.csv,
#              Final_Updated_Expanded_Users.csv (in same directory)
# Output files: integrated_travel_dataset.csv, roc_curve.png,
#               analysis_summary.json
# ==========================================

import pandas as pd
import numpy as np
import matplotlib.pyplot as plt
from scipy import stats
from sklearn.linear_model import LogisticRegression
from sklearn.metrics import roc_curve, roc_auc_score, precision_score, recall_score
from sklearn.preprocessing import StandardScaler
from statsmodels.stats.multicomp import pairwise_tukeyhsd
import json
import warnings
warnings.filterwarnings('ignore')

# ---------- Load CSVs Robustly ----------
print("Loading datasets...")
reviews = pd.read_csv('Final_Updated_Expanded_Reviews.csv')
destinations = pd.read_csv('Expanded_Destinations.csv')
users = pd.read_csv('Final_Updated_Expanded_Users.csv')

print(f"Reviews loaded: {len(reviews)} rows")
print(f"Destinations loaded: {len(destinations)} rows")
print(f"Users loaded: {len(users)} rows")

# ---------- Merge Datasets ----------
print("\nMerging datasets...")
# First merge reviews with destinations
merged = reviews.merge(destinations, on='DestinationID', how='inner')
# Then merge with users
merged = merged.merge(users, on='UserID', how='inner')

print(f"Merged dataset: {len(merged)} rows")

# ---------- Data Validation and Cleaning ----------
print("\nCleaning and validating data...")
# Convert Rating and Popularity to numeric, coerce errors to NaN
merged['Rating'] = pd.to_numeric(merged['Rating'], errors='coerce')
merged['Popularity'] = pd.to_numeric(merged['Popularity'], errors='coerce')
merged['NumberOfAdults'] = pd.to_numeric(merged['NumberOfAdults'], errors='coerce')
merged['NumberOfChildren'] = pd.to_numeric(merged['NumberOfChildren'], errors='coerce')

# Filter for valid numeric values
original_count = len(merged)
merged = merged.dropna(subset=['Rating', 'Popularity', 'NumberOfAdults', 'NumberOfChildren', 'Gender', 'Type'])
print(f"Valid records after cleaning: {len(merged)} (removed {original_count - len(merged)} invalid records)")

# Save integrated dataset
merged.to_csv('integrated_travel_dataset.csv', index=False)
print("Saved integrated_travel_dataset.csv")

# ---------- Create Binary Classification Target ----------
print("\nCreating binary classification target...")
# Define positive as rating >= 3, negative as rating < 3
# Based on typical 1-5 rating scale where 3+ is positive
merged['IsPositive'] = (merged['Rating'] >= 3).astype(int)
positive_count = merged['IsPositive'].sum()
negative_count = len(merged) - positive_count
print(f"Positive reviews: {positive_count} ({positive_count/len(merged)*100:.1f}%)")
print(f"Negative reviews: {negative_count} ({negative_count/len(merged)*100:.1f}%)")

# ---------- Logistic Regression Model ----------
print("\n" + "="*60)
print("LOGISTIC REGRESSION ANALYSIS")
print("="*60)

# Prepare features
X = merged[['Popularity', 'NumberOfAdults', 'NumberOfChildren']].copy()
y = merged['IsPositive'].copy()

# Standardize features for coefficient comparison
scaler = StandardScaler()
X_scaled = scaler.fit_transform(X)
X_scaled_df = pd.DataFrame(X_scaled, columns=X.columns)

# Fit logistic regression
lr_model = LogisticRegression(random_state=42, max_iter=1000)
lr_model.fit(X_scaled, y)

# Get predictions and probabilities
y_pred_proba = lr_model.predict_proba(X_scaled)[:, 1]
y_pred = lr_model.predict(X_scaled)

# ---------- ROC Curve Analysis ----------
print("\nROC Curve Analysis:")
fpr, tpr, thresholds = roc_curve(y, y_pred_proba)
auc = roc_auc_score(y, y_pred_proba)

print(f"AUC: {auc:.3f}")

# Find optimal threshold using Youden's J statistic
j_scores = tpr - fpr
optimal_idx = np.argmax(j_scores)
optimal_threshold = thresholds[optimal_idx]
optimal_tpr = tpr[optimal_idx]
optimal_fpr = fpr[optimal_idx]

print(f"Optimal Threshold: {optimal_threshold:.3f}")
print(f"True Positive Rate at optimal threshold: {optimal_tpr:.3f}")
print(f"False Positive Rate at optimal threshold: {optimal_fpr:.3f}")

# Calculate precision and recall at optimal threshold
y_pred_optimal = (y_pred_proba >= optimal_threshold).astype(int)
precision_optimal = precision_score(y, y_pred_optimal)
recall_optimal = recall_score(y, y_pred_optimal)

print(f"Precision at optimal threshold: {precision_optimal:.3f}")
print(f"Recall at optimal threshold: {recall_optimal:.3f}")

# ---------- Standardized Coefficients Analysis ----------
print("\nStandardized Coefficients:")
coefficients = pd.DataFrame({
    'Variable': X.columns,
    'Standardized_Coefficient': lr_model.coef_[0]
})
coefficients['Abs_Coefficient'] = np.abs(coefficients['Standardized_Coefficient'])
coefficients = coefficients.sort_values('Abs_Coefficient', ascending=False)

for idx, row in coefficients.iterrows():
    print(f"  {row['Variable']}: {row['Standardized_Coefficient']:.4f}")

most_influential = coefficients.iloc[0]
print(f"\nMost Influential Predictor: {most_influential['Variable']}")
print(f"Standardized Coefficient: {most_influential['Standardized_Coefficient']:.4f}")

# ---------- One-Way ANOVA Analysis ----------
print("\n" + "="*60)
print("ONE-WAY ANOVA: Rating by Destination Type")
print("="*60)

# Group ratings by destination type
groups = [group['Rating'].values for name, group in merged.groupby('Type')]
group_names = list(merged.groupby('Type').groups.keys())

# Perform ANOVA
f_statistic, p_value_anova = stats.f_oneway(*groups)

print(f"F-statistic: {f_statistic:.3f}")
print(f"P-value: {p_value_anova:.4f}")

alpha = 0.05
if p_value_anova < alpha:
    print(f"Result: SIGNIFICANT (p < {alpha})")

    # ---------- Tukey Post-Hoc Test ----------
    print("\nTukey Post-Hoc Analysis:")
    tukey_result = pairwise_tukeyhsd(merged['Rating'], merged['Type'], alpha=alpha)
    print(tukey_result)

    # Calculate mean ratings by type
    mean_ratings = merged.groupby('Type')['Rating'].mean().sort_values()
    print("\nMean Ratings by Destination Type:")
    for dest_type, mean_rating in mean_ratings.items():
        print(f"  {dest_type}: {mean_rating:.3f}")

    # Mean difference between highest and lowest
    highest_type = mean_ratings.idxmax()
    lowest_type = mean_ratings.idxmin()
    mean_difference = mean_ratings.max() - mean_ratings.min()

    print(f"\nHighest rated type: {highest_type} ({mean_ratings.max():.3f})")
    print(f"Lowest rated type: {lowest_type} ({mean_ratings.min():.3f})")
    print(f"Mean difference: {mean_difference:.3f}")
else:
    print(f"Result: NOT SIGNIFICANT (p >= {alpha})")
    mean_difference = 0.0

# ---------- Causal Analysis: Popularity → Rating (controlling for Gender) ----------
print("\n" + "="*60)
print("CAUSAL ANALYSIS: Popularity → Rating (controlling for Gender)")
print("="*60)

# Encode Gender as numeric
merged['Gender_Numeric'] = merged['Gender'].map({'Male': 1, 'Female': 0})

# Multiple regression: Rating ~ Popularity + Gender
X_causal = merged[['Popularity', 'Gender_Numeric']].copy()
y_rating = merged['Rating'].copy()

# Using OLS regression from scipy
from scipy.stats import linregress

# Simple approach: partial regression coefficient
# Full model: Rating = b0 + b1*Popularity + b2*Gender
X_causal_with_const = np.column_stack([np.ones(len(X_causal)), X_causal])
from numpy.linalg import lstsq

# Solve using least squares
coeffs, residuals, rank, s = lstsq(X_causal_with_const, y_rating, rcond=None)
causal_coefficient = coeffs[1]  # Coefficient for Popularity

# Calculate p-value using statsmodels for proper inference
import statsmodels.api as sm
X_causal_sm = sm.add_constant(X_causal)
causal_model = sm.OLS(y_rating, X_causal_sm).fit()
causal_pvalue = causal_model.pvalues['Popularity']

print(f"Causal Coefficient (Popularity → Rating): {causal_coefficient:.4f}")
print(f"P-value: {causal_pvalue:.4f}")
print(f"\nInterpretation: Controlling for gender, a 1-unit increase in Popularity")
print(f"is associated with a {causal_coefficient:.4f} change in Rating")

# ---------- Gender-Based T-Test ----------
print("\n" + "="*60)
print("TWO-SAMPLE T-TEST: Gender Differences in Rating")
print("="*60)

male_ratings = merged[merged['Gender'] == 'Male']['Rating']
female_ratings = merged[merged['Gender'] == 'Female']['Rating']

t_statistic, p_value_ttest = stats.ttest_ind(male_ratings, female_ratings)

print(f"Male ratings: n={len(male_ratings)}, mean={male_ratings.mean():.3f}, std={male_ratings.std():.3f}")
print(f"Female ratings: n={len(female_ratings)}, mean={female_ratings.mean():.3f}, std={female_ratings.std():.3f}")
print(f"\nT-statistic: {t_statistic:.3f}")
print(f"P-value: {p_value_ttest:.4f}")

if p_value_ttest < alpha:
    print(f"Result: SIGNIFICANT gender difference (p < {alpha})")
else:
    print(f"Result: NO SIGNIFICANT gender difference (p >= {alpha})")

# ---------- ROC Curve Visualization ----------
print("\n" + "="*60)
print("GENERATING ROC CURVE VISUALIZATION")
print("="*60)

plt.figure(figsize=(10, 8))
plt.plot(fpr, tpr, color='blue', lw=2, label=f'ROC Curve (AUC = {auc:.3f})')
plt.plot([0, 1], [0, 1], color='gray', lw=1, linestyle='--', label='Random Classifier')

# Mark optimal threshold point
plt.scatter(optimal_fpr, optimal_tpr, color='red', s=100, zorder=5,
            label=f'Optimal Threshold ({optimal_threshold:.3f})\nTPR={optimal_tpr:.3f}, FPR={optimal_fpr:.3f}')

plt.xlabel('False Positive Rate', fontsize=12)
plt.ylabel('True Positive Rate', fontsize=12)
plt.title('Logistic Regression ROC Curve\nTravel Satisfaction Classification', fontsize=14, fontweight='bold')
plt.legend(loc='lower right', fontsize=10)
plt.grid(alpha=0.3)
plt.tight_layout()
plt.savefig('roc_curve.png', dpi=300, bbox_inches='tight')
print("Saved roc_curve.png")
plt.close()

# ---------- Summary JSON Output ----------
print("\n" + "="*60)
print("GENERATING ANALYSIS SUMMARY")
print("="*60)

summary = {
    "logistic_regression": {
        "auc": round(auc, 3),
        "optimal_threshold": round(optimal_threshold, 3),
        "precision_at_optimal": round(precision_optimal, 3),
        "recall_at_optimal": round(recall_optimal, 3),
        "tpr_at_optimal": round(optimal_tpr, 3),
        "fpr_at_optimal": round(optimal_fpr, 3),
        "most_influential_predictor": most_influential['Variable'],
        "most_influential_coefficient": round(most_influential['Standardized_Coefficient'], 4)
    },
    "anova": {
        "f_statistic": round(f_statistic, 3),
        "p_value": round(p_value_anova, 4),
        "significant": "Yes" if p_value_anova < alpha else "No",
        "mean_difference_highest_lowest": round(mean_difference, 3)
    },
    "causal_analysis": {
        "causal_coefficient": round(causal_coefficient, 4),
        "p_value": round(causal_pvalue, 4)
    },
    "gender_ttest": {
        "t_statistic": round(t_statistic, 3),
        "p_value": round(p_value_ttest, 4),
        "male_mean": round(male_ratings.mean(), 3),
        "female_mean": round(female_ratings.mean(), 3)
    }
}

with open('analysis_summary.json', 'w') as f:
    json.dump(summary, f, indent=2)

print("Saved analysis_summary.json")

# ---------- Final Report ----------
print("\n" + "="*60)
print("FINAL ANALYSIS REPORT")
print("="*60)

print("\n1. LOGISTIC REGRESSION CLASSIFICATION:")
print(f"   - AUC: {summary['logistic_regression']['auc']}")
print(f"   - Precision at optimal threshold: {summary['logistic_regression']['precision_at_optimal']}")
print(f"   - Recall at optimal threshold: {summary['logistic_regression']['recall_at_optimal']}")
print(f"   - TPR at optimal threshold: {summary['logistic_regression']['tpr_at_optimal']}")
print(f"   - FPR at optimal threshold: {summary['logistic_regression']['fpr_at_optimal']}")

print("\n2. MOST INFLUENTIAL PREDICTOR:")
print(f"   - Variable: {summary['logistic_regression']['most_influential_predictor']}")
print(f"   - Standardized Coefficient: {summary['logistic_regression']['most_influential_coefficient']}")

print("\n3. ANOVA RESULTS:")
print(f"   - F-statistic: {summary['anova']['f_statistic']}")
print(f"   - P-value: {summary['anova']['p_value']}")
print(f"   - Mean difference (highest vs lowest): {summary['anova']['mean_difference_highest_lowest']}")

print("\n4. CAUSAL ANALYSIS:")
print(f"   - Causal Coefficient (Popularity → Rating): {summary['causal_analysis']['causal_coefficient']}")
print(f"   - P-value: {summary['causal_analysis']['p_value']}")

print("\n5. GENDER-BASED T-TEST:")
print(f"   - T-statistic: {summary['gender_ttest']['t_statistic']}")
print(f"   - P-value: {summary['gender_ttest']['p_value']}")

print("\n" + "="*60)
print("ANALYSIS COMPLETE")
print("="*60)
print("\nOutput files generated:")
print("  - integrated_travel_dataset.csv")
print("  - roc_curve.png")
print("  - analysis_summary.json")
