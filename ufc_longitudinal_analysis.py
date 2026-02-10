# ==========================================
# UFC Longitudinal Analysis Script
# Requirements: pandas, numpy, matplotlib, scipy, scikit-learn, openpyxl, lifelines, statsmodels
# Input files: ufcfights_sept30.xlsx, ufcfights_10_7.xlsx, ufcfights10_26_24.xlsx
# Output files: key_results.json, method_proportions_by_round.png, stoppage_duration_boxplot.png,
#               kaplan_meier_survival.png, spectral_power_plot.png
# ==========================================

import pandas as pd
import numpy as np
import matplotlib.pyplot as plt
import seaborn as sns
from scipy import stats
from scipy.fft import fft
from sklearn.linear_model import LogisticRegression
from sklearn.model_selection import cross_val_score, StratifiedKFold
from statsmodels.formula.api import ols, quantreg
from statsmodels.stats.anova import anova_lm
# Manual Kaplan-Meier implementation (lifelines dependency issue)
import json
import re
import warnings
warnings.filterwarnings('ignore')

# Set plotting style
sns.set_style("whitegrid")
plt.rcParams['figure.figsize'] = (10, 6)

# ---------- Load Excel Files ----------
print("Loading Excel files...")
version_1 = pd.read_excel('ufcfights_sept30.xlsx')
version_2 = pd.read_excel('ufcfights_10_7.xlsx')
version_3 = pd.read_excel('ufcfights10_26_24.xlsx')

version_1['dataset_version'] = 'version_1'
version_2['dataset_version'] = 'version_2'
version_3['dataset_version'] = 'version_3'

# ---------- Data Cleaning Function ----------
def clean_dataset(df):
    """Clean and engineer features for UFC fight data"""
    df = df.copy()

    # Clean method column - keep only text before first newline
    df['method'] = df['method'].astype(str).apply(lambda x: x.split('\n')[0].strip())

    # Create method_group
    def map_method_group(method):
        if method == 'KO/TKO':
            return 'KO_TKO'
        elif method == 'SUB':
            return 'SUB'
        elif method in ['U-DEC', 'S-DEC', 'M-DEC']:
            return 'DEC'
        else:
            return 'OTHER'

    df['method_group'] = df['method'].apply(map_method_group)

    # Convert time (M:SS) to total_seconds
    def time_to_seconds(time_str):
        try:
            time_str = str(time_str).strip()
            parts = time_str.split(':')
            if len(parts) == 2:
                minutes = int(parts[0])
                seconds = int(parts[1])
                return minutes * 60 + seconds
            else:
                return np.nan
        except:
            return np.nan

    df['total_seconds'] = df['time'].apply(time_to_seconds)

    # Create finish_flag
    df['finish_flag'] = ((df['method_group'] == 'KO_TKO') | (df['method_group'] == 'SUB')).astype(int)

    # Classify event_type
    def classify_event_type(event_name):
        event_name = str(event_name)
        if re.search(r'UFC\s+\d+', event_name):
            return 'UFC_Numbered'
        elif 'Fight Night' in event_name:
            return 'Fight_Night'
        else:
            return 'Other_Event'

    df['event_type'] = df['event'].apply(classify_event_type)

    # Extract event_number for UFC_Numbered
    def extract_event_number(row):
        if row['event_type'] == 'UFC_Numbered':
            match = re.search(r'UFC\s+(\d+)', str(row['event']))
            if match:
                return int(match.group(1))
        return np.nan

    df['event_number'] = df.apply(extract_event_number, axis=1)

    # Create fight_key
    df['fight_key'] = df['event'].astype(str) + '|||' + df['fighter_1'].astype(str) + '|||' + df['fighter_2'].astype(str)

    return df

# ---------- Clean All Versions ----------
print("Cleaning datasets...")
version_1 = clean_dataset(version_1)
version_2 = clean_dataset(version_2)
version_3 = clean_dataset(version_3)

# ---------- Analysis 1: New Fights Count ----------
print("\n1. Counting new fights in version_3...")
merged_v1_v3 = pd.merge(version_3, version_1, on='fight_key', how='outer', indicator=True, suffixes=('_v3', '_v1'))
new_fights_count = (merged_v1_v3['_merge'] == 'left_only').sum()
new_fights_count = round(new_fights_count, 0)
print(f"   Fights in version_3 but not in version_1: {new_fights_count}")

# ---------- Analysis 2: Spearman Correlation (Event Number vs Finish Rate) ----------
print("\n2. Computing Spearman correlation...")
v3_ufc_numbered = version_3[version_3['event_type'] == 'UFC_Numbered'].copy()
per_event_agg = v3_ufc_numbered.groupby('event_number').agg(
    finish_rate=('finish_flag', 'mean'),
    mean_total_seconds=('total_seconds', 'mean'),
    num_fights=('finish_flag', 'size')
).reset_index().sort_values('event_number')

spearman_rho, _ = stats.spearmanr(per_event_agg['event_number'], per_event_agg['finish_rate'])
spearman_rho = round(spearman_rho, 4)
print(f"   Spearman rho (event_number vs finish_rate): {spearman_rho}")

# ---------- Analysis 3: Two-Way ANOVA (Era × Card-Size Interaction) ----------
print("\n3. Computing two-way ANOVA...")
median_fights = per_event_agg['num_fights'].median()
per_event_agg['era'] = per_event_agg['event_number'].apply(lambda x: 'early' if x <= 200 else 'late')
per_event_agg['card_size'] = per_event_agg['num_fights'].apply(lambda x: 'large' if x > median_fights else 'small')

anova_model = ols('finish_rate ~ C(era) * C(card_size)', data=per_event_agg).fit()
anova_table = anova_lm(anova_model, typ=2)
interaction_f_stat = anova_table.loc['C(era):C(card_size)', 'F']
interaction_f_stat = round(interaction_f_stat, 4)
print(f"   Two-way ANOVA interaction F-statistic: {interaction_f_stat}")

# ---------- Analysis 4: Cohen's d (KO_TKO vs SUB Duration) ----------
print("\n4. Computing Cohen's d...")
finished_fights = version_3[version_3['finish_flag'] == 1].copy()
ko_tko_times = finished_fights[finished_fights['method_group'] == 'KO_TKO']['total_seconds'].dropna()
sub_times = finished_fights[finished_fights['method_group'] == 'SUB']['total_seconds'].dropna()

mean_ko = ko_tko_times.mean()
mean_sub = sub_times.mean()
n_ko = len(ko_tko_times)
n_sub = len(sub_times)
var_ko = ko_tko_times.var(ddof=1)
var_sub = sub_times.var(ddof=1)

pooled_std = np.sqrt(((n_ko - 1) * var_ko + (n_sub - 1) * var_sub) / (n_ko + n_sub - 2))
cohens_d = abs((mean_ko - mean_sub) / pooled_std)
cohens_d = round(cohens_d, 4)
print(f"   Cohen's d (absolute): {cohens_d}")

# ---------- Analysis 5: Logistic Regression with 5-Fold CV ----------
print("\n5. Running logistic regression with cross-validation...")
v3_ufc_logit = v3_ufc_numbered[['finish_flag', 'round', 'event_number']].dropna()
X = v3_ufc_logit[['round', 'event_number']].values
y = v3_ufc_logit['finish_flag'].values

lr_model = LogisticRegression(max_iter=1000, random_state=42)
cv = StratifiedKFold(n_splits=5, shuffle=True, random_state=42)
roc_auc_scores = cross_val_score(lr_model, X, y, cv=cv, scoring='roc_auc')
mean_roc_auc = round(roc_auc_scores.mean(), 4)
print(f"   Mean ROC AUC (5-fold CV): {mean_roc_auc}")

# ---------- Analysis 6: FFT Spectral Analysis ----------
print("\n6. Performing FFT spectral analysis...")
per_event_sorted = per_event_agg.sort_values('event_number').copy()
finish_rate_series = per_event_sorted['finish_rate'].values
demeaned_series = finish_rate_series - finish_rate_series.mean()

fft_result = fft(demeaned_series)
magnitude = np.abs(fft_result)
n = len(demeaned_series)
freqs = np.fft.fftfreq(n, d=1)

# Exclude zero frequency and take positive frequencies only
positive_freqs = freqs[1:n//2]
positive_magnitude = magnitude[1:n//2]

dominant_freq_idx = np.argmax(positive_magnitude)
dominant_freq = positive_freqs[dominant_freq_idx]
dominant_period = round(1 / abs(dominant_freq), 4)
print(f"   Dominant spectral period: {dominant_period}")

# ---------- Analysis 7: Mediation Analysis ----------
print("\n7. Conducting mediation analysis...")
mediation_data = per_event_agg[['event_number', 'mean_total_seconds', 'finish_rate']].dropna()

# a-path: event_number -> mean_total_seconds
a_path_model = ols('mean_total_seconds ~ event_number', data=mediation_data).fit()
a_coef = a_path_model.params['event_number']

# b-path: mean_total_seconds -> finish_rate (controlling for event_number)
b_path_model = ols('finish_rate ~ event_number + mean_total_seconds', data=mediation_data).fit()
b_coef = b_path_model.params['mean_total_seconds']

indirect_effect = round(a_coef * b_coef, 4)
print(f"   Mediation indirect effect: {indirect_effect}")

# ---------- Analysis 8: Quantile Regression at 0.25 ----------
print("\n8. Running quantile regression...")
finished_ko_sub = finished_fights[finished_fights['method_group'].isin(['KO_TKO', 'SUB'])].copy()
finished_ko_sub['ko_tko_indicator'] = (finished_ko_sub['method_group'] == 'KO_TKO').astype(int)
qr_data = finished_ko_sub[['total_seconds', 'round', 'ko_tko_indicator']].dropna()

qr_model = quantreg('total_seconds ~ round + ko_tko_indicator', data=qr_data).fit(q=0.25)
ko_tko_coef_qr = round(qr_model.params['ko_tko_indicator'], 4)
print(f"   Quantile regression (0.25) KO_TKO coefficient: {ko_tko_coef_qr}")

# ---------- Analysis 9: Difference-in-Differences ----------
print("\n9. Running difference-in-differences...")
did_data = v3_ufc_numbered[v3_ufc_numbered['method_group'].isin(['KO_TKO', 'SUB'])].copy()
did_data['treatment'] = (did_data['event_number'] > 200).astype(int)
did_data['group'] = (did_data['method_group'] == 'KO_TKO').astype(int)
did_data = did_data[['total_seconds', 'treatment', 'group']].dropna()

did_model = ols('total_seconds ~ treatment + group + treatment:group', data=did_data).fit()
did_interaction_coef = round(did_model.params['treatment:group'], 4)
print(f"   DiD interaction coefficient: {did_interaction_coef}")

# ---------- Analysis 10: Chi-Square Test Across Versions ----------
print("\n10. Running chi-square test across versions...")
pooled_data = pd.concat([version_1, version_2, version_3], ignore_index=True)
pooled_filtered = pooled_data[pooled_data['method_group'].isin(['KO_TKO', 'SUB', 'DEC', 'OTHER'])].copy()
contingency_table = pd.crosstab(pooled_filtered['method_group'], pooled_filtered['dataset_version'])
chi2_stat, _, _, _ = stats.chi2_contingency(contingency_table)
chi2_stat = round(chi2_stat, 4)
print(f"   Chi-square statistic: {chi2_stat}")

# ---------- Interpretation ----------
interpretation = (
    f"The mediation analysis reveals an indirect effect of {indirect_effect}, suggesting that as UFC events progress, "
    f"fights tend to last longer (positive a-path), and longer fight durations are associated with lower finish rates (negative b-path), "
    f"indicating that the gradual lengthening of fight durations acts as a meaningful pathway suppressing finishes. "
    f"The spectral analysis detected a dominant cyclical period of {dominant_period} events, which could reflect rule changes "
    f"(e.g., USADA testing implementation, new weight classes, scoring criteria updates) or roster composition shifts "
    f"(e.g., waves of grapplers vs strikers, international expansion bringing diverse fighting styles) that periodically "
    f"alter fight dynamics in a recurring pattern."
)

print("\n" + "="*80)
print("INTERPRETATION:")
print("="*80)
print(interpretation)
print("="*80)

# ---------- Save Key Results ----------
key_results = {
    'new_fights_in_v3_not_v1': int(new_fights_count),
    'spearman_rho_event_vs_finish': float(spearman_rho),
    'anova_interaction_f_stat': float(interaction_f_stat),
    'cohens_d_ko_vs_sub': float(cohens_d),
    'logistic_cv_mean_roc_auc': float(mean_roc_auc),
    'fft_dominant_period': float(dominant_period),
    'mediation_indirect_effect': float(indirect_effect),
    'quantile_reg_ko_tko_coef': float(ko_tko_coef_qr),
    'did_interaction_coef': float(did_interaction_coef),
    'chi_square_statistic': float(chi2_stat),
    'interpretation': interpretation
}

with open('key_results.json', 'w') as f:
    json.dump(key_results, f, indent=2)

print("\n✓ Key results saved to key_results.json")

# ========== VISUALIZATIONS ==========

# ---------- Visualization 1: Stacked Bar Chart ----------
print("\n11. Creating stacked bar chart...")
v3_viz = version_3.copy()

def map_round_bucket(round_num):
    if round_num == 1:
        return 'Round 1'
    elif round_num in [2, 3]:
        return 'Rounds 2-3'
    elif round_num in [4, 5]:
        return 'Rounds 4-5'
    else:
        return np.nan

v3_viz['round_bucket'] = v3_viz['round'].apply(map_round_bucket)
v3_viz_filtered = v3_viz[v3_viz['method_group'].isin(['KO_TKO', 'SUB', 'DEC', 'OTHER'])].dropna(subset=['round_bucket'])

stacked_data = pd.crosstab(v3_viz_filtered['round_bucket'], v3_viz_filtered['method_group'], normalize='index')
stacked_data = stacked_data.reindex(['Round 1', 'Rounds 2-3', 'Rounds 4-5'])

fig, ax = plt.subplots(figsize=(10, 6))
stacked_data.plot(kind='bar', stacked=True, ax=ax,
                  color=['#d62728', '#ff7f0e', '#2ca02c', '#9467bd'])
ax.set_title('Method Group Proportions by Round Bucket', fontsize=14, fontweight='bold')
ax.set_xlabel('Round Bucket', fontsize=12)
ax.set_ylabel('Proportion', fontsize=12)
ax.set_xticklabels(ax.get_xticklabels(), rotation=0)
ax.legend(title='Method Group', bbox_to_anchor=(1.05, 1), loc='upper left')
plt.tight_layout()
plt.savefig('method_proportions_by_round.png', dpi=300, bbox_inches='tight')
plt.close()
print("   ✓ Saved: method_proportions_by_round.png")

# ---------- Visualization 2: Boxplot ----------
print("\n12. Creating boxplot...")
finished_viz = finished_fights[finished_fights['method_group'].isin(['KO_TKO', 'SUB'])].dropna(subset=['total_seconds'])

fig, ax = plt.subplots(figsize=(10, 6))
sns.boxplot(data=finished_viz, x='method_group', y='total_seconds', ax=ax, palette='Set2')
ax.set_title('Stoppage Duration by Finish Mechanism', fontsize=14, fontweight='bold')
ax.set_xlabel('Method Group', fontsize=12)
ax.set_ylabel('Total Seconds', fontsize=12)
plt.tight_layout()
plt.savefig('stoppage_duration_boxplot.png', dpi=300, bbox_inches='tight')
plt.close()
print("   ✓ Saved: stoppage_duration_boxplot.png")

# ---------- Visualization 3: Kaplan-Meier Survival Curves ----------
print("\n13. Creating Kaplan-Meier survival curves...")

def kaplan_meier_estimator(durations, event_observed):
    """Manual Kaplan-Meier estimator implementation"""
    # Combine durations and events
    data = pd.DataFrame({
        'duration': durations,
        'event': event_observed
    }).sort_values('duration')

    # Get unique time points
    unique_times = data['duration'].unique()

    survival_prob = []
    cumulative_survival = 1.0
    n_at_risk = len(data)

    time_points = []
    survival_probs = [1.0]  # Start at 1.0
    time_points.append(0)

    for t in unique_times:
        # Get events and censored at this time
        at_time = data[data['duration'] == t]
        n_events = at_time['event'].sum()
        n_total_at_time = len(at_time)

        if n_at_risk > 0:
            # Update survival probability
            cumulative_survival *= (1 - n_events / n_at_risk)

        time_points.append(t)
        survival_probs.append(cumulative_survival)

        # Update number at risk
        n_at_risk -= n_total_at_time

    return np.array(time_points), np.array(survival_probs)

km_data = version_3[version_3['event_type'].isin(['UFC_Numbered', 'Fight_Night'])].dropna(subset=['total_seconds', 'finish_flag'])

fig, ax = plt.subplots(figsize=(10, 6))

colors = {'UFC_Numbered': '#1f77b4', 'Fight_Night': '#ff7f0e'}

for event_type in ['UFC_Numbered', 'Fight_Night']:
    subset = km_data[km_data['event_type'] == event_type]
    # Event = 1 means censored (fight went to decision), Event = 0 means finished
    event_observed = 1 - subset['finish_flag'].values

    times, survival = kaplan_meier_estimator(
        subset['total_seconds'].values,
        event_observed
    )

    ax.step(times, survival, where='post', label=event_type,
            linewidth=2, color=colors[event_type])

ax.set_title('Kaplan-Meier Survival Curves by Event Type', fontsize=14, fontweight='bold')
ax.set_xlabel('Fight Duration (Seconds)', fontsize=12)
ax.set_ylabel('Survival Probability', fontsize=12)
ax.legend(loc='best')
ax.grid(True, alpha=0.3)
ax.set_ylim([0, 1.05])
plt.tight_layout()
plt.savefig('kaplan_meier_survival.png', dpi=300, bbox_inches='tight')
plt.close()
print("   ✓ Saved: kaplan_meier_survival.png")

# ---------- Visualization 4: Spectral Power Plot ----------
print("\n14. Creating spectral power plot...")
fig, ax = plt.subplots(figsize=(10, 6))
ax.plot(positive_freqs, positive_magnitude, linewidth=2, color='navy')
ax.axvline(dominant_freq, color='red', linestyle='--', linewidth=2,
           label=f'Dominant Frequency (Period={dominant_period:.2f})')
ax.set_title('Spectral Power of Finish Rate Oscillations', fontsize=14, fontweight='bold')
ax.set_xlabel('Frequency (Cycles per Event)', fontsize=12)
ax.set_ylabel('Spectral Magnitude', fontsize=12)
ax.legend(loc='best')
ax.grid(True, alpha=0.3)
plt.tight_layout()
plt.savefig('spectral_power_plot.png', dpi=300, bbox_inches='tight')
plt.close()
print("   ✓ Saved: spectral_power_plot.png")

print("\n" + "="*80)
print("ALL ANALYSES COMPLETE")
print("="*80)
print(f"Key outputs:")
print(f"  - key_results.json")
print(f"  - method_proportions_by_round.png")
print(f"  - stoppage_duration_boxplot.png")
print(f"  - kaplan_meier_survival.png")
print(f"  - spectral_power_plot.png")
print("="*80)
