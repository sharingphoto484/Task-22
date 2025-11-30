# ==========================================
# Business Activity & Support Analysis Script
# Requirements: pandas, numpy, matplotlib, scipy, scikit-learn, statsmodels
# Input files: deals.csv, contacts.csv, tickets.csv (in same directory)
# Output files: 10 visualizations (png), key_outputs.txt
# ==========================================

import pandas as pd
import numpy as np
import matplotlib.pyplot as plt
import seaborn as sns
from scipy import stats
from scipy.stats import chi2_contingency, spearmanr
from sklearn.linear_model import LogisticRegression, LinearRegression
from sklearn.preprocessing import LabelEncoder
from statsmodels.tsa.arima.model import ARIMA
import warnings
warnings.filterwarnings('ignore')

# Set plotting style
sns.set_style('whitegrid')
plt.rcParams['figure.figsize'] = (10, 6)

# Storage for key outputs
key_outputs = {}

# ---------- Load CSVs Robustly ----------
print("Loading datasets...")
deals_df = pd.read_csv('deals.csv')
contacts_df = pd.read_csv('contacts.csv')
tickets_df = pd.read_csv('tickets.csv')

# ---------- Data Cleaning and Preparation ----------
print("Cleaning and preparing data...")

# Convert date fields to datetime
deals_df['close_date'] = pd.to_datetime(deals_df['close_date'], errors='coerce')
tickets_df['createdate'] = pd.to_datetime(tickets_df['createdate'], errors='coerce')
tickets_df['closed_date'] = pd.to_datetime(tickets_df['closed_date'], errors='coerce')

# Remove rows with missing values in required fields for deals
deals_df = deals_df.dropna(subset=['amount'])
deals_df = deals_df[pd.to_numeric(deals_df['amount'], errors='coerce').notna()]

# Clean tickets - keep all rows but handle missing values appropriately
tickets_df = tickets_df[tickets_df['createdate'].notna()]

print(f"Loaded {len(deals_df)} deals, {len(contacts_df)} contacts, {len(tickets_df)} tickets")

# ==========================================
# ANALYSIS 1: Ticket Closure Behavior - Logistic Regression
# ==========================================
print("\n" + "="*50)
print("ANALYSIS 1: Ticket Closure Behavior - Logistic Regression")
print("="*50)

# Calculate closure duration for tickets with both dates
tickets_with_closure = tickets_df[tickets_df['closed_date'].notna() & tickets_df['createdate'].notna()].copy()
tickets_with_closure['closure_duration'] = (tickets_with_closure['closed_date'] - tickets_with_closure['createdate']).dt.days

# Create binary variable ClosedWithin30
tickets_with_closure['ClosedWithin30'] = ((tickets_with_closure['closure_duration'] >= 0) &
                                           (tickets_with_closure['closure_duration'] <= 30)).astype(int)

# Create has_company indicator
tickets_with_closure['has_company'] = tickets_with_closure['associated_company_id'].notna().astype(int)

# Encode priority as ordered categorical
priority_order = {'Low': 0, 'Medium': 1, 'High': 2}
tickets_with_closure['priority_encoded'] = tickets_with_closure['priority'].map(priority_order)

# Remove rows with missing priority
tickets_logistic = tickets_with_closure.dropna(subset=['priority_encoded'])

# Fit logistic regression
X = tickets_logistic[['priority_encoded', 'has_company']].values
y = tickets_logistic['ClosedWithin30'].values

log_model = LogisticRegression(random_state=42, max_iter=1000)
log_model.fit(X, y)

priority_coefficient = log_model.coef_[0][0]
key_outputs['logistic_priority_coefficient'] = round(priority_coefficient, 3)
print(f"Priority coefficient (logistic regression): {key_outputs['logistic_priority_coefficient']}")

# Generate scatter plot for predicted closure probability
priority_values = np.array([0, 1, 2])  # Low, Medium, High
has_company_value = 1  # Average case with company

predictions = []
for priority in priority_values:
    X_pred = np.array([[priority, has_company_value]])
    prob = log_model.predict_proba(X_pred)[0][1]
    predictions.append(prob)

plt.figure(figsize=(10, 6))
plt.scatter(priority_values, predictions, s=200, alpha=0.6, c=['green', 'orange', 'red'])
plt.plot(priority_values, predictions, 'b--', alpha=0.5)
plt.xlabel('Ticket Priority (0=Low, 1=Medium, 2=High)', fontsize=12)
plt.ylabel('Predicted Probability of Closure Within 30 Days', fontsize=12)
plt.title('Ticket Closure Probability by Priority', fontsize=14, fontweight='bold')
plt.xticks([0, 1, 2], ['Low', 'Medium', 'High'])
plt.ylim(0, 1)
plt.grid(True, alpha=0.3)
plt.tight_layout()
plt.savefig('predicted_closure_probability.png', dpi=300, bbox_inches='tight')
plt.close()
print("Generated: predicted_closure_probability.png")

# ==========================================
# ANALYSIS 2: Resolution Speed
# ==========================================
print("\n" + "="*50)
print("ANALYSIS 2: Resolution Speed")
print("="*50)

# Calculate resolution duration for valid tickets
tickets_resolved = tickets_df[(tickets_df['createdate'].notna()) &
                              (tickets_df['closed_date'].notna())].copy()
tickets_resolved['resolution_duration'] = (tickets_resolved['closed_date'] - tickets_resolved['createdate']).dt.days
tickets_resolved = tickets_resolved[tickets_resolved['resolution_duration'] >= 0]

mean_resolution = tickets_resolved['resolution_duration'].mean()
key_outputs['mean_resolution_duration'] = round(mean_resolution, 2)
print(f"Mean resolution duration: {key_outputs['mean_resolution_duration']} days")

# Generate histogram
plt.figure(figsize=(10, 6))
plt.hist(tickets_resolved['resolution_duration'], bins=50, color='skyblue', edgecolor='black', alpha=0.7)
plt.xlabel('Resolution Time (Days)', fontsize=12)
plt.ylabel('Count of Tickets', fontsize=12)
plt.title('Distribution of Ticket Resolution Duration', fontsize=14, fontweight='bold')
plt.grid(True, alpha=0.3, axis='y')
plt.tight_layout()
plt.savefig('resolution_duration_histogram.png', dpi=300, bbox_inches='tight')
plt.close()
print("Generated: resolution_duration_histogram.png")

# ==========================================
# ANALYSIS 3: Temporal Patterns - Linear Regression
# ==========================================
print("\n" + "="*50)
print("ANALYSIS 3: Temporal Patterns in Ticket Creation")
print("="*50)

# Group tickets by createdate
daily_counts = tickets_df.groupby(tickets_df['createdate'].dt.date).size().reset_index()
daily_counts.columns = ['date', 'count']
daily_counts = daily_counts.sort_values('date')

# Create time index
daily_counts['time_index'] = range(1, len(daily_counts) + 1)

# Fit linear regression
X_time = daily_counts['time_index'].values.reshape(-1, 1)
y_count = daily_counts['count'].values

lin_model = LinearRegression()
lin_model.fit(X_time, y_count)

slope = lin_model.coef_[0]
key_outputs['temporal_slope'] = round(slope, 4)
print(f"Temporal trend slope: {key_outputs['temporal_slope']} tickets/day")

# Generate line plot
plt.figure(figsize=(12, 6))
plt.plot(daily_counts['date'], daily_counts['count'], marker='o', markersize=3, linewidth=1, alpha=0.7)
plt.plot(daily_counts['date'], lin_model.predict(X_time), 'r--', linewidth=2, label=f'Trend (slope={slope:.4f})')
plt.xlabel('Date', fontsize=12)
plt.ylabel('Number of Tickets Created', fontsize=12)
plt.title('Daily Ticket Volume Over Time', fontsize=14, fontweight='bold')
plt.xticks(rotation=45)
plt.legend()
plt.grid(True, alpha=0.3)
plt.tight_layout()
plt.savefig('daily_ticket_volume.png', dpi=300, bbox_inches='tight')
plt.close()
print("Generated: daily_ticket_volume.png")

# ==========================================
# ANALYSIS 4: ARIMA Forecasting
# ==========================================
print("\n" + "="*50)
print("ANALYSIS 4: ARIMA Forecasting")
print("="*50)

# Prepare daily count series
daily_series = daily_counts.set_index('date')['count']

try:
    # Fit ARIMA(1,1,1)
    arima_model = ARIMA(daily_series, order=(1, 1, 1))
    arima_fit = arima_model.fit()

    # Forecast next 30 days
    forecast = arima_fit.forecast(steps=30)
    mean_forecast = forecast.mean()
    key_outputs['mean_arima_forecast'] = round(mean_forecast, 2)
    print(f"Mean ARIMA forecast (next 30 days): {key_outputs['mean_arima_forecast']} tickets/day")

    # Generate area chart
    plt.figure(figsize=(12, 6))

    # Historical data
    plt.plot(daily_series.index, daily_series.values, label='Historical', color='blue', linewidth=2)

    # Forecast
    forecast_dates = pd.date_range(start=daily_series.index[-1] + pd.Timedelta(days=1), periods=30)
    plt.fill_between(forecast_dates, 0, forecast.values, alpha=0.3, color='orange', label='Forecast')
    plt.plot(forecast_dates, forecast.values, color='orange', linewidth=2)

    plt.xlabel('Date', fontsize=12)
    plt.ylabel('Daily Ticket Count', fontsize=12)
    plt.title('Ticket Volume Forecast (ARIMA 1,1,1)', fontsize=14, fontweight='bold')
    plt.legend()
    plt.xticks(rotation=45)
    plt.grid(True, alpha=0.3)
    plt.tight_layout()
    plt.savefig('ticket_volume_forecast.png', dpi=300, bbox_inches='tight')
    plt.close()
    print("Generated: ticket_volume_forecast.png")

except Exception as e:
    print(f"ARIMA forecasting failed: {e}")
    key_outputs['mean_arima_forecast'] = "N/A"

# ==========================================
# ANALYSIS 5: Customer Engagement vs Commercial Value
# ==========================================
print("\n" + "="*50)
print("ANALYSIS 5: Customer Engagement vs Commercial Value")
print("="*50)

# Count tickets per contact
ticket_counts = tickets_df.groupby('associated_contact_id').size().reset_index()
ticket_counts.columns = ['contact_id', 'ticket_count']

# Sum deal amounts per contact
deal_totals = deals_df.groupby('contact_id')['amount'].sum().reset_index()
deal_totals.columns = ['contact_id', 'total_deal_amount']

# Merge with contacts
contacts_analysis = contacts_df[['contact_id']].copy()
contacts_analysis = contacts_analysis.merge(ticket_counts, on='contact_id', how='left')
contacts_analysis = contacts_analysis.merge(deal_totals, on='contact_id', how='left')

# Fill missing values with 0
contacts_analysis['ticket_count'] = contacts_analysis['ticket_count'].fillna(0)
contacts_analysis['total_deal_amount'] = contacts_analysis['total_deal_amount'].fillna(0)

# Calculate Spearman correlation
spearman_corr, p_value = spearmanr(contacts_analysis['ticket_count'], contacts_analysis['total_deal_amount'])
key_outputs['spearman_correlation'] = round(spearman_corr, 3)
print(f"Spearman correlation (tickets vs deal value): {key_outputs['spearman_correlation']}")

# Create ticket count groups for bar chart
contacts_analysis['ticket_group'] = pd.cut(contacts_analysis['ticket_count'],
                                            bins=[-1, 0, 1, 2, 5, 100],
                                            labels=['0', '1', '2', '3-5', '6+'])

# Calculate mean deal amount per group
group_means = contacts_analysis.groupby('ticket_group', observed=True)['total_deal_amount'].mean().reset_index()

# Generate bar chart
plt.figure(figsize=(10, 6))
plt.bar(range(len(group_means)), group_means['total_deal_amount'], color='steelblue', alpha=0.7, edgecolor='black')
plt.xlabel('Ticket Count Groups', fontsize=12)
plt.ylabel('Mean Total Deal Amount ($)', fontsize=12)
plt.title('Deal Value by Customer Ticket Load', fontsize=14, fontweight='bold')
plt.xticks(range(len(group_means)), group_means['ticket_group'])
plt.grid(True, alpha=0.3, axis='y')
plt.tight_layout()
plt.savefig('ticket_load_deal_value.png', dpi=300, bbox_inches='tight')
plt.close()
print("Generated: ticket_load_deal_value.png")

# ==========================================
# ANALYSIS 6: Priority vs Status Chi-Square Test
# ==========================================
print("\n" + "="*50)
print("ANALYSIS 6: Priority vs Status Chi-Square Test")
print("="*50)

# Create contingency table
contingency_table = pd.crosstab(tickets_df['priority'], tickets_df['status'])

# Chi-square test
chi2, p, dof, expected = chi2_contingency(contingency_table)
key_outputs['chi_square_statistic'] = round(chi2, 3)
print(f"Chi-square test statistic: {key_outputs['chi_square_statistic']}")

# Generate heatmap
plt.figure(figsize=(10, 6))
sns.heatmap(contingency_table, annot=True, fmt='d', cmap='YlOrRd', cbar_kws={'label': 'Count'})
plt.xlabel('Ticket Status', fontsize=12)
plt.ylabel('Ticket Priority', fontsize=12)
plt.title('Priority-Status Relationship Heatmap', fontsize=14, fontweight='bold')
plt.tight_layout()
plt.savefig('priority_status_heatmap.png', dpi=300, bbox_inches='tight')
plt.close()
print("Generated: priority_status_heatmap.png")

# ==========================================
# ANALYSIS 7: Deal Value Coefficient of Variation
# ==========================================
print("\n" + "="*50)
print("ANALYSIS 7: Deal Value Coefficient of Variation")
print("="*50)

# Calculate CV
deal_std = deals_df['amount'].std()
deal_mean = deals_df['amount'].mean()
cv = deal_std / deal_mean
key_outputs['coefficient_of_variation'] = round(cv, 4)
print(f"Coefficient of variation (deal amount): {key_outputs['coefficient_of_variation']}")

# Generate density plot
plt.figure(figsize=(10, 6))
deals_df['amount'].plot(kind='density', linewidth=2, color='darkblue')
plt.xlabel('Deal Amount ($)', fontsize=12)
plt.ylabel('Density', fontsize=12)
plt.title('Deal Amount Distribution (Density Plot)', fontsize=14, fontweight='bold')
plt.grid(True, alpha=0.3)
plt.tight_layout()
plt.savefig('deal_amount_density.png', dpi=300, bbox_inches='tight')
plt.close()
print("Generated: deal_amount_density.png")

# ==========================================
# ANALYSIS 8: Rolling Average Analysis
# ==========================================
print("\n" + "="*50)
print("ANALYSIS 8: Seven-Day Rolling Average")
print("="*50)

# Reindex to include all dates in range with 0 for missing days
date_range = pd.date_range(start=daily_counts['date'].min(), end=daily_counts['date'].max())
daily_complete = pd.DataFrame({'date': date_range.date})
daily_complete = daily_complete.merge(daily_counts[['date', 'count']], on='date', how='left')
daily_complete['count'] = daily_complete['count'].fillna(0)

# Calculate 7-day rolling average
daily_complete['rolling_7day'] = daily_complete['count'].rolling(window=7, min_periods=1).mean()

# Find maximum
max_rolling = daily_complete['rolling_7day'].max()
key_outputs['max_rolling_average'] = round(max_rolling, 2)
print(f"Maximum 7-day rolling average: {key_outputs['max_rolling_average']} tickets/day")

# Generate spiral plot (using polar coordinates)
plt.figure(figsize=(10, 10))
ax = plt.subplot(111, projection='polar')

# Convert day index to angular position
n_days = len(daily_complete)
theta = np.linspace(0, 8 * np.pi, n_days)  # Multiple spirals
r = daily_complete['rolling_7day'].values

ax.plot(theta, r, linewidth=1.5, alpha=0.7)
ax.set_title('Spiral Plot: 7-Day Rolling Ticket Volume', fontsize=14, fontweight='bold', pad=20)
plt.tight_layout()
plt.savefig('rolling_ticket_volume_spiral.png', dpi=300, bbox_inches='tight')
plt.close()
print("Generated: rolling_ticket_volume_spiral.png")

# ==========================================
# ANALYSIS 9: Deal Stage Analysis
# ==========================================
print("\n" + "="*50)
print("ANALYSIS 9: Deal Stage Analysis")
print("="*50)

# Calculate mean amount per stage
stage_means = deals_df.groupby('stage')['amount'].mean().reset_index()
stage_means = stage_means.sort_values('amount', ascending=False)

# Find stage with largest mean
best_stage = stage_means.iloc[0]['stage']
key_outputs['best_performing_stage'] = best_stage
print(f"Stage with largest mean deal amount: {key_outputs['best_performing_stage']}")

# Generate boxplot
plt.figure(figsize=(12, 6))
stages_order = stage_means['stage'].tolist()
sns.boxplot(data=deals_df, x='stage', y='amount', order=stages_order, palette='Set2')
plt.xlabel('Deal Stage', fontsize=12)
plt.ylabel('Deal Amount ($)', fontsize=12)
plt.title('Deal Amount Distribution by Stage', fontsize=14, fontweight='bold')
plt.xticks(rotation=45, ha='right')
plt.grid(True, alpha=0.3, axis='y')
plt.tight_layout()
plt.savefig('deal_amount_by_stage.png', dpi=300, bbox_inches='tight')
plt.close()
print("Generated: deal_amount_by_stage.png")

# ==========================================
# WRITE KEY OUTPUTS TO FILE
# ==========================================
print("\n" + "="*50)
print("WRITING KEY OUTPUTS")
print("="*50)

with open('key_outputs.txt', 'w') as f:
    f.write("="*60 + "\n")
    f.write("BUSINESS ACTIVITY & SUPPORT ANALYSIS - KEY OUTPUTS\n")
    f.write("="*60 + "\n\n")

    f.write("1. TICKET CLOSURE BEHAVIOR (LOGISTIC REGRESSION)\n")
    f.write(f"   Priority Coefficient: {key_outputs['logistic_priority_coefficient']}\n\n")

    f.write("2. RESOLUTION SPEED\n")
    f.write(f"   Mean Resolution Duration: {key_outputs['mean_resolution_duration']} days\n\n")

    f.write("3. TEMPORAL PATTERNS (LINEAR REGRESSION)\n")
    f.write(f"   Temporal Slope: {key_outputs['temporal_slope']} tickets/day\n\n")

    f.write("4. ARIMA FORECASTING\n")
    f.write(f"   Mean Forecast (30 days): {key_outputs['mean_arima_forecast']} tickets/day\n\n")

    f.write("5. CUSTOMER ENGAGEMENT vs COMMERCIAL VALUE\n")
    f.write(f"   Spearman Correlation: {key_outputs['spearman_correlation']}\n\n")

    f.write("6. PRIORITY vs STATUS RELATIONSHIP\n")
    f.write(f"   Chi-Square Statistic: {key_outputs['chi_square_statistic']}\n\n")

    f.write("7. DEAL VALUE VARIABILITY\n")
    f.write(f"   Coefficient of Variation: {key_outputs['coefficient_of_variation']}\n\n")

    f.write("8. ROLLING AVERAGE ANALYSIS\n")
    f.write(f"   Maximum 7-Day Rolling Average: {key_outputs['max_rolling_average']} tickets/day\n\n")

    f.write("9. DEAL STAGE PERFORMANCE\n")
    f.write(f"   Best Performing Stage: {key_outputs['best_performing_stage']}\n\n")

    f.write("="*60 + "\n")

print("\nKey outputs written to: key_outputs.txt")

# ==========================================
# SUMMARY
# ==========================================
print("\n" + "="*60)
print("ANALYSIS COMPLETE")
print("="*60)
print("\nGenerated Files:")
print("  1. predicted_closure_probability.png")
print("  2. resolution_duration_histogram.png")
print("  3. daily_ticket_volume.png")
print("  4. ticket_volume_forecast.png")
print("  5. ticket_load_deal_value.png")
print("  6. priority_status_heatmap.png")
print("  7. deal_amount_density.png")
print("  8. rolling_ticket_volume_spiral.png")
print("  9. deal_amount_by_stage.png")
print(" 10. key_outputs.txt")
print("\n" + "="*60)
print("\nKEY OUTPUTS:")
print("="*60)
for key, value in key_outputs.items():
    print(f"{key}: {value}")
print("="*60)
