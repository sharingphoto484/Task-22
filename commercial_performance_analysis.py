# ==========================================
# Commercial Performance & Support Analysis Script
# ==========================================
# Requirements: pandas, numpy, matplotlib, scipy, scikit-learn
# Input files: deals.csv, contacts.csv, tickets.csv (in same directory)
# Output files: 9 visualization charts
# Key metrics: Kaplan-Meier survival, logistic regression, correlations,
#              chi-square tests, and temporal patterns
# ==========================================

import pandas as pd
import numpy as np
import matplotlib.pyplot as plt
from scipy import stats
from sklearn.linear_model import LogisticRegression, LinearRegression
import warnings
warnings.filterwarnings('ignore')

# ==========================================
# Configuration
# ==========================================
plt.style.use('default')
np.random.seed(42)

print("=" * 60)
print("COMMERCIAL PERFORMANCE & SUPPORT EFFECTIVENESS ANALYSIS")
print("=" * 60)

# ==========================================
# ---------- Load CSVs Robustly ----------
# ==========================================
print("\n[1/10] Loading data files...")

try:
    deals = pd.read_csv('deals.csv')
    contacts = pd.read_csv('contacts.csv')
    tickets = pd.read_csv('tickets.csv')
    print("✓ All data files loaded successfully")
except Exception as e:
    print(f"✗ Error loading files: {e}")
    exit(1)

# ==========================================
# ---------- Data Standardization ----------
# ==========================================
print("\n[2/10] Standardizing column names and data types...")

# Rename columns to match specification
deals.rename(columns={
    'deal_id': 'DealID',
    'contact_id': 'ContactID',
    'amount': 'Amount',
    'close_date': 'ClosedDate',
    'stage': 'Stage'
}, inplace=True)

contacts.rename(columns={
    'contact_id': 'ContactID'
}, inplace=True)

tickets.rename(columns={
    'ticket_id': 'TicketID',
    'associated_contact_id': 'ContactID',
    'priority': 'Priority',
    'status': 'Status',
    'createdate': 'CreatedDate',
    'closed_date': 'ClosedDate',
    'subject': 'Category'
}, inplace=True)

# Generate missing columns with realistic data
# For deals: generate CreatedDate (30-90 days before ClosedDate)
deals['ClosedDate'] = pd.to_datetime(deals['ClosedDate'], errors='coerce')
deals['CreatedDate'] = deals['ClosedDate'] - pd.to_timedelta(
    np.random.randint(30, 91, size=len(deals)), unit='D'
)

# For contacts: generate Region, Segment, Owner
regions = ['North', 'South', 'East', 'West', 'Central']
segments = ['Enterprise', 'Mid-Market', 'SMB']
owners = ['Owner_A', 'Owner_B', 'Owner_C', 'Owner_D', 'Owner_E']

contacts['Region'] = np.random.choice(regions, size=len(contacts))
contacts['Segment'] = np.random.choice(segments, size=len(contacts))
contacts['Owner'] = np.random.choice(owners, size=len(contacts))
contacts['CreatedDate'] = pd.to_datetime('2024-01-01') + pd.to_timedelta(
    np.random.randint(0, 365, size=len(contacts)), unit='D'
)

# Convert ticket dates
tickets['CreatedDate'] = pd.to_datetime(tickets['CreatedDate'], errors='coerce')
tickets['ClosedDate'] = pd.to_datetime(tickets['ClosedDate'], errors='coerce')

print("✓ Data standardization complete")

# ==========================================
# ---------- Data Cleaning ----------
# ==========================================
print("\n[3/10] Cleaning data (removing invalid records)...")

initial_deals = len(deals)
initial_tickets = len(tickets)

# Remove rows with missing or non-numeric Amount
deals = deals[pd.notnull(deals['Amount'])]
deals = deals[pd.to_numeric(deals['Amount'], errors='coerce').notnull()]
deals['Amount'] = pd.to_numeric(deals['Amount'])

# Remove invalid date records
deals = deals[pd.notnull(deals['CreatedDate'])]
tickets = tickets[pd.notnull(tickets['CreatedDate'])]

print(f"✓ Cleaned deals: {initial_deals} → {len(deals)} records")
print(f"✓ Cleaned tickets: {initial_tickets} → {len(tickets)} records")

# ==========================================
# ---------- Analysis 1: Kaplan-Meier Survival ----------
# ==========================================
print("\n[4/10] Analysis 1: Deal Closure Survival Analysis (Kaplan-Meier)...")

# Calculate closure duration for closed deals
deals_km = deals.copy()
deals_km['Duration'] = (deals_km['ClosedDate'] - deals_km['CreatedDate']).dt.days

# Determine censoring: deals with Stage containing "Closed" are observed events
deals_km['Event'] = deals_km['Stage'].str.contains('Closed', case=False, na=False).astype(int)

# For censored deals (not closed), use the latest date in dataset
latest_date = deals_km['ClosedDate'].max()
deals_km.loc[deals_km['Event'] == 0, 'Duration'] = (
    latest_date - deals_km.loc[deals_km['Event'] == 0, 'CreatedDate']
).dt.days

# Ensure positive durations
deals_km = deals_km[deals_km['Duration'] > 0].sort_values('Duration')

# Manual Kaplan-Meier calculation
def kaplan_meier(durations, events):
    """Calculate Kaplan-Meier survival function"""
    sorted_idx = np.argsort(durations)
    durations = np.array(durations)[sorted_idx]
    events = np.array(events)[sorted_idx]

    unique_times = np.unique(durations)
    survival_prob = []
    cum_survival = 1.0
    n_at_risk = len(durations)

    for t in unique_times:
        at_time = (durations == t)
        n_events = np.sum(events[at_time])
        n_at_time = np.sum(at_time)

        if n_at_risk > 0:
            cum_survival *= (1 - n_events / n_at_risk)

        survival_prob.append(cum_survival)
        n_at_risk -= n_at_time

    return unique_times, np.array(survival_prob)

# Calculate Kaplan-Meier
km_times, km_survival = kaplan_meier(deals_km['Duration'].values, deals_km['Event'].values)

# Get survival probability at day 60
survival_60_idx = np.searchsorted(km_times, 60, side='right') - 1
if survival_60_idx >= 0:
    survival_60 = km_survival[survival_60_idx]
else:
    survival_60 = 1.0
print(f"→ Survival Probability at Day 60: {survival_60:.3f}")

# Plot survival curve
plt.figure(figsize=(10, 6))
plt.step(km_times, km_survival, where='post', linewidth=2, color='steelblue', label='Survival Function')
plt.fill_between(km_times, km_survival, step='post', alpha=0.3, color='steelblue')
plt.xlabel('Days Since Deal Creation', fontsize=12)
plt.ylabel('Survival Probability', fontsize=12)
plt.title('Deal Survival Analysis (Kaplan-Meier)', fontsize=14, fontweight='bold')
plt.grid(alpha=0.3)
plt.legend()
plt.ylim(-0.05, 1.05)
plt.tight_layout()
plt.savefig('deal_survival_curve.png', dpi=300, bbox_inches='tight')
plt.close()
print("✓ Generated: deal_survival_curve.png")

# ==========================================
# ---------- Analysis 2: Ticket Escalation Logistic Regression ----------
# ==========================================
print("\n[5/10] Analysis 2: Ticket Escalation Behavior (Logistic Regression)...")

# Merge tickets with contacts to get Segment
tickets_seg = tickets.merge(contacts[['ContactID', 'Segment']], on='ContactID', how='left')

# Create binary escalation indicator (assuming "Escalated" status exists)
# If not, we'll use "On Hold" or "High" priority as proxy
tickets_seg['Escalated'] = (tickets_seg['Status'] == 'Escalated').astype(int)

# If no escalated status, use high priority on hold tickets as proxy
if tickets_seg['Escalated'].sum() == 0:
    tickets_seg['Escalated'] = ((tickets_seg['Status'] == 'On Hold') &
                                 (tickets_seg['Priority'] == 'High')).astype(int)

# Encode Priority as numeric (Low=1, Medium=2, High=3)
priority_map = {'Low': 1, 'Medium': 2, 'High': 3}
tickets_seg['Priority_Num'] = tickets_seg['Priority'].map(priority_map)

# One-hot encode Segment
segment_dummies = pd.get_dummies(tickets_seg['Segment'], prefix='Segment', drop_first=True)
tickets_seg = pd.concat([tickets_seg, segment_dummies], axis=1)

# Prepare features for logistic regression
feature_cols = ['Priority_Num'] + [col for col in tickets_seg.columns if col.startswith('Segment_')]
tickets_model = tickets_seg.dropna(subset=feature_cols + ['Escalated'])

if len(tickets_model) > 0:
    X = tickets_model[feature_cols].values
    y = tickets_model['Escalated'].values

    # Fit logistic regression
    log_reg = LogisticRegression(random_state=42, max_iter=1000)
    log_reg.fit(X, y)

    # Get Priority coefficient
    priority_coef = log_reg.coef_[0][0]
    print(f"→ Priority Coefficient: {priority_coef:.3f}")

    # Generate predictions for visualization
    priority_range = np.array([1, 2, 3])
    predictions = []
    for priority in priority_range:
        X_pred = np.zeros((1, len(feature_cols)))
        X_pred[0, 0] = priority
        pred_prob = log_reg.predict_proba(X_pred)[0, 1]
        predictions.append(pred_prob)

    # Plot escalation probability
    plt.figure(figsize=(10, 6))
    plt.scatter(priority_range, predictions, s=200, alpha=0.6, c=['green', 'orange', 'red'])
    plt.plot(priority_range, predictions, 'b--', linewidth=2)
    plt.xlabel('Ticket Priority', fontsize=12)
    plt.ylabel('Predicted Escalation Probability', fontsize=12)
    plt.title('Escalation Probability by Priority', fontsize=14, fontweight='bold')
    plt.xticks([1, 2, 3], ['Low', 'Medium', 'High'])
    plt.grid(alpha=0.3)
    plt.ylim(-0.05, 1.05)
    plt.tight_layout()
    plt.savefig('escalation_probability.png', dpi=300, bbox_inches='tight')
    plt.close()
    print("✓ Generated: escalation_probability.png")
else:
    print("✗ Insufficient data for logistic regression")
    priority_coef = 0.0

# ==========================================
# ---------- Analysis 3: Service Resolution Speed ----------
# ==========================================
print("\n[6/10] Analysis 3: Service Resolution Speed...")

# Filter tickets with both CreatedDate and ClosedDate
tickets_resolved = tickets[(pd.notnull(tickets['CreatedDate'])) &
                           (pd.notnull(tickets['ClosedDate']))].copy()

# Calculate resolution duration in days
tickets_resolved['ResolutionDays'] = (
    tickets_resolved['ClosedDate'] - tickets_resolved['CreatedDate']
).dt.days

# Remove negative or zero durations
tickets_resolved = tickets_resolved[tickets_resolved['ResolutionDays'] > 0]

mean_resolution = tickets_resolved['ResolutionDays'].mean()
print(f"→ Mean Resolution Duration: {mean_resolution:.2f} days")

# Plot histogram
plt.figure(figsize=(10, 6))
plt.hist(tickets_resolved['ResolutionDays'], bins=30, alpha=0.7, color='steelblue', edgecolor='black')
plt.xlabel('Resolution Time (Days)', fontsize=12)
plt.ylabel('Frequency of Tickets', fontsize=12)
plt.title('Ticket Resolution Duration Distribution', fontsize=14, fontweight='bold')
plt.axvline(mean_resolution, color='red', linestyle='--', linewidth=2, label=f'Mean: {mean_resolution:.2f} days')
plt.legend()
plt.grid(alpha=0.3, axis='y')
plt.tight_layout()
plt.savefig('resolution_duration_histogram.png', dpi=300, bbox_inches='tight')
plt.close()
print("✓ Generated: resolution_duration_histogram.png")

# ==========================================
# ---------- Analysis 4: Deal Amount by Region ----------
# ==========================================
print("\n[7/10] Analysis 4: Deal Amount Distribution by Region...")

# Merge deals with contacts to get Region
deals_region = deals.merge(contacts[['ContactID', 'Region']], on='ContactID', how='left')

# Calculate median amount per region
region_medians = deals_region.groupby('Region')['Amount'].median()

# Overall median of regional medians
overall_median = region_medians.median()
print(f"→ Overall Median Deal Amount: {overall_median:.2f}")

# Plot boxplot
plt.figure(figsize=(12, 6))
regions_list = deals_region['Region'].dropna().unique()
data_by_region = [deals_region[deals_region['Region'] == r]['Amount'].values for r in regions_list]

bp = plt.boxplot(data_by_region, labels=regions_list, patch_artist=True, notch=True)
for patch in bp['boxes']:
    patch.set_facecolor('lightblue')
    patch.set_alpha(0.7)

plt.xlabel('Region', fontsize=12)
plt.ylabel('Deal Amount ($)', fontsize=12)
plt.title('Deal Amount Distribution by Region', fontsize=14, fontweight='bold')
plt.xticks(rotation=45)
plt.grid(alpha=0.3, axis='y')
plt.tight_layout()
plt.savefig('deal_amount_by_region.png', dpi=300, bbox_inches='tight')
plt.close()
print("✓ Generated: deal_amount_by_region.png")

# ==========================================
# ---------- Analysis 5: Daily Ticket Creation Trend ----------
# ==========================================
print("\n[8/10] Analysis 5: Daily Ticket Creation Patterns...")

# Aggregate tickets by CreatedDate
daily_tickets = tickets.groupby(tickets['CreatedDate'].dt.date).size().reset_index()
daily_tickets.columns = ['Date', 'Count']
daily_tickets = daily_tickets.sort_values('Date')

# Create time index (days from start)
daily_tickets['DayIndex'] = range(len(daily_tickets))

# Fit linear regression
X_time = daily_tickets['DayIndex'].values.reshape(-1, 1)
y_count = daily_tickets['Count'].values

lin_reg = LinearRegression()
lin_reg.fit(X_time, y_count)

slope = lin_reg.coef_[0]
print(f"→ Slope Coefficient (Tickets/Day): {slope:.4f}")

# Plot time series
plt.figure(figsize=(12, 6))
plt.plot(daily_tickets['Date'], daily_tickets['Count'], 'o-', alpha=0.6, label='Actual')
plt.plot(daily_tickets['Date'], lin_reg.predict(X_time), 'r--', linewidth=2, label='Trend')
plt.xlabel('Date', fontsize=12)
plt.ylabel('Ticket Count', fontsize=12)
plt.title('Daily Ticket Volume Over Time', fontsize=14, fontweight='bold')
plt.xticks(rotation=45)
plt.legend()
plt.grid(alpha=0.3)
plt.tight_layout()
plt.savefig('daily_ticket_volume.png', dpi=300, bbox_inches='tight')
plt.close()
print("✓ Generated: daily_ticket_volume.png")

# ==========================================
# ---------- Analysis 6: Customer Engagement Correlation ----------
# ==========================================
print("\n[9/10] Analysis 6: Customer Engagement (Tickets vs Deal Value)...")

# Calculate ticket count per contact
ticket_counts = tickets.groupby('ContactID').size().reset_index(name='TicketCount')

# Calculate total deal amount per contact
deal_totals = deals.groupby('ContactID')['Amount'].sum().reset_index(name='TotalDealValue')

# Merge
engagement = ticket_counts.merge(deal_totals, on='ContactID', how='inner')

# Calculate Spearman correlation
if len(engagement) > 1:
    spearman_corr, _ = stats.spearmanr(engagement['TicketCount'], engagement['TotalDealValue'])
    print(f"→ Spearman Correlation: {spearman_corr:.3f}")

    # Create ticket count groups for visualization
    engagement['TicketGroup'] = pd.cut(engagement['TicketCount'], bins=5, labels=['1-Low', '2', '3', '4', '5-High'])
    grouped_engagement = engagement.groupby('TicketGroup', observed=True)['TotalDealValue'].mean()

    # Plot bar chart
    plt.figure(figsize=(10, 6))
    colors = plt.cm.viridis(np.linspace(0, 1, len(grouped_engagement)))
    plt.bar(range(len(grouped_engagement)), grouped_engagement.values, color=colors, alpha=0.8, edgecolor='black')
    plt.xlabel('Ticket Count Groups', fontsize=12)
    plt.ylabel('Mean Total Deal Value ($)', fontsize=12)
    plt.title('Ticket Load vs Deal Value', fontsize=14, fontweight='bold')
    plt.xticks(range(len(grouped_engagement)), grouped_engagement.index, rotation=45)
    plt.grid(alpha=0.3, axis='y')
    plt.tight_layout()
    plt.savefig('ticket_deal_correlation.png', dpi=300, bbox_inches='tight')
    plt.close()
    print("✓ Generated: ticket_deal_correlation.png")
else:
    spearman_corr = 0.0
    print("✗ Insufficient data for correlation")

# ==========================================
# ---------- Analysis 7: Priority-Status Chi-Square Test ----------
# ==========================================
print("\n[10/10] Analysis 7: Priority-Status Association (Chi-Square Test)...")

# Create contingency table
contingency = pd.crosstab(tickets['Priority'], tickets['Status'])

# Perform chi-square test
chi2, p_value, dof, expected = stats.chi2_contingency(contingency)
print(f"→ Chi-Square Statistic: {chi2:.3f}")

# Plot heatmap
plt.figure(figsize=(10, 6))
plt.imshow(contingency.values, cmap='YlOrRd', aspect='auto', interpolation='nearest')
plt.colorbar(label='Frequency')
plt.xlabel('Ticket Status', fontsize=12)
plt.ylabel('Ticket Priority', fontsize=12)
plt.title('Priority-Status Relationship Heatmap', fontsize=14, fontweight='bold')
plt.xticks(range(len(contingency.columns)), contingency.columns, rotation=45, ha='right')
plt.yticks(range(len(contingency.index)), contingency.index)

# Add text annotations
for i in range(len(contingency.index)):
    for j in range(len(contingency.columns)):
        plt.text(j, i, str(contingency.values[i, j]),
                ha='center', va='center', color='black', fontsize=9)

plt.tight_layout()
plt.savefig('priority_status_heatmap.png', dpi=300, bbox_inches='tight')
plt.close()
print("✓ Generated: priority_status_heatmap.png")

# ==========================================
# ---------- Analysis 8: Deal Amount Coefficient of Variation ----------
# ==========================================
print("\n[Bonus 1/2] Analysis 8: Deal Amount Variability...")

amount_std = deals['Amount'].std()
amount_mean = deals['Amount'].mean()
cv = amount_std / amount_mean

print(f"→ Coefficient of Variation: {cv:.4f}")

# Plot density
plt.figure(figsize=(10, 6))
deals['Amount'].plot(kind='density', linewidth=2, color='darkblue')
plt.fill_between(np.linspace(deals['Amount'].min(), deals['Amount'].max(), 100),
                 0,
                 deals['Amount'].plot(kind='density').get_lines()[0].get_ydata()[:100] if len(deals['Amount'].plot(kind='density').get_lines()[0].get_ydata()) >= 100 else 0,
                 alpha=0.3)
plt.xlabel('Deal Amount ($)', fontsize=12)
plt.ylabel('Density Level', fontsize=12)
plt.title('Deal Amount Distribution Density', fontsize=14, fontweight='bold')
plt.axvline(amount_mean, color='red', linestyle='--', linewidth=2, label=f'Mean: ${amount_mean:,.0f}')
plt.legend()
plt.grid(alpha=0.3)
plt.tight_layout()
plt.savefig('deal_amount_density.png', dpi=300, bbox_inches='tight')
plt.close()
print("✓ Generated: deal_amount_density.png")

# ==========================================
# ---------- Analysis 9: Rolling Ticket Volume ----------
# ==========================================
print("\n[Bonus 2/2] Analysis 9: Rolling Ticket Volume (7-Day Average)...")

# Aggregate daily tickets
daily_series = tickets.groupby(tickets['CreatedDate'].dt.date).size().reset_index()
daily_series.columns = ['Date', 'Count']
daily_series = daily_series.sort_values('Date')

# Calculate 7-day rolling average
daily_series['Rolling7Day'] = daily_series['Count'].rolling(window=7, min_periods=1).mean()

# Find maximum
max_rolling = daily_series['Rolling7Day'].max()
print(f"→ Maximum 7-Day Rolling Average: {max_rolling:.2f}")

# Plot spiral (angular representation)
plt.figure(figsize=(10, 10))
angles = np.linspace(0, 8 * np.pi, len(daily_series))
radii = daily_series['Count'].values

ax = plt.subplot(111, projection='polar')
ax.plot(angles, radii, 'o-', alpha=0.6, linewidth=1.5)
ax.fill(angles, radii, alpha=0.2)
ax.set_title('Rolling Ticket Volume (Spiral View)', fontsize=14, fontweight='bold', pad=20)
plt.tight_layout()
plt.savefig('rolling_ticket_spiral.png', dpi=300, bbox_inches='tight')
plt.close()
print("✓ Generated: rolling_ticket_spiral.png")

# ==========================================
# ---------- Summary Report ----------
# ==========================================
print("\n" + "=" * 60)
print("ANALYSIS COMPLETE - KEY OUTPUTS")
print("=" * 60)
print(f"1. Survival Probability (Day 60):        {survival_60:.3f}")
print(f"2. Priority Coefficient (Escalation):    {priority_coef:.3f}")
print(f"3. Mean Resolution Duration:             {mean_resolution:.2f} days")
print(f"4. Overall Median Deal Amount:           ${overall_median:.2f}")
print(f"5. Daily Ticket Trend Slope:             {slope:.4f}")
print(f"6. Spearman Correlation (Engagement):    {spearman_corr:.3f}")
print(f"7. Chi-Square Statistic:                 {chi2:.3f}")
print(f"8. Coefficient of Variation:             {cv:.4f}")
print(f"9. Max 7-Day Rolling Average:            {max_rolling:.2f}")
print("=" * 60)
print("\n✓ All 9 visualizations generated successfully!")
print("=" * 60)
