# ==========================================
# Commodity Price Forecasting Analysis Script
# ==========================================
# Requirements: pandas, numpy, matplotlib, statsmodels, scikit-learn, scipy
# Input files: daily.xlsx, weekly.xlsx, monthly.xlsx
# Output files: arima_forecast.png, quantile_regression.png,
#               holt_winters_decomposition.png, kaplan_meier_survival.png,
#               cross_frequency_correlation.png
#
# PRICING STRATEGY RECOMMENDATION: Based on ARIMA forecasting showing short-term price
# trajectory and Holt-Winters revealing seasonal patterns with trend components, analysts
# should implement dynamic pricing with seasonal adjustments, maintaining price floors
# during low-season troughs while capitalizing on predictable peaks with premium pricing.
# ==========================================

import pandas as pd
import numpy as np
import matplotlib.pyplot as plt
from statsmodels.tsa.arima.model import ARIMA
from statsmodels.regression.quantile_regression import QuantReg
from statsmodels.tsa.holtwinters import ExponentialSmoothing
from scipy import stats
import warnings
warnings.filterwarnings('ignore')

# ---------- Kaplan-Meier Estimator Implementation ----------
class KaplanMeierEstimator:
    """Manual implementation of Kaplan-Meier survival analysis"""

    def __init__(self):
        self.survival_function_ = None
        self.timeline = None
        self.median_survival_time_ = None

    def fit(self, durations, event_observed):
        """Fit the Kaplan-Meier model"""
        data = pd.DataFrame({'time': durations, 'event': event_observed})
        data = data.sort_values('time')

        times = []
        survival_probs = []
        at_risk = len(data)
        survival_prob = 1.0

        for time in sorted(data['time'].unique()):
            time_data = data[data['time'] == time]
            events = time_data['event'].sum()

            if at_risk > 0:
                survival_prob *= (1 - events / at_risk)

            times.append(time)
            survival_probs.append(survival_prob)
            at_risk -= len(time_data)

        self.timeline = np.array(times)
        self.survival_function_ = np.array(survival_probs)

        # Calculate median survival time
        try:
            idx = np.where(self.survival_function_ <= 0.5)[0]
            if len(idx) > 0:
                self.median_survival_time_ = self.timeline[idx[0]]
            else:
                self.median_survival_time_ = self.timeline[-1]
        except:
            self.median_survival_time_ = np.median(durations)

        return self

    def plot(self, ci_show=False):
        """Plot the survival function"""
        plt.step(self.timeline, self.survival_function_, where='post', linewidth=2, label='KM estimate')

        # Calculate confidence intervals using Greenwood's formula
        if ci_show:
            # Simple approximation for CI
            var_estimate = self.survival_function_ * (1 - self.survival_function_) / len(self.timeline)
            std_estimate = np.sqrt(var_estimate)
            ci_upper = np.minimum(self.survival_function_ + 1.96 * std_estimate, 1.0)
            ci_lower = np.maximum(self.survival_function_ - 1.96 * std_estimate, 0.0)
            plt.fill_between(self.timeline, ci_lower, ci_upper, alpha=0.25, step='post')

# ---------- Load Data Files ----------
print("Loading data files...")
daily_df = pd.read_excel('daily.xlsx')
weekly_df = pd.read_excel('weekly.xlsx')
monthly_df = pd.read_excel('monthly.xlsx')

print(f"Daily dataset: {len(daily_df)} observations")
print(f"Weekly dataset: {len(weekly_df)} observations")
print(f"Monthly dataset: {len(monthly_df)} observations")

# ==========================================
# ANALYSIS 1: ARIMA FORECASTING ON DAILY DATA
# ==========================================
print("\n" + "="*60)
print("ANALYSIS 1: ARIMA(2,1,1) FORECASTING")
print("="*60)

# ---------- Sort and Split Data ----------
daily_df['Date'] = pd.to_datetime(daily_df['Date'])
daily_df = daily_df.sort_values('Date').reset_index(drop=True)

train_data = daily_df.iloc[:7200].copy()
validation_data = daily_df.iloc[7200:].copy()

print(f"Training observations: {len(train_data)}")
print(f"Validation observations: {len(validation_data)}")

# ---------- Fit ARIMA Model ----------
print("\nFitting ARIMA(2,1,1) model...")
arima_model = ARIMA(train_data['Price'], order=(2, 1, 1))
arima_fit = arima_model.fit()

# ---------- Generate Forecast ----------
forecast_steps = 30
forecast_result = arima_fit.forecast(steps=forecast_steps)
forecast_values = forecast_result

# Get fitted values
fitted_values = arima_fit.fittedvalues

# ---------- Calculate MAE ----------
overlap_periods = min(len(validation_data), forecast_steps)
mae = np.mean(np.abs(forecast_values[:overlap_periods].values - validation_data['Price'].iloc[:overlap_periods].values))
print(f"\nMean Absolute Error (MAE): {mae:.2f}")

# ---------- Plot ARIMA Results ----------
plt.figure(figsize=(14, 6))
plt.plot(range(len(train_data)), train_data['Price'], 'b-', label='Actual Prices', alpha=0.7)
plt.plot(range(len(fitted_values)), fitted_values, 'g-', label='Fitted Values', alpha=0.7)

forecast_start = len(train_data)
forecast_index = range(forecast_start, forecast_start + forecast_steps)
plt.plot(forecast_index, forecast_values, 'r-', label='Forecasted Prices', linewidth=2)

# Add confidence interval
forecast_ci = arima_fit.get_forecast(steps=forecast_steps).conf_int()
plt.fill_between(forecast_index, forecast_ci.iloc[:, 0], forecast_ci.iloc[:, 1],
                 color='red', alpha=0.2, label='95% Confidence Interval')

plt.xlabel('Time Period')
plt.ylabel('Price')
plt.title('ARIMA(2,1,1) Forecast - Daily Price Analysis')
plt.legend()
plt.grid(True, alpha=0.3)
plt.tight_layout()
plt.savefig('arima_forecast.png', dpi=300, bbox_inches='tight')
print("Saved: arima_forecast.png")

# ==========================================
# ANALYSIS 2: QUANTILE REGRESSION ON WEEKLY DATA
# ==========================================
print("\n" + "="*60)
print("ANALYSIS 2: 90TH PERCENTILE QUANTILE REGRESSION")
print("="*60)

# ---------- Sort and Create Time Index ----------
weekly_df['Week Start'] = pd.to_datetime(weekly_df['Week Start'])
weekly_df = weekly_df.sort_values('Week Start').reset_index(drop=True)
weekly_df['time_index'] = range(len(weekly_df))

print(f"Time index range: 0 to {len(weekly_df)-1}")

# ---------- Fit Quantile Regression ----------
print("\nFitting quantile regression (q=0.90)...")
X = weekly_df[['time_index']].values
y = weekly_df['Price'].values

quantreg_model = QuantReg(y, np.column_stack([np.ones(len(X)), X]))
quantreg_fit = quantreg_model.fit(q=0.90)

time_coefficient = quantreg_fit.params[1]
print(f"\nTime Coefficient (90th percentile): {time_coefficient:.4f}")

# ---------- Plot Quantile Regression ----------
plt.figure(figsize=(12, 6))
plt.scatter(weekly_df['time_index'], weekly_df['Price'], alpha=0.5, s=20, label='Actual Prices')

predicted_90th = quantreg_fit.predict(np.column_stack([np.ones(len(X)), X]))
plt.plot(weekly_df['time_index'], predicted_90th, 'r-', linewidth=2,
         label='90th Percentile Regression Line')

plt.xlabel('Time Index')
plt.ylabel('Price')
plt.title('Quantile Regression (90th Percentile) - Weekly Price Analysis')
plt.legend()
plt.grid(True, alpha=0.3)
plt.tight_layout()
plt.savefig('quantile_regression.png', dpi=300, bbox_inches='tight')
print("Saved: quantile_regression.png")

# ==========================================
# ANALYSIS 3: HOLT-WINTERS EXPONENTIAL SMOOTHING
# ==========================================
print("\n" + "="*60)
print("ANALYSIS 3: HOLT-WINTERS EXPONENTIAL SMOOTHING")
print("="*60)

# ---------- Aggregate Daily to Monthly ----------
daily_df['Year'] = daily_df['Date'].dt.year
daily_df['Month'] = daily_df['Date'].dt.month
monthly_agg = daily_df.groupby(['Year', 'Month'])['Price'].mean().reset_index()
monthly_agg['Date'] = pd.to_datetime(monthly_agg[['Year', 'Month']].assign(day=1))
monthly_agg = monthly_agg.sort_values('Date').reset_index(drop=True)

print(f"Monthly aggregated observations: {len(monthly_agg)}")

# ---------- Fit Holt-Winters Model ----------
print("\nFitting Holt-Winters model (seasonal_periods=12, additive)...")
hw_model = ExponentialSmoothing(monthly_agg['Price'],
                                seasonal_periods=12,
                                trend='add',
                                seasonal='add')
hw_fit = hw_model.fit()

alpha_parameter = hw_fit.params['smoothing_level']
print(f"\nAlpha (Level Smoothing Parameter): {alpha_parameter:.4f}")

# ---------- Extract Components ----------
fitted_values_hw = hw_fit.fittedvalues
trend_component = hw_fit.trend
seasonal_component = hw_fit.season
residuals = hw_fit.resid

# ---------- Plot Decomposition ----------
fig, axes = plt.subplots(4, 1, figsize=(14, 10))

# Original series
axes[0].plot(monthly_agg['Date'], monthly_agg['Price'], 'b-')
axes[0].set_ylabel('Observed')
axes[0].set_title('Holt-Winters Decomposition - Monthly Price Analysis')
axes[0].grid(True, alpha=0.3)

# Trend component
axes[1].plot(monthly_agg['Date'][:len(trend_component)], trend_component, 'g-')
axes[1].set_ylabel('Trend')
axes[1].grid(True, alpha=0.3)

# Seasonal component
axes[2].plot(monthly_agg['Date'][:len(seasonal_component)], seasonal_component, 'orange')
axes[2].set_ylabel('Seasonal')
axes[2].grid(True, alpha=0.3)

# Residuals
axes[3].plot(monthly_agg['Date'][:len(residuals)], residuals, 'r-')
axes[3].set_ylabel('Residual')
axes[3].set_xlabel('Time')
axes[3].grid(True, alpha=0.3)

plt.tight_layout()
plt.savefig('holt_winters_decomposition.png', dpi=300, bbox_inches='tight')
print("Saved: holt_winters_decomposition.png")

# ==========================================
# ANALYSIS 4: SURVIVAL ANALYSIS
# ==========================================
print("\n" + "="*60)
print("ANALYSIS 4: KAPLAN-MEIER SURVIVAL ANALYSIS")
print("="*60)

# ---------- Create Survival Variables ----------
threshold = 6.0
print(f"\nThreshold: ${threshold}")

# Create survival time and event indicator
survival_data = []
current_time = 0

for i in range(len(daily_df)):
    price = daily_df.iloc[i]['Price']

    if price <= threshold:
        current_time += 1
        event = 0  # Censored (still surviving)
    else:
        current_time += 1
        event = 1  # Event occurred (exceeded threshold)
        survival_data.append({'time': current_time, 'event': event})
        current_time = 0  # Reset counter

# Add final period if censored
if current_time > 0:
    survival_data.append({'time': current_time, 'event': 0})

survival_df = pd.DataFrame(survival_data)
print(f"Survival episodes: {len(survival_df)}")
print(f"Events (threshold crossings): {survival_df['event'].sum()}")

# ---------- Fit Kaplan-Meier Model ----------
print("\nFitting Kaplan-Meier survival curve...")
kmf = KaplanMeierEstimator()
kmf.fit(survival_df['time'].values, survival_df['event'].values)

median_survival = kmf.median_survival_time_
print(f"\nMedian Survival Time: {int(median_survival)} days")

# ---------- Plot Kaplan-Meier Curve ----------
plt.figure(figsize=(12, 6))
kmf.plot(ci_show=True)
plt.axhline(y=0.5, color='red', linestyle='--', linewidth=1, label='Median (50%)')
plt.xlabel('Survival Time (Days)')
plt.ylabel('Survival Probability')
plt.title(f'Kaplan-Meier Survival Curve - Time Until Price Exceeds ${threshold}')
plt.ylim(-0.05, 1.05)
plt.legend()
plt.grid(True, alpha=0.3)
plt.tight_layout()
plt.savefig('kaplan_meier_survival.png', dpi=300, bbox_inches='tight')
print("Saved: kaplan_meier_survival.png")

# ==========================================
# ANALYSIS 5: CROSS-FREQUENCY CORRELATION
# ==========================================
print("\n" + "="*60)
print("ANALYSIS 5: CROSS-FREQUENCY CORRELATION ANALYSIS")
print("="*60)

# ---------- Aggregate Daily to Monthly ----------
daily_monthly = daily_df.groupby(['Year', 'Month'])['Price'].mean().reset_index()
daily_monthly.columns = ['Year', 'Month', 'Daily_Derived_Price']

# ---------- Extract Monthly Data ----------
monthly_df['Month Start'] = pd.to_datetime(monthly_df['Month Start'])
monthly_df['Year'] = monthly_df['Month Start'].dt.year
monthly_df['Month'] = monthly_df['Month Start'].dt.month
monthly_df_extracted = monthly_df[['Year', 'Month', 'Price']].copy()
monthly_df_extracted.columns = ['Year', 'Month', 'Monthly_Price']

# ---------- Merge and Align ----------
merged = pd.merge(daily_monthly, monthly_df_extracted, on=['Year', 'Month'], how='inner')
print(f"Overlapping periods: {len(merged)}")

# ---------- Calculate Pearson Correlation ----------
correlation = merged['Daily_Derived_Price'].corr(merged['Monthly_Price'])
print(f"\nPearson Correlation Coefficient: {correlation:.4f}")

# ---------- Plot Scatter with Correlation ----------
plt.figure(figsize=(10, 8))
plt.scatter(merged['Daily_Derived_Price'], merged['Monthly_Price'], alpha=0.6, s=50)

# Add diagonal reference line
min_val = min(merged['Daily_Derived_Price'].min(), merged['Monthly_Price'].min())
max_val = max(merged['Daily_Derived_Price'].max(), merged['Monthly_Price'].max())
plt.plot([min_val, max_val], [min_val, max_val], 'r--', linewidth=2, label='y=x Reference')

plt.xlabel('Daily-Derived Monthly Price')
plt.ylabel('Original Monthly Price')
plt.title(f'Cross-Frequency Correlation Analysis\n(Pearson r = {correlation:.4f})')
plt.legend()
plt.grid(True, alpha=0.3)
plt.tight_layout()
plt.savefig('cross_frequency_correlation.png', dpi=300, bbox_inches='tight')
print("Saved: cross_frequency_correlation.png")

# ==========================================
# ANALYSIS 6: MULTI-FREQUENCY VOLATILITY COMPARISON
# ==========================================
print("\n" + "="*60)
print("ANALYSIS 6: MULTI-FREQUENCY VOLATILITY COMPARISON")
print("="*60)

# ---------- Calculate Coefficients of Variation ----------
daily_cv = (daily_df['Price'].std() / daily_df['Price'].mean()) * 100
weekly_cv = (weekly_df['Price'].std() / weekly_df['Price'].mean()) * 100
monthly_cv = (monthly_df['Price'].std() / monthly_df['Price'].mean()) * 100

print(f"\nDaily Coefficient of Variation: {daily_cv:.2f}%")
print(f"Weekly Coefficient of Variation: {weekly_cv:.2f}%")
print(f"Monthly Coefficient of Variation: {monthly_cv:.2f}%")

# ---------- Calculate Volatility Ratio ----------
volatility_ratio = daily_cv / monthly_cv
print(f"\nVolatility Ratio (Daily/Monthly): {volatility_ratio:.4f}")

# ==========================================
# ANALYSIS 7: PRICE RANGE RATIO BY YEAR
# ==========================================
print("\n" + "="*60)
print("ANALYSIS 7: PRICE RANGE RATIO ANALYSIS BY YEAR")
print("="*60)

# ---------- Extract Year and Calculate Range Ratios ----------
monthly_df['Year'] = monthly_df['Month Start'].dt.year
yearly_stats = monthly_df.groupby('Year')['Price'].agg(['min', 'max']).reset_index()
yearly_stats['range_ratio'] = yearly_stats['max'] / yearly_stats['min']

max_range_ratio = yearly_stats['range_ratio'].max()
max_year = yearly_stats.loc[yearly_stats['range_ratio'].idxmax(), 'Year']

print(f"\nYear with Maximum Range Ratio: {int(max_year)}")
print(f"Maximum Range Ratio: {max_range_ratio:.2f}")

# ==========================================
# ANALYSIS 8: PRICE MOMENTUM ANALYSIS
# ==========================================
print("\n" + "="*60)
print("ANALYSIS 8: PRICE MOMENTUM ANALYSIS")
print("="*60)

# ---------- Sort and Extract Periods ----------
weekly_df = weekly_df.sort_values('Week Start').reset_index(drop=True)

first_52_mean = weekly_df['Price'].iloc[:52].mean()
last_52_mean = weekly_df['Price'].iloc[-52:].mean()

price_momentum = last_52_mean - first_52_mean
print(f"\nHistorical Average (First 52 weeks): ${first_52_mean:.2f}")
print(f"Recent Average (Last 52 weeks): ${last_52_mean:.2f}")
print(f"Price Momentum Difference: ${price_momentum:.2f}")

# ==========================================
# KEY OUTPUTS SUMMARY
# ==========================================
print("\n" + "="*60)
print("KEY OUTPUTS SUMMARY")
print("="*60)
print(f"1. ARIMA MAE: {mae:.2f}")
print(f"2. Quantile Regression Time Coefficient: {time_coefficient:.4f}")
print(f"3. Holt-Winters Alpha Parameter: {alpha_parameter:.4f}")
print(f"4. Median Survival Time: {int(median_survival)} days")
print(f"5. Cross-Frequency Correlation: {correlation:.4f}")
print(f"6. Volatility Ratio: {volatility_ratio:.4f}")
print(f"7. Maximum Range Ratio: {max_range_ratio:.2f}")
print(f"8. Price Momentum: ${price_momentum:.2f}")
print("="*60)
print("\nAnalysis complete. All plots saved.")
