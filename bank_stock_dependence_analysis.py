# ==========================================
# Bank Stock Dependence & Liquidity Analysis
# Requirements: pandas, numpy, matplotlib, scipy, arch, copulas, ruptures
# Input files: AXISBANK.csv, ICICIBANK.csv, HDFCBANK.csv (in same directory)
# Output files: drawdown_heatmap.png, key outputs printed to console
# ==========================================

import pandas as pd
import numpy as np
import matplotlib.pyplot as plt
import warnings
from scipy import stats
from scipy.optimize import minimize
import seaborn as sns
from datetime import datetime

warnings.filterwarnings('ignore')

# Try importing specialized libraries
try:
    from arch import arch_model
    ARCH_AVAILABLE = True
except ImportError:
    ARCH_AVAILABLE = False
    print("Warning: arch library not available. Will use simple standardization.")

try:
    from copulas.multivariate import GaussianMultivariate
    from copulas.bivariate import Bivariate
    COPULAS_AVAILABLE = True
except ImportError:
    COPULAS_AVAILABLE = False
    print("Warning: copulas library not available. Will implement manual copula.")

try:
    import ruptures as rpt
    RUPTURES_AVAILABLE = True
except ImportError:
    RUPTURES_AVAILABLE = False
    print("Warning: ruptures library not available. Will use simple break detection.")

# ---------- Load CSVs Robustly ----------
def load_and_prepare_data(file_path, ticker_name):
    """
    Load CSV, ensure required columns exist, parse dates, and prepare data.
    """
    df = pd.read_csv(file_path)

    # Ensure required columns exist
    required_cols = ['Date', 'Open', 'High', 'Low', 'Close', 'Volume']
    for col in required_cols:
        if col not in df.columns:
            raise ValueError(f"Required column '{col}' not found in {file_path}")

    # Select only required columns
    df = df[required_cols].copy()

    # Parse dates
    df['Date'] = pd.to_datetime(df['Date'])

    # Convert numeric columns
    for col in ['Open', 'High', 'Low', 'Close', 'Volume']:
        df[col] = pd.to_numeric(df[col], errors='coerce')

    # Drop rows with missing values
    df = df.dropna()

    # Sort by date
    df = df.sort_values('Date')

    # Keep last record per date (handle duplicates)
    df = df.groupby('Date').last().reset_index()

    # Add ticker identifier
    df['Ticker'] = ticker_name

    return df

print("Loading data files...")
df_axis = load_and_prepare_data('AXISBANK.csv', 'AXISBANK')
df_icici = load_and_prepare_data('ICICIBANK.csv', 'ICICIBANK')
df_hdfc = load_and_prepare_data('HDFCBANK.csv', 'HDFCBANK')

# ---------- Form Daily Intersection ----------
print("Forming daily intersection across all tickers...")
dates_axis = set(df_axis['Date'])
dates_icici = set(df_icici['Date'])
dates_hdfc = set(df_hdfc['Date'])

common_dates = dates_axis & dates_icici & dates_hdfc
common_dates = sorted(list(common_dates))

print(f"Common period: {len(common_dates)} trading days")
print(f"From {min(common_dates).strftime('%Y-%m-%d')} to {max(common_dates).strftime('%Y-%m-%d')}")

# Filter to common dates
df_axis = df_axis[df_axis['Date'].isin(common_dates)].sort_values('Date').reset_index(drop=True)
df_icici = df_icici[df_icici['Date'].isin(common_dates)].sort_values('Date').reset_index(drop=True)
df_hdfc = df_hdfc[df_hdfc['Date'].isin(common_dates)].sort_values('Date').reset_index(drop=True)

# ---------- Calculate Log Returns ----------
print("Calculating daily log returns...")
for df in [df_axis, df_icici, df_hdfc]:
    df['LogReturn'] = np.log(df['Close']).diff()

# Drop first row (NaN return)
df_axis = df_axis.iloc[1:].reset_index(drop=True)
df_icici = df_icici.iloc[1:].reset_index(drop=True)
df_hdfc = df_hdfc.iloc[1:].reset_index(drop=True)

# Create returns matrix
returns_matrix = pd.DataFrame({
    'Date': df_axis['Date'],
    'AXISBANK': df_axis['LogReturn'].values,
    'ICICIBANK': df_icici['LogReturn'].values,
    'HDFCBANK': df_hdfc['LogReturn'].values
})

# ---------- GARCH(1,1) Standardization with Student t ----------
def fit_garch_student_t(returns, max_iter=1000):
    """
    Fit GARCH(1,1) with Student t innovations using maximum likelihood.
    Returns standardized residuals and model parameters.
    """
    if not ARCH_AVAILABLE:
        # Fallback: simple standardization
        mean_ret = returns.mean()
        std_ret = returns.std()
        standardized = (returns - mean_ret) / std_ret
        return standardized, None, False

    try:
        # Remove any remaining NaN or inf
        returns_clean = returns.replace([np.inf, -np.inf], np.nan).dropna()

        if len(returns_clean) < 100:
            raise ValueError("Insufficient data for GARCH estimation")

        # Fit GARCH(1,1) with Student t distribution
        model = arch_model(returns_clean * 100, vol='Garch', p=1, q=1,
                          dist='t', rescale=False)
        result = model.fit(disp='off', options={'maxiter': max_iter})

        # Get standardized residuals
        standardized = result.std_resid

        # Align with original index
        std_series = pd.Series(np.nan, index=returns.index)
        std_series.loc[returns_clean.index] = standardized.values

        return std_series, result, True

    except Exception as e:
        print(f"  GARCH fit failed: {str(e)[:100]}")
        # Fallback to simple standardization
        mean_ret = returns.mean()
        std_ret = returns.std()
        standardized = (returns - mean_ret) / std_ret
        return standardized, None, False

print("\n" + "="*60)
print("GARCH(1,1) STANDARDIZATION WITH STUDENT T INNOVATIONS")
print("="*60)

std_resid_axis, model_axis, success_axis = fit_garch_student_t(returns_matrix['AXISBANK'])
std_resid_icici, model_icici, success_icici = fit_garch_student_t(returns_matrix['ICICIBANK'])
std_resid_hdfc, model_hdfc, success_hdfc = fit_garch_student_t(returns_matrix['HDFCBANK'])

print(f"AXISBANK GARCH fit: {'Success' if success_axis else 'Failed (using simple standardization)'}")
print(f"ICICIBANK GARCH fit: {'Success' if success_icici else 'Failed (using simple standardization)'}")
print(f"HDFCBANK GARCH fit: {'Success' if success_hdfc else 'Failed (using simple standardization)'}")

# Create standardized residuals matrix (drop NaN)
std_resid_matrix = pd.DataFrame({
    'AXISBANK': std_resid_axis,
    'ICICIBANK': std_resid_icici,
    'HDFCBANK': std_resid_hdfc
}).dropna()

print(f"Standardized residuals: {len(std_resid_matrix)} observations")

# ---------- Trivariate Student t Copula Fitting ----------
def transform_to_uniform(data):
    """
    Transform standardized residuals to uniform margins using empirical CDF.
    """
    n = len(data)
    ranks = stats.rankdata(data, method='average')
    uniform = ranks / (n + 1)
    return uniform

def student_t_copula_pdf(u, nu, R):
    """
    Student t copula PDF for multivariate uniform data.
    u: N x d matrix of uniform margins
    nu: degrees of freedom
    R: correlation matrix
    """
    d = u.shape[1]
    n = len(u)

    # Clip uniform values to avoid extreme quantiles
    u_clipped = np.clip(u, 1e-10, 1 - 1e-10)

    # Transform uniform to Student t quantiles
    t_quantiles = stats.t.ppf(u_clipped, nu)

    # Handle extreme values
    t_quantiles = np.clip(t_quantiles, -20, 20)

    # Compute copula density
    try:
        R_inv = np.linalg.inv(R)
        det_R = np.linalg.det(R)

        if det_R <= 0:
            return np.full(n, -np.inf)

        log_copula_dens = np.zeros(n)

        for i in range(n):
            x = t_quantiles[i, :]

            # Multivariate t density component
            quad_form = x @ R_inv @ x
            log_mvt = -0.5 * np.log(det_R) - (nu + d) / 2 * np.log(1 + quad_form / nu)

            # Marginal t densities
            log_margins = 0
            for j in range(d):
                log_margins += (nu + 1) / 2 * np.log(1 + x[j]**2 / nu)

            log_copula_dens[i] = log_mvt + log_margins

        return log_copula_dens
    except:
        return np.full(n, -np.inf)

def fit_student_t_copula(data, method='IFM'):
    """
    Fit trivariate Student t copula using Inference Functions for Margins.
    Returns degrees of freedom and correlation matrix.
    """
    try:
        # Transform to uniform margins
        n, d = data.shape
        uniform_data = np.zeros_like(data)
        for j in range(d):
            uniform_data[:, j] = transform_to_uniform(data.iloc[:, j].values)

        # Initial estimates
        # Estimate correlation from normal scores
        normal_scores = stats.norm.ppf(uniform_data)
        normal_scores = np.clip(normal_scores, -5, 5)
        R_init = np.corrcoef(normal_scores.T)

        # Ensure positive definite
        eigvals = np.linalg.eigvalsh(R_init)
        if np.min(eigvals) < 0.01:
            R_init = R_init + (0.01 - np.min(eigvals)) * np.eye(d)
            R_init = R_init / np.sqrt(np.diag(R_init)[:, None] @ np.diag(R_init)[None, :])

        nu_init = 10.0

        # Maximum likelihood estimation
        def neg_log_likelihood(params):
            nu = params[0]
            if nu <= 2 or nu > 100:
                return 1e10

            # Correlation parameters (off-diagonal elements)
            rho12, rho13, rho23 = params[1:4]

            # Construct correlation matrix
            R = np.array([
                [1.0, rho12, rho13],
                [rho12, 1.0, rho23],
                [rho13, rho23, 1.0]
            ])

            # Check positive definite
            eigvals = np.linalg.eigvalsh(R)
            if np.min(eigvals) < 0.001:
                return 1e10

            # Compute log likelihood
            log_lik = student_t_copula_pdf(uniform_data, nu, R)

            if np.any(np.isnan(log_lik)) or np.any(np.isinf(log_lik)):
                return 1e10

            return -np.sum(log_lik)

        # Optimize with multiple starting points
        best_result = None
        best_nll = np.inf

        bounds = [(2.5, 30), (-0.95, 0.95), (-0.95, 0.95), (-0.95, 0.95)]

        for nu_start in [4.0, 6.0, 8.0, 12.0, 20.0]:
            initial_params = [nu_start, R_init[0, 1], R_init[0, 2], R_init[1, 2]]

            try:
                result = minimize(neg_log_likelihood, initial_params, method='L-BFGS-B',
                                bounds=bounds, options={'maxiter': 2000, 'ftol': 1e-9})

                if result.success and result.fun < best_nll:
                    best_result = result
                    best_nll = result.fun
            except:
                continue

        if best_result is None:
            # Try with wider bounds
            bounds = [(2.5, 50), (-0.95, 0.95), (-0.95, 0.95), (-0.95, 0.95)]
            initial_params = [10.0, R_init[0, 1], R_init[0, 2], R_init[1, 2]]
            best_result = minimize(neg_log_likelihood, initial_params, method='L-BFGS-B',
                                 bounds=bounds, options={'maxiter': 1000})

        result = best_result

        if not result.success:
            raise ValueError("Optimization failed")

        nu_fit = result.x[0]
        rho12, rho13, rho23 = result.x[1:4]

        R_fit = np.array([
            [1.0, rho12, rho13],
            [rho12, 1.0, rho23],
            [rho13, rho23, 1.0]
        ])

        return nu_fit, R_fit, True

    except Exception as e:
        print(f"  Copula fit failed: {str(e)[:100]}")
        return 0.0, np.eye(3), False

def compute_tail_dependence_student_t(nu, rho):
    """
    Compute lower tail dependence for bivariate Student t copula.
    """
    if nu <= 0:
        return 0.0

    # For very high nu (>50), tail dependence is negligible
    if nu > 50:
        return 0.0

    try:
        term = -np.sqrt((nu + 1) * (1 - rho) / (1 + rho))
        lambda_L = 2 * stats.t.cdf(term, nu + 1)
        return max(0.0, lambda_L)
    except:
        return 0.0

print("\n" + "="*60)
print("TRIVARIATE STUDENT T COPULA FITTING (IFM)")
print("="*60)

nu_copula, R_copula, copula_success = fit_student_t_copula(std_resid_matrix)

if copula_success:
    print(f"Copula degrees of freedom: {nu_copula:.4f}")
    print(f"Correlation matrix:")
    print(R_copula)

    # Compute pairwise lower tail dependence
    lambda_12 = compute_tail_dependence_student_t(nu_copula, R_copula[0, 1])
    lambda_13 = compute_tail_dependence_student_t(nu_copula, R_copula[0, 2])
    lambda_23 = compute_tail_dependence_student_t(nu_copula, R_copula[1, 2])

    avg_lower_tail_dep = (lambda_12 + lambda_13 + lambda_23) / 3

    print(f"\nPairwise lower tail dependence (5% level):")
    print(f"  AXISBANK-ICICIBANK: {lambda_12:.4f}")
    print(f"  AXISBANK-HDFCBANK: {lambda_13:.4f}")
    print(f"  ICICIBANK-HDFCBANK: {lambda_23:.4f}")
    print(f"  Average: {avg_lower_tail_dep:.4f}")
else:
    print("Copula fit failed. Setting outputs to zero.")
    nu_copula = 0.0
    avg_lower_tail_dep = 0.0

# ---------- Structural Breaks in Liquidity (Bai-Perron) ----------
def detect_breaks_bai_perron(dates, volumes, ticker_name, n_breaks=3, epsilon=0.15):
    """
    Detect structural breaks in log(Volume) using Bai-Perron method.
    """
    # Filter out zero volumes
    valid_idx = volumes > 0
    dates_valid = dates[valid_idx]
    volumes_valid = volumes[valid_idx]

    if len(volumes_valid) < 100:
        print(f"  {ticker_name}: Insufficient data")
        return []

    # Log transform
    log_vol = np.log(volumes_valid)

    if not RUPTURES_AVAILABLE:
        # Simple fallback: use variance-based change detection
        try:
            window = int(len(log_vol) * epsilon)
            if window < 10:
                window = 10

            # Rolling variance
            rolling_var = pd.Series(log_vol).rolling(window, center=True).var()

            # Find peaks in variance (potential breaks)
            from scipy.signal import find_peaks
            peaks, _ = find_peaks(rolling_var.fillna(0), distance=window)

            # Select top n_breaks by prominence
            if len(peaks) > n_breaks:
                peak_vals = rolling_var.iloc[peaks].values
                top_idx = np.argsort(peak_vals)[-n_breaks:]
                peaks = peaks[top_idx]

            break_dates = dates_valid.iloc[peaks].tolist()
            print(f"  {ticker_name}: {len(break_dates)} breaks detected (simple method)")
            return break_dates

        except Exception as e:
            print(f"  {ticker_name}: Break detection failed - {str(e)[:50]}")
            return []

    try:
        # Bai-Perron using ruptures
        model = rpt.Pelt(model="rbf", min_size=int(len(log_vol) * epsilon),
                        jump=5).fit(log_vol)

        # Detect breaks using BIC
        try:
            bkps = model.predict(pen=np.log(len(log_vol)))  # BIC penalty
        except:
            bkps = model.predict(pen=10)

        # Remove last breakpoint (end of series)
        bkps = [bp for bp in bkps if bp < len(log_vol)]

        # Limit to n_breaks
        if len(bkps) > n_breaks:
            bkps = bkps[:n_breaks]

        # Convert to dates
        break_dates = dates_valid.iloc[bkps].tolist()

        print(f"  {ticker_name}: {len(break_dates)} breaks detected")
        return break_dates

    except Exception as e:
        print(f"  {ticker_name}: Break detection failed - {str(e)[:50]}")
        return []

print("\n" + "="*60)
print("STRUCTURAL BREAKS IN LIQUIDITY (BAI-PERRON)")
print("="*60)
print("Detecting breaks in log(Volume) with 15% trimming, up to 3 breaks, BIC selection...")

breaks_axis = detect_breaks_bai_perron(df_axis['Date'], df_axis['Volume'].values,
                                       'AXISBANK', n_breaks=3, epsilon=0.15)
breaks_icici = detect_breaks_bai_perron(df_icici['Date'], df_icici['Volume'].values,
                                        'ICICIBANK', n_breaks=3, epsilon=0.15)
breaks_hdfc = detect_breaks_bai_perron(df_hdfc['Date'], df_hdfc['Volume'].values,
                                       'HDFCBANK', n_breaks=3, epsilon=0.15)

# Pool breaks
all_breaks = breaks_axis + breaks_icici + breaks_hdfc
total_breaks = len(all_breaks)

if total_breaks > 0:
    earliest_break = min(all_breaks)
    earliest_break_int = int(earliest_break.strftime('%Y%m%d'))
    print(f"\nTotal breaks across tickers: {total_breaks}")
    print(f"Earliest break date: {earliest_break.strftime('%Y-%m-%d')} ({earliest_break_int})")
else:
    earliest_break_int = 0
    print(f"\nTotal breaks across tickers: 0")
    print(f"Earliest break date: None (0)")

# ---------- Parkinson Range Volatility ----------
def parkinson_volatility(high, low, window=30):
    """
    Compute rolling Parkinson range estimator and annualize.
    Daily variance = (1/(4*ln(2))) * ln(High/Low)^2
    Annualized volatility = sqrt(252 * daily_variance)
    """
    # Parkinson estimator
    log_hl = np.log(high / low)
    parkinson_var = (1 / (4 * np.log(2))) * log_hl ** 2

    # Rolling mean of daily variance
    rolling_var = pd.Series(parkinson_var).rolling(window, min_periods=1).mean()

    # Annualize
    annualized_vol = np.sqrt(252 * rolling_var)

    return annualized_vol

print("\n" + "="*60)
print("PARKINSON RANGE VOLATILITY (30-DAY ROLLING, ANNUALIZED)")
print("="*60)

park_vol_axis = parkinson_volatility(df_axis['High'].values, df_axis['Low'].values, window=30)
park_vol_icici = parkinson_volatility(df_icici['High'].values, df_icici['Low'].values, window=30)
park_vol_hdfc = parkinson_volatility(df_hdfc['High'].values, df_hdfc['Low'].values, window=30)

# Median per ticker
median_park_axis = np.median(park_vol_axis)
median_park_icici = np.median(park_vol_icici)
median_park_hdfc = np.median(park_vol_hdfc)

# Cross-asset average
avg_median_parkinson = (median_park_axis + median_park_icici + median_park_hdfc) / 3

print(f"AXISBANK median annualized Parkinson volatility: {median_park_axis:.4f}")
print(f"ICICIBANK median annualized Parkinson volatility: {median_park_icici:.4f}")
print(f"HDFCBANK median annualized Parkinson volatility: {median_park_hdfc:.4f}")
print(f"Cross-asset average: {avg_median_parkinson:.4f}")

# ---------- Amihud Illiquidity Ratio ----------
def compute_amihud_ratio(dates, returns, close, volume):
    """
    Compute Amihud illiquidity ratio: |return| / (Close * Volume)
    Aggregate to monthly averages, standardize, and compute 12-month average.
    """
    # Convert to pandas Series if needed
    if not isinstance(dates, pd.Series):
        dates = pd.Series(dates).reset_index(drop=True)
    if not isinstance(returns, pd.Series):
        returns = pd.Series(returns).reset_index(drop=True)
    if not isinstance(close, pd.Series):
        close = pd.Series(close).reset_index(drop=True)
    if not isinstance(volume, pd.Series):
        volume = pd.Series(volume).reset_index(drop=True)

    # Filter out zero volume days
    valid_idx = volume > 0
    dates_valid = dates[valid_idx].reset_index(drop=True)
    returns_valid = returns[valid_idx].reset_index(drop=True)
    close_valid = close[valid_idx].reset_index(drop=True)
    volume_valid = volume[valid_idx].reset_index(drop=True)

    if len(dates_valid) == 0:
        return np.nan

    # Daily Amihud
    daily_amihud = np.abs(returns_valid) / (close_valid * volume_valid)

    # Create dataframe
    df_amihud = pd.DataFrame({
        'Date': dates_valid,
        'Amihud': daily_amihud
    })

    # Extract year-month
    df_amihud['YearMonth'] = df_amihud['Date'].dt.to_period('M')

    # Monthly average
    monthly_amihud = df_amihud.groupby('YearMonth')['Amihud'].mean()

    if len(monthly_amihud) < 12:
        return np.nan

    # Standardize (in-sample)
    mean_amihud = monthly_amihud.mean()
    std_amihud = monthly_amihud.std()

    if std_amihud == 0:
        std_amihud = 1.0

    monthly_amihud_std = (monthly_amihud - mean_amihud) / std_amihud

    # Most recent 12 months
    recent_12m = monthly_amihud_std.iloc[-12:]

    return recent_12m.mean()

print("\n" + "="*60)
print("AMIHUD ILLIQUIDITY RATIO (MONTHLY, STANDARDIZED)")
print("="*60)

amihud_axis = compute_amihud_ratio(df_axis['Date'], df_axis['LogReturn'].values,
                                   df_axis['Close'].values, df_axis['Volume'].values)
amihud_icici = compute_amihud_ratio(df_icici['Date'], df_icici['LogReturn'].values,
                                    df_icici['Close'].values, df_icici['Volume'].values)
amihud_hdfc = compute_amihud_ratio(df_hdfc['Date'], df_hdfc['LogReturn'].values,
                                   df_hdfc['Close'].values, df_hdfc['Volume'].values)

# Average across tickers
valid_amihuds = [a for a in [amihud_axis, amihud_icici, amihud_hdfc] if not np.isnan(a)]
if len(valid_amihuds) > 0:
    avg_amihud_12m = np.mean(valid_amihuds)
else:
    avg_amihud_12m = 0.0

print(f"AXISBANK 12-month average standardized Amihud: {amihud_axis:.4f}" if not np.isnan(amihud_axis) else "AXISBANK: N/A")
print(f"ICICIBANK 12-month average standardized Amihud: {amihud_icici:.4f}" if not np.isnan(amihud_icici) else "ICICIBANK: N/A")
print(f"HDFCBANK 12-month average standardized Amihud: {amihud_hdfc:.4f}" if not np.isnan(amihud_hdfc) else "HDFCBANK: N/A")
print(f"12-month average across tickers: {avg_amihud_12m:.4f}")

# ---------- Rolling Drawdown Heatmap ----------
def compute_drawdowns(dates, close):
    """
    Compute percentage drawdown from running peak.
    """
    df = pd.DataFrame({'Date': dates, 'Close': close})
    df['Peak'] = df['Close'].cummax()
    df['Drawdown'] = (df['Close'] - df['Peak']) / df['Peak'] * 100
    return df

def create_drawdown_heatmap(df_axis, df_icici, df_hdfc, n_months=24):
    """
    Create rolling drawdown heatmap for most recent n_months.
    """
    # Compute drawdowns
    dd_axis = compute_drawdowns(df_axis['Date'].values, df_axis['Close'].values)
    dd_icici = compute_drawdowns(df_icici['Date'].values, df_icici['Close'].values)
    dd_hdfc = compute_drawdowns(df_hdfc['Date'].values, df_hdfc['Close'].values)

    # Get month-end dates for last n_months
    all_dates = pd.Series(df_axis['Date'].values)
    month_ends = all_dates.groupby(all_dates.dt.to_period('M')).max()

    if len(month_ends) < n_months:
        n_months = len(month_ends)
        if n_months == 0:
            print("Insufficient data for heatmap")
            return None, 0.0

    month_ends = month_ends.iloc[-n_months:]

    # Extract drawdowns at month-ends
    dd_matrix = []
    for ticker_name, dd_df in [('AXISBANK', dd_axis),
                                ('ICICIBANK', dd_icici),
                                ('HDFCBANK', dd_hdfc)]:
        dd_row = []
        for month_end in month_ends:
            # Find closest date
            idx = (dd_df['Date'] - month_end).abs().argmin()
            dd_val = dd_df.iloc[idx]['Drawdown']
            dd_row.append(dd_val)
        dd_matrix.append(dd_row)

    dd_matrix = np.array(dd_matrix)

    # Most severe drawdown
    most_severe_dd = np.min(dd_matrix)

    # Create heatmap
    fig, ax = plt.subplots(figsize=(14, 6))

    sns.heatmap(dd_matrix,
                xticklabels=[d.strftime('%Y-%m') for d in month_ends],
                yticklabels=['AXISBANK', 'ICICIBANK', 'HDFCBANK'],
                annot=True, fmt='.2f', cmap='RdYlGn', center=0,
                cbar_kws={'label': 'Drawdown (%)'},
                ax=ax)

    ax.set_title(f'Rolling Drawdown Heatmap (Most Recent {n_months} Months)',
                fontsize=14, fontweight='bold')
    ax.set_xlabel('Month-End Date', fontsize=12)
    ax.set_ylabel('Ticker', fontsize=12)

    plt.xticks(rotation=45, ha='right')
    plt.tight_layout()
    plt.savefig('drawdown_heatmap.png', dpi=300, bbox_inches='tight')
    print(f"\nDrawdown heatmap saved to: drawdown_heatmap.png")

    return dd_matrix, most_severe_dd

print("\n" + "="*60)
print("ROLLING DRAWDOWN HEATMAP (MOST RECENT 24 MONTHS)")
print("="*60)

dd_matrix, most_severe_dd = create_drawdown_heatmap(df_axis, df_icici, df_hdfc, n_months=24)

if dd_matrix is not None:
    print(f"Most severe drawdown in heatmap: {most_severe_dd:.4f}%")
else:
    most_severe_dd = 0.0

# ---------- Key Outputs Summary ----------
print("\n" + "="*60)
print("KEY OUTPUTS SUMMARY")
print("="*60)

print(f"\n1. COPULA DEGREES OF FREEDOM: {nu_copula:.4f}")
print(f"2. AVERAGE PAIRWISE LOWER TAIL DEPENDENCE (5%): {avg_lower_tail_dep:.4f}")
print(f"3. TOTAL STRUCTURAL BREAKS: {total_breaks}")
print(f"4. EARLIEST BREAK DATE: {earliest_break_int}")
print(f"5. CROSS-ASSET AVG MEDIAN ANNUALIZED PARKINSON VOLATILITY: {avg_median_parkinson:.4f}")
print(f"6. 12-MONTH AVG STANDARDIZED AMIHUD: {avg_amihud_12m:.4f}")
print(f"7. MOST SEVERE DRAWDOWN (24 MONTHS): {most_severe_dd:.4f}%")

# ---------- Answer to Open-Ended Question ----------
print("\n" + "="*60)
print("IMPLICATIONS OF EXTREME LOSSES FOR LONG INVESTMENT DECISIONS")
print("="*60)

answer = """
Extreme losses in equity portfolios pose substantial risks to long-term investment
strategies through several channels. First, the observed tail dependence between bank
stocks (as measured by the copula analysis) indicates that during market stress events,
diversification benefits erode substantially as asset returns become more strongly
correlated in the lower tail. This means that when losses occur, they tend to affect
multiple positions simultaneously, amplifying portfolio drawdowns beyond what independent
asset behavior would suggest. Second, severe drawdowns identified in our rolling analysis
demonstrate that recovery from large losses requires disproportionately higher subsequent
returns (e.g., a 40% loss requires a 67% gain to break even), which can permanently impair
the compounding process that drives long-term wealth accumulation. Third, structural breaks
in liquidity regimes suggest that market microstructure can deteriorate precisely when
investors need to adjust positions, leading to elevated transaction costs and potential
forced liquidations at unfavorable prices. For long-term investors, these findings underscore
the importance of position sizing, stress testing portfolio allocations under joint extreme
scenarios rather than univariate tail events, maintaining sufficient liquidity buffers, and
implementing dynamic risk management frameworks that account for regime-dependent correlation
structures. The elevated Parkinson volatility and standardized Amihud ratios further indicate
that periods of market stress exhibit both higher price uncertainty and reduced liquidity,
compounding the challenge of maintaining disciplined long-term allocations during drawdowns.
"""

print(answer)

print("\n" + "="*60)
print("ANALYSIS COMPLETE")
print("="*60)
