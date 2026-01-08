"""
risk_metrics.py
VaR and Expected Shortfall calculations
"""

import numpy as np
from scipy import stats


def historical_var(returns, confidence=0.95, holding_period=1):
    """
    VaR using historical simulation.
    Sorts actual returns and finds the percentile cutoff.
    """
    scaled = returns * np.sqrt(holding_period)
    pct = (1 - confidence) * 100
    var = np.percentile(scaled, pct)
    return abs(var)


def parametric_var(returns, confidence=0.95, holding_period=1):
    """
    VaR assuming normal distribution.
    Uses mean and std to compute analytical VaR.
    """
    mu = returns.mean()
    sigma = returns.std()
    z = stats.norm.ppf(1 - confidence)
    var = -(mu * holding_period + z * sigma * np.sqrt(holding_period))
    return var


def monte_carlo_var(returns, n_sims=10000, confidence=0.95, holding_period=1):
    """
    VaR using Monte Carlo simulation.
    Generates random scenarios from fitted normal dist.
    """
    mu = returns.mean()
    sigma = returns.std()
    
    np.random.seed(42)
    sims = np.random.normal(mu * holding_period, 
                            sigma * np.sqrt(holding_period), 
                            n_sims)
    
    pct = (1 - confidence) * 100
    var = abs(np.percentile(sims, pct))
    return var


def expected_shortfall(returns, confidence=0.95):
    """
    Expected Shortfall (CVaR).
    Average loss when we exceed VaR threshold.
    """
    var = historical_var(returns, confidence)
    tail = returns[returns < -var]
    
    if len(tail) > 0:
        es = abs(tail.mean())
    else:
        es = var
    
    return es


def rolling_var(returns, window=60, confidence=0.95):
    """
    Calculate rolling VaR over a window.
    Good for tracking risk over time.
    """
    n = len(returns)
    var_series = np.full(n, np.nan)
    
    for i in range(window, n):
        chunk = returns[i-window:i]
        var_series[i] = historical_var(chunk, confidence)
    
    return var_series


def var_backtest(returns, var_series, confidence=0.95):
    """
    Backtest VaR predictions.
    Count how many times actual loss exceeded VaR.
    """
    # align arrays
    valid = ~np.isnan(var_series)
    actual = returns[valid]
    predicted = var_series[valid]
    
    # count exceptions (actual loss > predicted VaR)
    exceptions = np.sum(actual < -predicted)
    total = len(actual)
    
    exception_rate = exceptions / total
    expected_rate = 1 - confidence
    
    # traffic light test
    if exception_rate < expected_rate * 1.5:
        status = 'GREEN'
    elif exception_rate < expected_rate * 2:
        status = 'YELLOW'
    else:
        status = 'RED'
    
    return {
        'exceptions': exceptions,
        'total_days': total,
        'exception_rate': exception_rate,
        'expected_rate': expected_rate,
        'status': status
    }


def portfolio_var(weights, cov_matrix, confidence=0.95, portfolio_value=1e6):
    """
    Portfolio VaR using variance-covariance method.
    Accounts for correlations between assets.
    """
    weights = np.array(weights)
    port_var = np.sqrt(weights.T @ cov_matrix @ weights)
    z = stats.norm.ppf(confidence)
    var = port_var * z * portfolio_value
    return var
