"""
stress_testing.py
Scenario analysis and stress testing tools
"""

import numpy as np
import pandas as pd


# historical crisis periods (approximate return shocks)
HISTORICAL_SCENARIOS = {
    '2008_crisis': {
        'description': 'Global Financial Crisis',
        'coffee': -0.35,
        'equities': -0.50,
        'rates': -0.02,
        'fx_em': -0.25
    },
    '2020_covid': {
        'description': 'COVID-19 Market Crash',
        'coffee': -0.28,
        'equities': -0.34,
        'rates': -0.015,
        'fx_em': -0.18
    },
    '2022_commodity_spike': {
        'description': 'Post-COVID Commodity Rally',
        'coffee': 0.45,
        'equities': -0.15,
        'rates': 0.03,
        'fx_em': -0.10
    },
    'brazil_drought': {
        'description': 'Severe Brazil Drought',
        'coffee': 0.40,
        'equities': 0.0,
        'rates': 0.0,
        'fx_em': -0.08
    },
    'frost_event': {
        'description': 'Major Frost in Coffee Regions',
        'coffee': 0.55,
        'equities': 0.0,
        'rates': 0.0,
        'fx_em': -0.05
    }
}


def run_stress_test(positions, scenario):
    """
    Apply a scenario to portfolio positions.
    Returns P&L breakdown by asset.
    """
    pnl = {}
    total = 0
    
    for asset, size in positions.items():
        if asset in scenario:
            shock = scenario[asset]
            asset_pnl = size * shock
            pnl[asset] = asset_pnl
            total += asset_pnl
    
    pnl['total'] = total
    return pnl


def run_all_scenarios(positions, scenarios=None):
    """
    Run all predefined stress scenarios.
    Returns DataFrame with results.
    """
    if scenarios is None:
        scenarios = HISTORICAL_SCENARIOS
    
    results = []
    for name, scenario in scenarios.items():
        pnl = run_stress_test(positions, scenario)
        pnl['scenario'] = name
        pnl['description'] = scenario.get('description', '')
        results.append(pnl)
    
    return pd.DataFrame(results)


def sensitivity_analysis(positions, asset, shock_range):
    """
    Run sensitivity for one asset across a range of shocks.
    Useful for seeing how P&L changes with price moves.
    """
    results = []
    base_position = positions.get(asset, 0)
    
    for shock in shock_range:
        pnl = base_position * shock
        results.append({
            'shock': shock,
            'pnl': pnl
        })
    
    return pd.DataFrame(results)


def create_custom_scenario(shocks_dict, name='custom'):
    """
    Build a custom stress scenario from dict.
    """
    scenario = shocks_dict.copy()
    scenario['description'] = name
    return scenario


def reverse_stress_test(positions, target_loss, asset='coffee'):
    """
    Find what price move would cause a specific loss.
    Answers: "what would break us?"
    """
    position = positions.get(asset, 0)
    
    if position == 0:
        return None
    
    required_shock = target_loss / position
    return required_shock


def correlation_stress(returns_df, shock_correlation=0.9):
    """
    Stress test assuming correlations spike to near 1.
    Common during crises when diversification fails.
    """
    n_assets = returns_df.shape[1]
    
    # stressed correlation matrix
    stressed_corr = np.full((n_assets, n_assets), shock_correlation)
    np.fill_diagonal(stressed_corr, 1.0)
    
    # get volatilities
    vols = returns_df.std().values
    
    # stressed covariance
    stressed_cov = np.outer(vols, vols) * stressed_corr
    
    return stressed_cov


def tail_dependence_check(returns_df, threshold_pct=5):
    """
    Check if assets move together in the tails.
    Important for understanding crisis behavior.
    """
    cols = returns_df.columns
    n = len(cols)
    
    # lower tail threshold
    thresholds = returns_df.quantile(threshold_pct / 100)
    
    results = {}
    for i in range(n):
        for j in range(i+1, n):
            col1, col2 = cols[i], cols[j]
            
            # both in lower tail
            both_low = (returns_df[col1] < thresholds[col1]) & \
                       (returns_df[col2] < thresholds[col2])
            
            # lower tail dependence
            joint_prob = both_low.mean()
            marginal = (threshold_pct / 100) ** 2
            
            results[f'{col1}_{col2}'] = joint_prob / marginal if marginal > 0 else 0
    
    return results
