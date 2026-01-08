"""
credit_risk.py
Credit risk models and counterparty analysis
"""

import numpy as np
from scipy import stats


def altman_z_score(working_capital, total_assets, retained_earnings,
                   ebit, market_equity, total_liabilities, sales):
    """
    Altman Z-Score for bankruptcy prediction.
    Higher score = lower default risk.
    """
    X1 = working_capital / total_assets
    X2 = retained_earnings / total_assets
    X3 = ebit / total_assets
    X4 = market_equity / total_liabilities
    X5 = sales / total_assets
    
    z = 1.2*X1 + 1.4*X2 + 3.3*X3 + 0.6*X4 + 0.999*X5
    
    if z > 2.99:
        zone = 'Safe'
    elif z > 1.81:
        zone = 'Grey'
    else:
        zone = 'Distress'
    
    return z, zone


def probability_of_default(z_score):
    """
    Estimate PD from Z-score.
    Uses logistic mapping (simplified).
    """
    # logistic transformation
    pd = 1 / (1 + np.exp(z_score - 2.5))
    return pd


def expected_loss(pd, lgd, ead):
    """
    Expected Credit Loss = PD x LGD x EAD
    
    pd: probability of default
    lgd: loss given default (recovery = 1 - lgd)
    ead: exposure at default
    """
    return pd * lgd * ead


def unexpected_loss(pd, lgd, ead, confidence=0.99):
    """
    Unexpected loss at a given confidence level.
    Based on Vasicek model for credit risk.
    """
    # asset correlation (Basel assumption)
    rho = 0.12
    
    # inverse normal
    norm_inv = stats.norm.ppf
    
    # conditional PD
    z = norm_inv(confidence)
    conditional_pd = stats.norm.cdf(
        (norm_inv(pd) + np.sqrt(rho) * z) / np.sqrt(1 - rho)
    )
    
    # UL
    ul = lgd * ead * (conditional_pd - pd)
    return ul


def credit_rating_to_pd(rating):
    """
    Map credit rating to approximate PD.
    Based on historical default rates.
    """
    pd_map = {
        'AAA': 0.0001,
        'AA': 0.0003,
        'A': 0.001,
        'BBB': 0.003,
        'BB': 0.01,
        'B': 0.04,
        'CCC': 0.15,
        'CC': 0.30,
        'C': 0.50,
        'D': 1.0
    }
    return pd_map.get(rating.upper(), 0.05)


def collateral_requirement(exposure, rating, haircut=0.0):
    """
    Calculate required collateral based on credit rating.
    """
    margin_rates = {
        'AAA': 0.02,
        'AA': 0.05,
        'A': 0.08,
        'BBB': 0.12,
        'BB': 0.18,
        'B': 0.25,
        'CCC': 0.35
    }
    
    rate = margin_rates.get(rating.upper(), 0.50)
    collateral = exposure * rate / (1 - haircut)
    return collateral


def cds_spread_estimate(pd, lgd=0.45, risk_free=0.02):
    """
    Estimate CDS spread from PD and LGD.
    Simplified hazard rate model.
    """
    # annual hazard rate approximation
    spread = pd * lgd * 10000  # in bps
    return spread


def credit_var(exposures, pds, lgds, correlation=0.20, n_sims=10000, confidence=0.99):
    """
    Portfolio credit VaR using Monte Carlo.
    Simulates correlated defaults.
    """
    n_counterparties = len(exposures)
    exposures = np.array(exposures)
    pds = np.array(pds)
    lgds = np.array(lgds)
    
    np.random.seed(42)
    
    losses = []
    for _ in range(n_sims):
        # systematic factor
        z = np.random.normal()
        
        # idiosyncratic factors
        eps = np.random.normal(size=n_counterparties)
        
        # asset values
        asset = np.sqrt(correlation) * z + np.sqrt(1 - correlation) * eps
        
        # default if asset < threshold
        thresholds = stats.norm.ppf(pds)
        defaults = asset < thresholds
        
        # calculate loss
        loss = np.sum(defaults * exposures * lgds)
        losses.append(loss)
    
    losses = np.array(losses)
    
    el = losses.mean()
    var = np.percentile(losses, confidence * 100)
    
    return {
        'expected_loss': el,
        'credit_var': var,
        'worst_case': losses.max()
    }


def netting_benefit(gross_exposures, netting_sets):
    """
    Calculate netting benefit for derivative exposures.
    Netting reduces credit exposure significantly.
    """
    # gross exposure
    gross = sum(abs(e) for e in gross_exposures)
    
    # net by netting set
    net_by_set = {}
    for i, nset in enumerate(netting_sets):
        if nset not in net_by_set:
            net_by_set[nset] = 0
        net_by_set[nset] += gross_exposures[i]
    
    # net exposure (only positive)
    net = sum(max(0, v) for v in net_by_set.values())
    
    benefit = 1 - (net / gross) if gross > 0 else 0
    
    return {
        'gross_exposure': gross,
        'net_exposure': net,
        'netting_benefit': benefit
    }
