"""
data_loader.py
Fetch and prepare market data
"""

import os
import pandas as pd
import numpy as np

try:
    import yfinance as yf
    HAS_YFINANCE = True
except ImportError:
    HAS_YFINANCE = False


def get_coffee_futures(start='2020-01-01', end=None, save_path=None):
    """
    Download coffee futures data from Yahoo Finance.
    KC=F is the ICE Coffee C Futures contract.
    """
    if not HAS_YFINANCE:
        raise ImportError("yfinance not installed. Run: pip install yfinance")
    
    ticker = 'KC=F'
    
    print(f"Fetching {ticker} from {start}...")
    df = yf.download(ticker, start=start, end=end, progress=False)
    
    if df.empty:
        print("Warning: No data returned")
        return None
    
    # flatten multi-index columns if present
    if isinstance(df.columns, pd.MultiIndex):
        df.columns = df.columns.get_level_values(0)
    
    # add returns
    df['returns'] = df['Close'].pct_change()
    df['log_returns'] = np.log(df['Close'] / df['Close'].shift(1))
    
    # rolling volatility
    df['volatility_20d'] = df['returns'].rolling(20).std() * np.sqrt(252)
    df['volatility_60d'] = df['returns'].rolling(60).std() * np.sqrt(252)
    
    if save_path:
        df.to_csv(save_path)
        print(f"Saved to {save_path}")
    
    return df


def get_multiple_commodities(tickers=None, start='2020-01-01'):
    """
    Fetch multiple commodity futures for correlation analysis.
    """
    if not HAS_YFINANCE:
        raise ImportError("yfinance not installed")
    
    if tickers is None:
        tickers = {
            'coffee': 'KC=F',
            'sugar': 'SB=F',
            'cocoa': 'CC=F',
            'cotton': 'CT=F'
        }
    
    data = {}
    for name, ticker in tickers.items():
        print(f"Fetching {name} ({ticker})...")
        df = yf.download(ticker, start=start, progress=False)
        if not df.empty:
            if isinstance(df.columns, pd.MultiIndex):
                df.columns = df.columns.get_level_values(0)
            data[name] = df['Close']
    
    combined = pd.DataFrame(data)
    combined = combined.dropna()
    
    return combined


def load_csv(filepath):
    """
    Load data from CSV file.
    """
    df = pd.read_csv(filepath, index_col=0, parse_dates=True)
    
    # ensure returns exist
    if 'returns' not in df.columns and 'Close' in df.columns:
        df['returns'] = df['Close'].pct_change()
    
    return df


def calculate_returns(prices, method='simple'):
    """
    Calculate returns from price series.
    """
    if method == 'simple':
        return prices.pct_change()
    elif method == 'log':
        return np.log(prices / prices.shift(1))
    else:
        raise ValueError(f"Unknown method: {method}")


def clean_data(df, drop_na=True, fill_method=None):
    """
    Clean dataframe - handle missing values.
    """
    if fill_method:
        df = df.fillna(method=fill_method)
    
    if drop_na:
        df = df.dropna()
    
    return df


def resample_to_weekly(df, price_col='Close'):
    """
    Resample daily data to weekly.
    """
    weekly = df.resample('W').agg({
        'Open': 'first',
        'High': 'max',
        'Low': 'min',
        'Close': 'last',
        'Volume': 'sum'
    })
    
    weekly['returns'] = weekly['Close'].pct_change()
    
    return weekly


def get_risk_free_rate(start='2020-01-01'):
    """
    Get 10Y Treasury rate as risk-free proxy.
    """
    if not HAS_YFINANCE:
        return None
    
    df = yf.download('^TNX', start=start, progress=False)
    
    if isinstance(df.columns, pd.MultiIndex):
        df.columns = df.columns.get_level_values(0)
    
    # convert to decimal
    df['rate'] = df['Close'] / 100
    
    return df['rate']


if __name__ == '__main__':
    # run as script to download data
    import os
    
    script_dir = os.path.dirname(os.path.abspath(__file__))
    data_dir = os.path.join(script_dir, '..', 'data')
    
    os.makedirs(data_dir, exist_ok=True)
    
    save_path = os.path.join(data_dir, 'coffee_futures.csv')
    
    df = get_coffee_futures(save_path=save_path)
    
    if df is not None:
        print(f"\nData shape: {df.shape}")
        print(f"Date range: {df.index[0]} to {df.index[-1]}")
        print(f"\nSample data:")
        print(df.tail())
