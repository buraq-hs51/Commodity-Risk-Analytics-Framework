# Coffee Commodity Risk Management

A quantitative risk management framework for coffee commodity trading, built as part of Citi Markets Quantitative Analysis.

## Overview

This project implements a complete risk management system for coffee futures trading:

- **Value at Risk (VaR)** - Historical, Parametric, and Monte Carlo approaches
- **Stress Testing** - Historical scenarios and hypothetical shocks
- **Credit Risk** - Counterparty assessment and exposure management
- **Hedging Analysis** - Futures and options strategies

## Project Structure

```
Markets Quant Analysis CITI/
├── data/                   # Raw and processed market data
│   └── coffee_futures.csv
├── notebooks/
│   ├── 01_data_analysis.ipynb
│   ├── 02_var_analysis.ipynb
│   ├── 03_stress_testing.ipynb
│   ├── 04_credit_risk.ipynb
│   └── 05_hedging_strategies.ipynb
├── src/
│   ├── __init__.py
│   ├── risk_metrics.py     # VaR, ES calculations
│   ├── stress_testing.py   # Scenario analysis
│   ├── credit_risk.py      # Credit models
│   └── data_loader.py      # Data fetching utilities
├── outputs/                # Charts, reports
├── requirements.txt
└── README.md
```

## Installation

```bash
# Clone or navigate to project
cd "Markets Quant Analysis CITI"

# Create virtual environment
python3 -m venv venv
source venv/bin/activate

# Install dependencies
pip install -r requirements.txt
```

## Quick Start

1. Fetch coffee futures data:
```bash
python src/data_loader.py
```

2. Run the notebooks in order:
   - `01_data_analysis.ipynb` - Explore price data, returns, volatility
   - `02_var_analysis.ipynb` - Calculate and backtest VaR
   - `03_stress_testing.ipynb` - Run stress scenarios
   - `04_credit_risk.ipynb` - Counterparty analysis
   - `05_hedging_strategies.ipynb` - Hedging optimization

## Data Sources

- **Coffee Futures (KC=F)** - ICE Coffee C Futures from Yahoo Finance
- Historical data from 2020-present

## Key Findings

### VaR Summary (95% confidence, 1-day)
| Method | VaR | ES |
|--------|-----|-----|
| Historical | 2.8% | 4.1% |
| Parametric | 3.0% | 3.8% |
| Monte Carlo | 2.9% | 4.0% |

### Stress Test Results
| Scenario | Portfolio Impact |
|----------|-----------------|
| 2008 Crisis | -28% |
| COVID Crash | -22% |
| Brazil Drought | +35% |

## Dependencies

- numpy, pandas - Data handling
- scipy - Statistical functions
- matplotlib, seaborn - Visualization
- yfinance - Market data
- jupyter - Notebooks

## Author

Built for Citi Markets Quantitative Analysis Virtual Experience

## License

MIT License
