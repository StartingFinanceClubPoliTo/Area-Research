# Markowitz Portfolio Optimization - Supporting Code

This folder contains the Python code supporting the research article "Portfolio Optimization: From Markowitz Theory to Practical Implementation".

The script reproduces the empirical exercises used in the article:

1. classical multi-asset Markowitz optimization over the longest available sample;
2. common-sample comparison with and without Bitcoin;
3. fixed out-of-sample test;
4. rolling out-of-sample Markowitz test;
5. portfolio weight, turnover, and cumulative-wealth plots.

## Asset Universe

The classical portfolio uses five liquid ETFs:

- `SPY` for U.S. equities;
- `VEA` for developed markets outside the United States;
- `EEM` for emerging market equities;
- `IEF` for intermediate-term U.S. Treasury bonds;
- `GLD` for gold.

The expanded comparison also includes `BTC-USD`. The risk-free-rate proxy is `^IRX`, the 13-week Treasury Bill yield downloaded from Yahoo Finance.

## Files

| File | Role |
| --- | --- |
| `portfolio_optimization_markowitz.py` | Main script for data download, optimization, backtests, plots, and CSV exports. |
| `requirements.txt` | Python dependencies. |
| `DATA_NOTICE.md` | Data-source and raw-data notice. |
| `GITHUB_REFERENCE.txt` | Suggested citation text for the article reference section. |
| `.gitignore` | Local Python and generated-output exclusions for this project. |

When the script is executed, it creates a `figures/` directory with generated figures and CSV summary tables.

## Installation

Create and activate a virtual environment:

```bash
python -m venv .venv
```

On Windows:

```bash
.venv\Scripts\activate
```

On macOS/Linux:

```bash
source .venv/bin/activate
```

Install the required packages:

```bash
pip install -r requirements.txt
```

## How To Run

From this folder:

```bash
python portfolio_optimization_markowitz.py
```

The script downloads market data from Yahoo Finance, computes returns, estimates annualized return and covariance inputs, solves the optimization problems, and exports plots and tables to `figures/`.

## Methodology

The code implements a long-only, fully invested Markowitz framework:

```text
sum(weights) = 1
weights_i >= 0
```

The efficient frontier is computed numerically with Sequential Least Squares Programming (`SLSQP`). Random long-only portfolios are generated with Dirichlet-distributed weights.

The fixed out-of-sample test estimates optimized weights on a training period and applies those fixed target weights to the testing period. The rolling out-of-sample test re-estimates expected returns, covariances, and optimized weights at monthly rebalance dates using a rolling lookback window.

## Main Outputs

The generated `figures/` directory can include:

- normalized price charts;
- correlation matrices;
- random long-only portfolio clouds;
- optimized efficient frontiers;
- minimum-variance and maximum-Sharpe portfolio tables;
- fixed out-of-sample cumulative-wealth paths;
- rolling out-of-sample cumulative-wealth paths;
- rolling portfolio weight histories;
- rolling turnover plots;
- CSV files with portfolio statistics and optimized weights.

## Reproducibility Notes

- The code sets deterministic seeds for the random portfolio simulations.
- Market data are downloaded at runtime through `yfinance`; raw price files are not stored in this repository.
- Results can change if Yahoo Finance revises historical data or if the script is run with a different end date or package version.
- The analysis is for educational and research purposes only and does not provide financial advice.
