# Markoviz

This folder contains portfolio optimization research material for Starting Finance Club PoliTo.

## Articles

| Article | Topic | Materials |
| --- | --- | --- |
| [Markowitz Portfolio Optimization](./markowitz-portfolio-optimization-github/) | Mean-variance portfolio optimization, efficient frontiers, and out-of-sample portfolio tests | Python script, requirements, data notice, and GitHub reference text |

## Current Project

The uploaded material supports a research article on Markowitz portfolio optimization. The code studies a long-only, fully invested portfolio across a liquid ETF universe:

- `SPY` for U.S. equities;
- `VEA` for developed markets outside the United States;
- `EEM` for emerging market equities;
- `IEF` for intermediate-term U.S. Treasury bonds;
- `GLD` for gold;
- `BTC-USD` for the Bitcoin-augmented comparison.

The script downloads adjusted prices and a risk-free-rate proxy from Yahoo Finance, estimates annualized returns and covariances, solves minimum-variance and maximum-Sharpe portfolios, and exports figures and CSV tables.

## Repository Structure

```text
Markoviz/
|-- README.md
`-- markowitz-portfolio-optimization-github/
    |-- portfolio_optimization_markowitz.py
    |-- requirements.txt
    |-- README.md
    |-- DATA_NOTICE.md
    |-- GITHUB_REFERENCE.txt
    `-- .gitignore
```

## How To Reproduce

From the project folder:

```bash
cd Markoviz/markowitz-portfolio-optimization-github
python -m venv .venv
.venv\Scripts\activate
pip install -r requirements.txt
python portfolio_optimization_markowitz.py
```

On macOS/Linux, activate the environment with `source .venv/bin/activate`.

Generated charts and tables are written to `figures/`. Raw market data are not committed because they are downloaded at runtime.

## Notes

- The analysis uses Yahoo Finance data through `yfinance`; outputs can change if the provider revises historical data.
- The code is for educational and research use only and does not provide investment advice.
- Local zip archives are ignored at repository level; version control should contain the extracted project folder.
