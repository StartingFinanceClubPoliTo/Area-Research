# Markowitz Portfolio Optimization 📊

📄 Article PDF: https://sfclubpolito.it/pdf-viewer.html?file=Pubblicazioni%2FPortfolio%20Optimization%20-%20From%20Markowitz%20Theory%20to%20Practical%20Implementation.pdf&lang=it

This folder contains the Python code supporting the Research article *Portfolio Optimization: From Markowitz Theory to Practical Implementation*.

## 👥 Author

- [Antonio Guarini](https://github.com/Anthony27-x)

## 🎯 Purpose

The script reproduces classical and out-of-sample Markowitz portfolio exercises over a liquid ETF universe, with and without a Bitcoin-augmented comparison.

## 🧺 Asset Universe

- `SPY` for U.S. equities;
- `VEA` for developed markets outside the United States;
- `EEM` for emerging market equities;
- `IEF` for intermediate-term U.S. Treasury bonds;
- `GLD` for gold;
- `BTC-USD` for the expanded comparison;
- `^IRX` as the 13-week Treasury Bill risk-free-rate proxy.

## 🗂️ Structure

| File | Role |
| --- | --- |
| `portfolio_optimization_markowitz.py` | Main script for data download, optimization, backtests, plots, and CSV exports. |
| `requirements.txt` | Python dependencies. |
| `DATA_NOTICE.md` | Data-source and raw-data notice. |
| `GITHUB_REFERENCE.txt` | Suggested citation text for the article reference section. |
| `.gitignore` | Local Python and generated-output exclusions for this project. |

## ⚙️ Setup

```bash
python -m venv .venv
pip install -r requirements.txt
```

## ▶️ Run

```bash
python portfolio_optimization_markowitz.py
```

Generated charts and CSV tables are written to `figures/`.

## 📊 Outputs

The generated `figures/` directory can include normalized price charts, correlation matrices, portfolio clouds, efficient frontiers, optimized weight tables, fixed and rolling out-of-sample cumulative wealth, turnover plots, and CSV summary tables.

## 🧪 Reproducibility Notes

- The code sets deterministic seeds for random portfolio simulations.
- Market data are downloaded at runtime through `yfinance`; raw price files are not stored in this repository.
- Results can change if Yahoo Finance revises historical data or if package versions change.
- The analysis is for educational and research purposes only and does not provide financial advice.

## 📚 Citation

Guarini, A., *Portfolio Optimization: From Markowitz Theory to Practical Implementation: supporting code*, Starting Finance Club PoliTo Research, GitHub repository, accessed 30/05/2026.
