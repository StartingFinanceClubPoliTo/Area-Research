# Source Code 🧩

Python scripts for the gold, silver, and macro-financial driver analysis.

## 📄 Files

| File | Role |
| --- | --- |
| `config.py` | Shared paths, ticker symbols, sample dates, and formal variable names. |
| `utils.py` | Shared data-loading, OLS, figure, CSV, and LaTeX-output helpers. |
| `01_data_download.py` | Downloads raw market data from Yahoo Finance. |
| `02_data_preparation.py` | Builds clean levels, returns, and yield-change datasets. |
| `03_descriptive_analysis.py` | Produces descriptive statistics, correlations, and quality-check figures. |
| `04_gold_regressions.py` | Estimates gold-return regressions. |
| `05_silver_regressions.py` | Estimates silver-return regressions. |
| `06_stylized_facts.py` | Produces stylized-fact diagnostics. |
| `07_garch_estimation.py` | Estimates GARCH(1,1) volatility models and exports conditional-volatility outputs. |
| `08_arfima_analysis.py` | Computes ARFIMA-style/GPH long-memory diagnostics for absolute returns. |
| `09_regression_tables.py` | Builds publication-facing regression tables in CSV and LaTeX formats. |
| `main.py` | Runs the full pipeline in order. |

## ▶️ Run

From the project root:

```bash
python src/main.py
```
