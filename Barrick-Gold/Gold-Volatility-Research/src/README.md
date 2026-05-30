# Source Code 🧩

Python scripts for the gold, silver, and macro-financial driver analysis.

## 📄 Files

| File | Role |
| --- | --- |
| `config.py` | Shared paths, ticker symbols, sample dates, and formal variable names. |
| `utils.py` | Shared data-loading and helper utilities. |
| `01_data_download.py` | Downloads raw market data from Yahoo Finance. |
| `02_data_preparation.py` | Builds clean levels, returns, and yield-change datasets. |
| `03_descriptive_analysis.py` | Produces descriptive statistics and correlation outputs. |
| `04_gold_regressions.py` | Estimates gold-return regressions. |
| `05_silver_regressions.py` | Estimates silver-return regressions. |
| `06_stylized_facts.py` | Produces stylized-fact outputs. |
| `07_main.py` | Runs the full pipeline in order. |
| `regression_tables.py` | Builds publication-facing regression tables. |

## ▶️ Run

From the project root:

```bash
python src/07_main.py
```
