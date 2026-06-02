# Gold, Monetary Regimes, and Volatility Dynamics 🪙

This folder supports the Starting Finance Club PoliTo Research article on gold, silver, macro-financial drivers, and volatility dynamics from the Gold Standard to modern econometric evidence.

## 👥 Authors

- Davide D'Amico
- Pietro Weisz

## 🎯 Purpose

The project provides a reproducible empirical pipeline for studying daily relationships among gold, silver, U.S. equities, the U.S. dollar, and long-term U.S. Treasury yields.

The workflow covers data download, cleaning, descriptive statistics, OLS regressions, rolling correlations, stylized-fact diagnostics, GARCH volatility estimation, ARFIMA-style long-memory checks, and LaTeX-ready output generation.

## 🗂️ Structure

| Path | Role |
| --- | --- |
| `data/raw/` | Raw Yahoo Finance downloads and intermediate level datasets. |
| `data/processed/` | Clean asset levels and return datasets used by the analysis scripts. |
| `output/csv/` | Machine-readable regression, volatility, long-memory, and summary outputs. |
| `output/figures/` | Generated charts for correlations, quality checks, volatility, and long-memory diagnostics. |
| `output/tables/` | LaTeX-ready tables for article integration. |
| `src/` | Python scripts for the full empirical pipeline. |
| `requirements.txt` | Python dependencies. |
| `CITATION.cff` | Citation metadata for the code package. |

## 📈 Data

The scripts use Yahoo Finance data for:

- `GLD` for gold;
- `SLV` for silver;
- `^GSPC` for the S&P 500;
- `^TNX` for the 10-year U.S. Treasury yield;
- `DX-Y.NYB` for the U.S. dollar index.

The raw download starts from `2000-01-01`. The common asset-level sample starts on `2006-05-01`, and the effective return sample starts on `2006-05-02`.

## ⚙️ Setup

```bash
python -m venv .venv
pip install -r requirements.txt
```

On Windows, activate with `.venv\Scripts\activate`; on macOS/Linux, use `source .venv/bin/activate`.

## ▶️ Run

From this folder:

```bash
python src/main.py
```

The full pipeline runs the scripts in order:

```text
01_data_download.py
02_data_preparation.py
03_descriptive_analysis.py
04_gold_regressions.py
05_silver_regressions.py
06_stylized_facts.py
07_garch_estimation.py
08_arfima_analysis.py
09_regression_tables.py
```

Each numbered script can also be run independently from the project root.

## 📊 Outputs

The pipeline writes processed datasets, descriptive statistics, correlation tables, rolling-correlation figures, OLS regression outputs, stylized-fact diagnostics, GARCH conditional-volatility estimates, ARFIMA/GPH long-memory outputs, and LaTeX-ready tables.

## 🧪 Reproducibility Notes

- Running the pipeline overwrites files with matching names in `data/` and `output/`.
- The first step relies on Yahoo Finance, so results may vary if upstream data are revised.
- Preserved raw data are included to make the pre-publication outputs inspectable without repeating the download step.
- The article is scheduled for publication on June 19, 2026.
- The material is for research and education only and does not provide investment advice.

## 📚 Citation

D'Amico, D. and Weisz, P., *Gold, Monetary Regimes, and Volatility Dynamics: supporting code*, Starting Finance Club PoliTo Research, GitHub repository, expected publication 19/06/2026, accessed 02/06/2026.
