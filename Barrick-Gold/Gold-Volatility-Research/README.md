# Gold, Monetary Regimes, and Volatility Dynamics 🪙

This repository contains the Python code, data pipeline, and generated outputs associated with the Starting Finance Club PoliTo Research article on gold, silver, macro-financial drivers, and volatility dynamics.

## 👥 Authors

- Davide D'Amico
- Pietro Weisz

## 🎯 Purpose

The project studies daily relationships among gold, silver, U.S. equities, the U.S. dollar, and long-term U.S. Treasury yields. It combines data cleaning, descriptive statistics, OLS regressions, rolling correlations, and stylized-fact outputs.

## 🗂️ Structure

```text
Gold-Volatility-Research/
|-- README.md
|-- requirements.txt
|-- data/
|   |-- raw/
|   `-- processed/
|-- output/
`-- src/
```

## 📈 Data

The scripts use Yahoo Finance data for:

- `GLD` for gold;
- `SLV` for silver;
- `^GSPC` for the S&P 500;
- `^TNX` for the 10-year U.S. Treasury yield;
- `DX-Y.NYB` for the U.S. dollar index.

The raw download starts from `2000-01-01`; the effective common sample starts on `2006-05-01`.

## ⚙️ Setup

```bash
python -m venv .venv
pip install -r requirements.txt
```

On Windows, activate with `.venv\Scripts\activate`; on macOS/Linux, use `source .venv/bin/activate`.

## ▶️ Run

From this folder:

```bash
python src/07_main.py
```

The full pipeline runs data download, preparation, descriptive analysis, gold regressions, silver regressions, and stylized-fact output generation.

## 📊 Outputs

The pipeline generates processed datasets, descriptive tables, a correlation matrix and heatmap, rolling-correlation figures, regression tables, and stylized-fact summaries in `data/processed/` and `output/`.

## 🧪 Reproducibility Notes

- Running the pipeline overwrites files with matching names in `data/` and `output/`.
- Results may vary slightly if Yahoo Finance revises or extends upstream data.
- The material is for research and education only and does not provide investment advice.

## 📚 Citation

D'Amico, D. and Weisz, P., *Gold, Monetary Regimes, and Volatility Dynamics: supporting code*, Starting Finance Club PoliTo Research, GitHub repository, accessed 30/05/2026.
