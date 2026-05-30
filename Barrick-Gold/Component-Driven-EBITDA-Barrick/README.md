# Component-Driven EBITDA Forecast for Barrick Gold 🧱

📄 Article PDF: https://sfclubpolito.it/pdf-viewer.html?file=Pubblicazioni%2FEBITDA%20forecasting%20in%20gold%20mining.pdf&lang=it

This folder contains the code supporting Article 5 in the Barrick Gold Research track and the Starting Finance Club PoliTo Research article *A Component-Driven Methodology for EBITDA Forecasting in Gold Mining*.

## 👥 Authors

- [Giacomo Scali](https://www.linkedin.com/in/giacomo-scali-abp01/)
- [Jacopo Foralosso](https://www.linkedin.com/in/jacopo-foralosso-6753a2256)

## 🎯 Purpose

The workflow decomposes Barrick Gold's five-year EBITDA forecast into cost of sales, mine-level production drivers, option-implied gold-price volatility, and Monte Carlo EBITDA scenarios.

## 🗂️ Structure

| Folder | Role |
| --- | --- |
| `cost_of_sales/` | Log-space ARIMA workflow for cost-of-sales forecasting. |
| `production/` | Bottom-up production forecast using ore processed, average grade, recovery rate, and aggregation logic. |
| `ebitda_montecarlo/` | Black-Scholes inversion of GLD option prices, maturity-level volatility calibration, GBM gold-price simulation, and final EBITDA distribution. |

## ⚙️ Setup

Python 3.10 or newer is recommended.

```bash
pip install numpy pandas scipy matplotlib seaborn plotly pmdarima nelson_siegel_svensson ib_insync
```

## ▶️ Usage

Open the notebook in the component you need and run the cells in order. The notebooks are publication-facing references for the article methodology.

## 📊 Data

Operational inputs were extracted from Barrick Gold Corporation quarterly reports:

https://www.barrick.com/English/investors/

The development workbook `Quarterly Data.xlsx` is not committed because it contains private working notes. GLD option-chain data were retrieved through the IB API on April 2, 2026 and are not redistributed here.

## 🧪 Reproducibility Notes

- Public source documents must be downloaded again from the original providers.
- Local working files and private spreadsheets are intentionally excluded from Git.
- The material is for research and education only and does not provide investment advice.

## 📚 Citation

Scali, G. and Foralosso, J., *Component-Driven EBITDA Forecast for Barrick Gold: supporting code*, Starting Finance Club PoliTo Research, GitHub repository, published 08/05/2026, accessed 30/05/2026.
