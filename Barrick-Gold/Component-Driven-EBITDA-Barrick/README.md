# Component-Driven EBITDA Forecast for Barrick Gold

| Field | Value |
| --- | --- |
| Project | [Barrick Gold](../) |
| Article | 1 |
| Authors | [Giacomo Scali](https://www.linkedin.com/in/giacomo-scali-abp01/), [Jacopo Foralosso](https://www.linkedin.com/in/jacopo-foralosso-6753a2256) |
| Reviewer | [Salvatore Messina](https://www.linkedin.com/in/salvatore-messinaa) |
| Published | May 8, 2026 |

This folder contains the code supporting the article *"A Component-Driven Methodology for EBITDA Forecasting in Gold Mining"*. The workflow decomposes Barrick Gold's five-year EBITDA forecast into cost of sales, physical production, and gold-price scenarios.

## Purpose

The project separates the valuation workflow into reproducible analytical components:

- cost-of-sales forecasting;
- production forecasting from operating drivers;
- option-implied gold-price volatility calibration;
- Monte Carlo EBITDA distribution construction.

## Structure

| Folder | Role |
| --- | --- |
| `cost_of_sales/` | Log-space ARIMA workflow for cost-of-sales forecasting. |
| `production/` | Bottom-up production forecast using ore processed, average grade, recovery rate, and aggregation logic. |
| `ebitda_montecarlo/` | Black-Scholes inversion of GLD option prices, maturity-level volatility calibration, GBM gold-price simulation, and final EBITDA distribution. |

## Setup

Python 3.10 or newer is recommended.

```bash
pip install numpy pandas scipy matplotlib seaborn plotly pmdarima nelson_siegel_svensson ib_insync
```

## Data

Operational inputs such as ore processed, average grade, recovery rate, gold production, and cost of sales were extracted from Barrick Gold Corporation quarterly reports:

https://www.barrick.com/English/investors/

The development workbook `Quarterly Data.xlsx` is not included in this repository because it contains private working notes. To reproduce the production workflow, rebuild the workbook from the public quarterly reports with one sheet per mine and a `Dati_P` aggregate sheet.

Option-chain data for GLD were retrieved through the Interactive Brokers API on April 2, 2026 and are not redistributed here. The notebook contains the regeneration pipeline, which requires an active IB Gateway or TWS connection on `127.0.0.1:7497`.

## Reproducibility Notes

- Public source documents must be downloaded again from the original providers.
- Local working files and private spreadsheets are intentionally excluded from Git.
- The notebooks are publication-facing references for the article methodology, not investment advice.
