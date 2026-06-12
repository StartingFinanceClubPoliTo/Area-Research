# Calibration Scripts

This folder contains script entrypoints that mirror the article structure:

| File | Role |
| --- | --- |
| `common.py` | Shared dataset loading, metadata parsing, and vega preparation. |
| `01_calibrate_black_scholes.py` | Rebuilds Black-Scholes implied-volatility and vega diagnostics. |
| `02_calibrate_heston.py` | Runs the Heston Differential Evolution plus SLSQP calibration wrapper. |
| `03_calibrate_bates.py` | Runs the Bates calibration wrapper. |
| `04_calibrate_bates_hawkes_proxy.py` | Runs the stationary-intensity Bates-Hawkes proxy calibration. |
| `05_generate_hawkes_diagnostics.py` | Generates the Hawkes intensity comparison figure used by the article. |

Run scripts from the `Gold-Options-Stochastic-Modeling/` root folder so relative data and output paths stay stable.
