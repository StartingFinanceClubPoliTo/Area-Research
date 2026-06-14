# Data and Diagnostics

This folder contains the GLD option inputs and publication-facing outputs.

| File | Description |
| --- | --- |
| `gld_chain_wide_chain.csv` | Wide option-chain input. |
| `gld_chain_wide_meta.json` | Spot and dataset metadata. |
| `gld_iv_dataset_chebyshev.csv` | 64-option structured calibration sample. |
| `bates_hawkes_calibrated_params.json` | Full affine Heston-Hawkes parameters calibrated from call prices. |
| `bates_hawkes_option_diagnostics.csv` | Option-level prices, implied volatilities, and residuals for full Bates-Hawkes and Bates. |
| `bates_hawkes_calibration_metrics.csv` | Dataset-level price and implied-volatility error statistics. |
| `bates_hawkes_calibration_metrics.json` | Metrics plus objective, branching ratio, and Bates-limit comparison. |
| `bates_hawkes_residual_heatmap.png` | Full-model implied-volatility residuals across maturity and moneyness. |
| `bates_hawkes_volatility_smile.png` | Full Bates-Hawkes, Bates, and market smile slices. |
| `bates_hawkes_vs_bates.png` | Full-model fit and direct dataset-level Bates comparison. |
| `hawkes_intensity_comparison.png` | Conceptual constant-intensity versus self-exciting jump comparison. |
| `hawkes_exact_constvol_params.json` | Constant-volatility exact Hawkes benchmark parameters. |
| `hawkes_exact_vs_proxy.png` | Exact constant-volatility Hawkes versus stationary proxy benchmark. |
| `path_simulation_summary.csv` | Terminal statistics for GBM, Heston, Bates, and full Bates-Hawkes. |
| `gold_path_stats_by_model.png` | Mean and 5--95 percent path bands. |
| `paths_black_scholes_5.png` | Representative GBM paths. |
| `paths_heston_5.png` | Representative Heston paths. |
| `paths_bates_5.png` | Representative Bates paths. |
| `paths_bates_hawkes_5.png` | Representative full Heston-Hawkes paths. |
| `volatility_state_paths.png` | Separate stochastic-volatility state panels. |
| `hawkes_jump_paths.png` | Full Bates-Hawkes intensity and cumulative jump paths. |
| `bates_poisson_jump_paths.png` | Bates constant intensity and cumulative Poisson jump paths. |

Regenerated figures can change with data, filters, calibration bounds, numerical settings, or random seeds.
