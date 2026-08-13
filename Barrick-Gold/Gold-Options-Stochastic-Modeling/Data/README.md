# Data and Diagnostics

This folder contains publication-safe aggregate outputs. Raw LSE chain rows and
the 64-row calibration sample stay in the Git-ignored `lse_local/` directory.

| File | Description |
| --- | --- |
| `lse_publication_manifest.json` | Snapshot time, assumptions, counts, and hash of the local sample; no raw rows. |
| `baseline_calibration_metrics.csv` | Aggregate Black--Scholes, Heston, and Bates errors. |
| `black_scholes_calibrated_params.json` | Calibrated constant-volatility Black--Scholes parameter. |
| `heston_calibrated_params.json` | Deterministic Heston parameters from the current LSE surface. |
| `bates_calibrated_params.json` | Deterministic Bates parameters from the current LSE surface. |
| `bates_hawkes_calibrated_params.json` | Deterministic full affine Heston-Hawkes parameters from the explicit Feller-constrained workflow. |
| `bates_hawkes_calibration_metrics.csv` | Refreshed dataset-level price and implied-volatility error statistics. |
| `bates_hawkes_calibration_metrics.json` | Refreshed metrics, objectives, branching ratio, and reproducibility status. |
| `bates_hawkes_residual_heatmap.png` | Full-model implied-volatility residuals across maturity and moneyness. |
| `bates_hawkes_volatility_smile.png` | Full Bates-Hawkes, Bates, and market smile slices. |
| `bates_hawkes_vs_bates.png` | Full-model fit and direct dataset-level Bates comparison. |
| `hawkes_intensity_comparison.png` | Conceptual constant-intensity versus self-exciting jump comparison. |
| `hawkes_exact_constvol_params.json` | Constant-volatility exact Hawkes benchmark parameters. |
| `hawkes_exact_vs_proxy.png` | Exact constant-volatility Hawkes versus stationary proxy benchmark. |
| `path_simulation_summary.csv` | Terminal statistics for GBM, Heston, Bates, and full Bates-Hawkes. |
| `terminal_return_percentiles_0_100.csv` | Every integer percentile of five-year simulated GLD percentage returns by model. |
| `terminal_return_percentiles.png` | Percentile curves spanning 0--100 for all four models. |
| `model_parameters_long.csv` | Long-form calibrated parameter table for all four models. |
| `model_appendix_summary.csv` | Joined calibration errors, residual normality, simulation methods, and return-shape statistics. |
| `calibration_residual_normality.csv` | Shapiro--Wilk, Jarque--Bera, and D'Agostino tests on IV residuals by model. |
| `gld_return_normality_tests.csv` | Hypothesis tests for normality of LSE daily GLD log returns. |
| `gld_return_normality.png` | Empirical density and normal Q--Q plot for GLD returns. |
| `usd_treasury_curve.png` | LSE USD Treasury curve used by pricing and simulation. |
| `online_validation_design.json` | No-look-ahead contract for the primary rolling `t -> t+1` test. |
| `online_validation_metrics.csv` | OOS errors and R-squared against the expanding mean and daily IV random walk. |
| `online_pairwise_r2.csv` | Directed model-versus-model OOS R-squared comparisons. |
| `online_loss_differential_tests.csv` | Newey--West HAC tests on daily squared-error differences. |
| `online_welch_goyal_cumulative.png` | Cumulative Welch--Goyal loss differences against benchmarks and peers. |
| `online_parameter_stability.csv` | Minimum, median, maximum, and final rolling parameters. |
| `online_calibration_convergence.csv` | Accepted updates, optimizer status, fallback count, and run time by model. |
| `oos_validation_design.json` | Secondary fixed-cutoff six-month stress-test design. |
| `oos_validation_metrics.csv` | Fixed-parameter holdout metrics; not used by current valuation. |
| `gold_path_stats_by_model.png` | Mean and 5--95 percent path bands. |
| `paths_black_scholes_5.png` | Representative GBM paths. |
| `paths_heston_5.png` | Representative Heston paths. |
| `paths_bates_5.png` | Representative Bates paths. |
| `paths_bates_hawkes_5.png` | Representative full Heston-Hawkes paths. |
| `volatility_state_paths.png` | Separate stochastic-volatility state panels. |
| `hawkes_jump_paths.png` | Full Bates-Hawkes intensity and cumulative jump paths. |
| `bates_poisson_jump_paths.png` | Bates constant intensity and cumulative Poisson jump paths. |

Regenerated figures can change with data, filters, calibration bounds, numerical settings, or random seeds.

`lse_dataset.py` creates the sole calibration and return inputs under
`Data/lse_local/`, including the maturity-specific LSE Treasury curve.
That directory is excluded from version control and must not be redistributed.
The repository intentionally contains no Interactive Brokers dataset.
