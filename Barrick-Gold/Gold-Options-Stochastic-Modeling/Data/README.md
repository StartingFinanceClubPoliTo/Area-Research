# Data and Diagnostics

This folder contains publication-safe aggregate outputs. Raw LSE chain rows and
the 64-row calibration sample stay in the Git-ignored `lse_local/` directory.

| File | Description |
| --- | --- |
| `lse_publication_manifest.json` | Snapshot time, assumptions, counts, and hash of the local sample; no raw rows. |
| `baseline_calibration_metrics.csv` | Aggregate Black--Scholes, Heston, and Bates errors. |
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
| `gold_path_stats_by_model.png` | Mean and 5--95 percent path bands. |
| `paths_black_scholes_5.png` | Representative GBM paths. |
| `paths_heston_5.png` | Representative Heston paths. |
| `paths_bates_5.png` | Representative Bates paths. |
| `paths_bates_hawkes_5.png` | Representative full Heston-Hawkes paths. |
| `volatility_state_paths.png` | Separate stochastic-volatility state panels. |
| `hawkes_jump_paths.png` | Full Bates-Hawkes intensity and cumulative jump paths. |
| `bates_poisson_jump_paths.png` | Bates constant intensity and cumulative Poisson jump paths. |

Regenerated figures can change with data, filters, calibration bounds, numerical settings, or random seeds.

`lse_dataset.py` creates the sole calibration input under `Data/lse_local/`.
That directory is excluded from version control and must not be redistributed.
The repository intentionally contains no Interactive Brokers dataset.
