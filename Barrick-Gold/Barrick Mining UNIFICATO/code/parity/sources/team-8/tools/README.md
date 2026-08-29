# Maintenance tools

`Main.ipynb` and `main.py` are the default workflow. The notebook contains no
model implementation and delegates ingestion and every rebuild step to Python.
It performs a local LSE-key gate before any write.

`rebuild_bates_notebooks.py` deterministically rebuilds the three
Bates/Hawkes model notebooks when their thin presentation layer needs refresh.

`rebuild_lse_benchmarks.py` reads the local-only LSE sample and regenerates the
publication-safe Black--Scholes, Heston, and Bates parameters, aggregate
metrics, residual normality tests, return normality tests, manifest, and figures.

`rebuild_exact_hawkes_outputs.py` recalibrates the constant-volatility method
benchmark and performs the deterministic Feller-constrained full affine
calibration, emitting only aggregate metrics, parameters, and figures.

`rebuild_online_validation.py` is the primary predictive evaluation. It
reconstructs a fixed normalized IV grid, launches four parallel Python worker
processes, performs 124 rolling one-step-ahead calibrations per model with
chronological warm starts, and publishes OOS R-squared, pairwise comparisons,
Welch--Goyal curves, HAC tests, parameter stability, and convergence summaries.
Local caches and checkpoints are accepted only when their SHA-256 input
fingerprints match the current option, GLD, grid, and rate files.

`rebuild_oos_validation.py` retains the stricter secondary stress test: one
historical calibration is frozen for an untouched six-month holdout. Its
parameters are never reused by current valuation or simulation.

`rebuild_path_outputs.py` regenerates the five-year GBM, Heston, Bates, and
full Bates--Hawkes paths from the constrained parameter snapshots and the LSE
forward-rate curve, including all terminal return percentiles from 0 to 100.
