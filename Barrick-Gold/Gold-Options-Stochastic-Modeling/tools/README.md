# Maintenance tools

`rebuild_bates_notebooks.py` deterministically rebuilds only the three
Bates/Hawkes notebooks that are in scope for the restructuring. The protected
Black--Scholes and Heston notebooks are never opened for writing.

`rebuild_lse_benchmarks.py` reads the local-only LSE sample and regenerates the
publication-safe Black--Scholes, Heston, and Bates parameters, aggregate
metrics, manifest, and figures without changing the protected model files.

`rebuild_exact_hawkes_outputs.py` recalibrates the constant-volatility method
benchmark and performs the deterministic Feller-constrained full affine
calibration, emitting only aggregate metrics, parameters, and figures.

`rebuild_path_outputs.py` regenerates the five-year GBM, Heston, Bates, and
full Bates--Hawkes paths from the constrained parameter snapshots.
