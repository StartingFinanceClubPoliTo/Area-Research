# Maintenance tools

`rebuild_bates_notebooks.py` deterministically rebuilds only the three
Bates/Hawkes notebooks that are in scope for the restructuring. The protected
Black--Scholes and Heston notebooks are never opened for writing.

`rebuild_exact_hawkes_outputs.py` performs the deterministic Feller-constrained
full affine calibration and regenerates its option-level metrics and figures.

`rebuild_path_outputs.py` regenerates the five-year GBM, Heston, Bates, and
full Bates--Hawkes paths from the constrained parameter snapshots.
