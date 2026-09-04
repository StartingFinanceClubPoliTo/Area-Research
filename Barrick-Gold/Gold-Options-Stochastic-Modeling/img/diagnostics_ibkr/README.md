# Model and market diagnostics

This folder contains the diagnostic figures and small aggregate tables prepared for the Team 8 article. The assets cover surface quality, sampling, residuals, smiles, simulated paths, Treasury curves, and rolling out-of-sample comparisons.

`figure_manifest.json` records the publication-facing set. Regenerate assets with `src/make_missing_ibkr_figures.py` or the relevant validation script, and verify the resulting image before synchronizing it with Overleaf. Raw market observations and row-level diagnostic CSVs do not belong in Git.
