# Processed research data

Processed data supports coverage audits, sampling comparisons, calibration, and rolling out-of-sample validation.

Row-level option surfaces and underlying-price histories are generated and retained locally; they are ignored by Git. The repository tracks only documentation, public Treasury-derived inputs, aggregate coverage metadata, and reviewed publication assets. Do not overwrite a dated local surface without recording its provenance and validation status.

## Versioned inventory

| Path | Role |
| --- | --- |
| `.gitkeep` | Preserves the processed-data directory in a fresh clone. |
| `usd_treasury_history.csv` | Normalized, continuously compounded Treasury history used by `src/rates.py`. |
| `full_surfaces/README.md` | Explains dense-surface gates and the local-only CSV policy. |
| `full_surfaces/GLD_2026-09-02_nss_curve.json` | NSS parameters and fit diagnostics for 2 September 2026. |
| `sparse_historical_surfaces/README.md` | Explains sparse-surface handling and exclusions. |
| `sparse_historical_surfaces/historical_surface_summary.csv` | Aggregate coverage audit by date. |
