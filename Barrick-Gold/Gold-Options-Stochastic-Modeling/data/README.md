# Data assets 🗃️

This folder separates redistributable source inputs and publication-safe aggregates from local-only processed research surfaces.

- `raw/` contains manually supplied public inputs.
- `processed/full_surfaces/` is the local destination for dense option surfaces; Git tracks only its documentation and public Treasury curve-fit metadata.
- `processed/sparse_historical_surfaces/` is the local destination for lower-coverage dates; Git tracks only an aggregate coverage summary.

Do not commit credentials, account metadata, unrestricted API dumps, contract identifiers, or row-level market records. Every calibration should identify its exact local source surface and curve date in a run manifest.

## Versioned inventory

| Path | Role |
| --- | --- |
| `raw/README.md` | Policy and provenance notes for redistributable raw inputs. |
| `raw/daily-treasury-rates2026.csv` | Manually supplied U.S. Treasury source table. |
| `processed/.gitkeep` | Keeps the local processed-data destination present in a fresh clone. |
| `processed/README.md` | Processing and publication policy for derived data. |
| `processed/usd_treasury_history.csv` | Normalized Treasury history used by the no-look-ahead rate module. |
| `processed/full_surfaces/README.md` | Dense-surface admission and local-storage policy. |
| `processed/full_surfaces/GLD_2026-09-02_nss_curve.json` | Public NSS curve-fit metadata for the final dense date. |
| `processed/sparse_historical_surfaces/README.md` | Sparse-surface coverage and local-storage policy. |
| `processed/sparse_historical_surfaces/historical_surface_summary.csv` | Date-level coverage ranges and counts without contract rows. |
