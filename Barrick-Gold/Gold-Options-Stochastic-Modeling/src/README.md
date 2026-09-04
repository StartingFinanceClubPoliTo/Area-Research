# Source modules ⚙️

This folder contains the executable research pipeline for the Team 8 gold-options study.

## Main entry points

| File | Role |
| --- | --- |
| `audit_market_data.py` | Audits date-level coverage and applies the official dense-date admission rules. |
| `ibkr_gld_today_surface.py` | Collects a current GLD option surface through the IB API. |
| `ibkr_gld_historical_surface.py` | Reconstructs bounded historical surfaces through the IB API. |
| `rebuild_full_surface_nss.py` | Rebuilds a dated full surface with the appropriate Treasury curve. |
| `compare_sampling_all.py` | Compares structured 8×8 sampling geometries on one surface. |
| `batch_sampling_offline.py` | Applies the sampling comparison consistently across dates. |
| `calibrate_surface.py` | Calibrates the four model families on one admitted surface. |
| `batch_calibrate.py` | Runs resumable calibration across multiple dates. |
| `oos_validation.py` | Performs dense-date rolling-origin calibration and out-of-sample scoring. |
| `make_missing_ibkr_figures.py` | Rebuilds publication-facing diagnostic figures from verified aggregates. |
| `model_smoke_test.py` | Runs a fast deterministic check of the pricing stack. |

The remaining modules provide numerical kernels and shared logic. Run scripts from the repository root so relative paths resolve consistently. Generated files belong in `outputs/`, `data/processed/`, or `img/diagnostics_ibkr/`, not here.
