# Team 8 calibration workflow

This document defines the reproducible calibration and rolling out-of-sample workflow for Black--Scholes, Heston, Bates, and exact affine Bates--Hawkes models.

## 1. Pre-calibration gate

Run the market-data audit before any expensive optimization:

```powershell
python src\audit_market_data.py --start 2026-07-06 --end 2026-09-02 --min-dte 75 --cc-size 64 --min-expiries 3
```

The official dense-date exercise admits a date only when it contains at least 64 unique valid observations after the DTE filter and at least three expiries by default. Sparse dates remain available for coverage analysis but are not padded or duplicated.

## 2. Common sampling geometry

Sampling geometry must be selected across admitted dates, not independently on each date. The candidate set is:

| Code | Maturity nodes | Strike nodes |
| --- | --- | --- |
| `UU` | Uniform | Uniform |
| `CU` | Chebyshev | Uniform |
| `UC` | Uniform | Chebyshev |
| `CC` | Chebyshev | Chebyshev |
| `EU` | Exponential | Uniform |
| `EC` | Exponential | Chebyshev |
| `GU` | Gaussian-centered | Uniform |
| `GC` | Gaussian-centered | Chebyshev |

The primary selection criterion is cross-date holdout maximum absolute implied-volatility error. Holdout RMSE and MAE are tie-breakers. The production dense rolling exercise currently uses a fixed `CC` 8×8 sample of actual observations at every origin.

```powershell
python src\batch_sampling_offline.py --start 2026-07-06 --end 2026-09-02 --n-t 8 --n-k 8 --min-dte 75
```

## 3. Single-date calibration

`calibrate_surface.py` reads an admitted surface directly, creates the selected 64-point sample, and writes each model result as soon as it completes.

```powershell
python src\calibrate_surface.py --date 2026-09-02 --surface data\processed\full_surfaces\GLD_2026-09-02_eligible_full_surface.csv --strategy CC --profile full
```

The per-date directory contains a manifest, the exact calibration surface, one JSON result per model, diagnostics, and a compact summary. Successful result files are reusable checkpoints.

## 4. Batch calibration

```powershell
python src\batch_calibrate.py --dates 2026-08-28,2026-08-31,2026-09-01,2026-09-02 --strategy CC --models bs,heston,bates,hawkes --profile full --seed 8
```

The order is Black--Scholes, Heston, Bates, and exact Bates--Hawkes. Bates is retained as the nested jump-diffusion benchmark and as a useful initialization reference. Resume behavior prevents successful models from being recalibrated unless a forced rerun is explicitly requested.

## 5. Dense rolling out-of-sample validation

Preview admitted origins and targets first:

```powershell
python src\oos_validation.py --start 2026-07-06 --end 2026-09-02 --min-dte 75 --min-surface-points 64 --min-origin-expiries 3 --dry-run
```

Run or resume origin calibrations:

```powershell
python src\oos_validation.py --start 2026-07-06 --end 2026-09-02 --min-dte 75 --min-surface-points 64 --min-origin-expiries 3 --profile full --calibrate-only
```

Score the next available dense-date forecasts:

```powershell
python src\oos_validation.py --start 2026-07-06 --end 2026-09-02 --min-dte 75 --min-surface-points 64 --min-origin-expiries 3 --profile full
```

The target is the next admitted dense date, not necessarily the next calendar date. The final admitted date is target-only unless a later target becomes available.

## 6. Required run metadata

Every final run must preserve:

- date and exact source-surface path;
- source hash or equivalent immutable identifier;
- full and calibration sample sizes;
- maturity and strike sampling geometry;
- spot and no-look-ahead Treasury curve date;
- numerical profile, objective, model list, and seed;
- success state, elapsed time, parameters, and diagnostics for each model.

The default deterministic seed is `8`. Final results use the `full` profile; `quick` is reserved for smoke and pipeline checks.

## 7. Output and artifact policy

Generated runs belong under `outputs/` and remain local by default. Only reviewed aggregate tables and figures needed by the article may be promoted to `img/diagnostics_ibkr/`. Raw restricted records, credentials, caches, temporary notebooks, and failed or superseded artifacts must not enter the SF GitHub staging or Drive publication package.

Before article synchronization, verify the model smoke test, manifests, figure hashes, and the exact numerical claims transferred to Overleaf.
