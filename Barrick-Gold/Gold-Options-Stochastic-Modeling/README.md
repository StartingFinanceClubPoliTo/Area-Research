# Gold Options Stochastic Modeling 📈

Research code and reproducibility assets for the Starting Finance Club PoliTo article on stochastic modeling, calibration, and rolling out-of-sample validation of gold-option models.

## 👥 Authors

- [Salvatore Gabriele Messina](https://github.com/SalvatoreMessina11)
- [Alessandro Coco](https://github.com/0c0c)

## 🎯 Purpose

The project compares Black--Scholes, Heston, Bates, and an exact affine Bates--Hawkes specification on GLD option surfaces acquired through the IB API. It provides a reproducible path from market-data quality checks and Treasury-curve preparation to structured 8×8 sampling, calibration, diagnostics, and dense-date rolling out-of-sample evaluation.

This repository is research material. It does not provide financial advice or trading recommendations.

## 🗂️ Structure

| Path | Role |
| --- | --- |
| `src/` | Data collection, surface construction, sampling, pricing, calibration, validation, and figure generation. |
| `data/raw/` | Manually supplied source inputs that may be redistributed. |
| `data/processed/full_surfaces/` | Local destination for dense option surfaces; Git tracks only documentation and the public Treasury curve-fit metadata. |
| `data/processed/sparse_historical_surfaces/` | Local destination for historical surfaces; Git tracks only the aggregate coverage summary. |
| `img/diagnostics_ibkr/` | Article-ready diagnostic figures and aggregate supporting tables; row-level diagnostic CSVs stay local. |
| `outputs/` | Local generated runs; ignored by Git except for the placeholder. |
| `paper/` | Methodological LaTeX fragments supporting the article revision. |
| `.gitignore` | Publication boundary for environments, caches, local workflows, row-level market data, and generated runs. |
| `requirements.txt` | Runtime Python dependencies for the published scripts. |
| `CALIBRATION_WORKFLOW.md` | Detailed calibration, resume, and output conventions. |
| `SOURCE_MANIFEST.txt` | Provenance and hashes for the clean source snapshot. |

Folder-level README files document every published file in the corresponding area, together with its role and publication policy. See `src/README.md` for the complete 22-module source index.

## ⚙️ Setup

Python 3.11 or later is recommended. From PowerShell:

```powershell
py -m venv .venv
.\.venv\Scripts\Activate.ps1
python -m pip install --upgrade pip
python -m pip install -r requirements.txt
```

Live collection requires a locally configured IB API session. Credentials and account details must remain outside the repository.

The repository deliberately excludes row-level option quotes and underlying-price histories. Run the collection/build scripts with your own authorized data access before executing the surface audit, calibration, or rolling validation commands below.

## ▶️ Usage

Run the deterministic model-stack smoke test first:

```powershell
python src\model_smoke_test.py
```

Audit the admitted dense-date universe before calibration:

```powershell
python src\audit_market_data.py --start 2026-07-06 --end 2026-09-02 --min-dte 75 --cc-size 64 --min-expiries 3
```

Preview the rolling design without launching expensive optimizations:

```powershell
python src\oos_validation.py --start 2026-07-06 --end 2026-09-02 --min-dte 75 --min-surface-points 64 --min-origin-expiries 3 --dry-run
```

Run final rolling-origin calibrations with resume support:

```powershell
python src\oos_validation.py --start 2026-07-06 --end 2026-09-02 --min-dte 75 --min-surface-points 64 --min-origin-expiries 3 --profile full --calibrate-only
```

Then compute the rolling out-of-sample scores:

```powershell
python src\oos_validation.py --start 2026-07-06 --end 2026-09-02 --min-dte 75 --min-surface-points 64 --min-origin-expiries 3 --profile full
```

See `CALIBRATION_WORKFLOW.md` for single-date calibration, cross-date sampling selection, manifests, checkpoints, and restart behavior.

## 📊 Outputs

Generated calibration and validation runs are written below `outputs/` and remain local by default. Each completed calibration stores its manifest and model result immediately, so interrupted runs can resume without repeating successful optimizations. Publication-facing aggregate figures and small tables live in `img/diagnostics_ibkr/`.

The implementation enforces the following research controls:

- no look-ahead in Treasury-curve selection;
- a minimum DTE of 75 days for the official dense-date exercise;
- at least 64 unique valid observations and three expiries per admitted origin by default;
- a fixed 8×8 Chebyshev--Chebyshev sample of actual observations;
- no synthetic duplication when market coverage is insufficient;
- the next available dense date, rather than the next calendar day, as the out-of-sample target;
- persistent per-model checkpoints and deterministic seeds.

## 🔒 Data and publication policy

Raw account-linked or restricted market records, contract identifiers, row-level quotes, credentials, caches, virtual environments, and local run outputs must not be committed. The repository contains only code, redistributable public inputs, publication-safe aggregates, and reviewed figures. Code belongs in the Research GitHub repository; the SF Drive publication package must contain only the editorial deliverables required by the current publication guide.

## 📚 Citation

Messina, Salvatore Gabriele, and Alessandro Coco. *Advanced Stochastic Modeling for Gold Options: Supporting Code*. Starting Finance Club PoliTo, Research repository, 2026.

The direct article PDF link will be added after publication.
