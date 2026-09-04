# UNIFICATO - Stochastic Valuation of Barrick Mining

Publication-facing code, aggregate evidence and reproducibility material for the Starting Finance Club PoliTo Barrick research programme. Release date: **TBD**.

> Educational and research material only. The valuation outputs are conditional model sensitivities, not fair values, target prices or investment recommendations.

## Current authoritative experiment

The current companion run is `20260904T130000Z-team8-refresh-v4`. It integrates the Team 8 snapshot dated 2 September 2026 while changing only the gold-price layer:

- 605 eligible GLD calls, 12 expiries, 146 strikes and DTE 79--653 before sampling;
- fixed CC64 sample: 64 distinct actual contracts, 8 expiries and 20 strikes;
- U.S. Treasury NSS fit on 14 tenors, same date, RMSE 2.0608 bp;
- Heston best in-sample IV RMSE: 49.4881 bp;
- Full Bates--Hawkes best date-equal rolling OOS IV RMSE: 65.0507 bp;
- rolling panel: 31 dense dates, 30 forecast origins and 4,667 common forecasts;
- Full Bates--Hawkes branching ratio: 0.556903;
- 8,192 Barrick valuation paths with common operating, WACC and DCF layers;
- 2 September NYSE:B close: USD 44.13; conditional medians USD 35.14--35.87.

The primary structural scenario is selected on rolling OOS performance. The distinct in-sample Heston ranking is preserved in the configuration, output manifest and paper.

## Research architecture

Team 8 governs the option-implied gold-path shape. Team 4 supplies Barrick Q1--Q2 2026 operating actuals followed by the frozen Q3 2026--2030 production and unit-cost forecast. The unified corporate layer supplies the same tax, growth/ROIC/reinvestment, stochastic WACC, terminal-value and equity-bridge logic to every gold engine.

The GLD calibration is under the risk-neutral measure. The companion does not claim a one-for-one GLD-share-to-ounce conversion or a validated risk-neutral-to-physical mapping. The common USD 4,417/oz gold level is Barrick's Q2 average realized price and is not a point-in-time spot quote.

## Repository map

| Path | Contents |
| --- | --- |
| `src/barrick_unified/` | Validated market, operating, valuation and reporting modules. |
| `config/` | Versioned base and multi-model valuation contracts. |
| `data/processed/team8/calibration_20260902/` | Redistribution-safe calibration parameters and aggregate diagnostics. |
| `data/processed/team8/oos_20260902/` | Aggregate rolling OOS manifest and model summary. |
| `data/manifests/rates/` | Versioned NSS curve metadata and parameters. |
| `data/manifests/valuation/` | Hash-complete valuation run manifests and authoritative pointer. |
| `parity/sources/team-8-current/` | Frozen public Team 8 source package plus the Barrick path adapter. |
| `outputs/valuation/<run-id>/` | Aggregate valuation tables and JSON summaries. |
| `figures/valuation/<run-id>/` | Generated conditional valuation figures. |
| `tests/` | Offline unit and integration tests, including the 2 September refresh contract. |
| `tools/` | Acquisition, rendering and provenance-audit utilities. |
| `docs/` | Handoff, code-map and snapshot notes. |

Historical versioned inputs and runs remain available for audit. They are not current merely because their files remain in the repository; use the authoritative pointers.

## Reproduce the current valuation

Python 3.11 or newer is required.

```bash
python -m venv .venv
.venv/Scripts/python -m pip install -e ".[test]"
.venv/Scripts/python run_multimodel_valuation.py --config config/multimodel_valuation_20260902_team8_refresh.json --run-id <fresh-run-id>
```

The runner refuses to overwrite an existing run directory. A successful run writes valuation CSV/JSON/LaTeX outputs, two figures and a manifest containing input, source-code and artifact hashes.

Run the refresh-specific regression tests with:

```bash
.venv/Scripts/python -m pytest -q tests/test_team8_20260902_refresh.py
```

## Data and provenance rules

- Row-level option observations and licensed/provider-controlled raw data are not published.
- Public Team 8 source modules, calibration parameters and aggregate OOS results are versioned.
- Every promoted run has an immutable identifier and hash-complete manifest.
- Legacy Team 4 gold-price simulations and illustrative valuation outputs are excluded from the current experiment.
- Caches, environments, logs and local review assets remain outside publication.

## Citation

Starting Finance Club PoliTo Research, *UNIFICATO - Stochastic Valuation of Barrick Mining*, code and reproducibility companion, working draft, release date TBD.
