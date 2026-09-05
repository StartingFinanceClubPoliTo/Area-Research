# Barrick Mining UNIFICATO 🪙

Unified thesis and executable reproducibility companion for the Starting Finance Club PoliTo Barrick research programme. Release date: **TBD**.

> Educational and research material only. Outputs are conditional model sensitivities, not fair values, target prices or investment recommendations.

## Current executable companion

The authoritative code run is `20260904T130000Z-team8-refresh-v4`, based on the Team 8 snapshot of September 2, 2026:

- 605 eligible GLD calls, 12 expiries, 146 strikes and DTE 79–653 before sampling;
- fixed CC64 sample: 64 actual contracts, 8 expiries and 20 strikes;
- same-date 14-tenor Treasury NSS curve, 2.0608 bp RMSE;
- Heston best in-sample IV RMSE: 49.4881 bp;
- Full Bates–Hawkes best date-equal rolling OOS IV RMSE: 65.0507 bp;
- 31 dense dates, 30 OOS origins and 4,667 common forecasts;
- 8,192 Barrick valuation paths and a USD 44.13 market reference;
- conditional model medians between USD 35.14 and USD 35.87.

## Structure

| Path | Role |
| --- | --- |
| `thesis/` | September-refreshed unified thesis, 198-page PDF and complete build sources, dated `TBD`; historical evidence is explicitly dated. |
| `code/` | Current Team 8 adapter, valuation engine, aggregate evidence, tests, manifests and immutable outputs. |

The current paper-level analysis is in [`../Barrick Mining PAPER/`](../Barrick%20Mining%20PAPER/). Licensed/provider-controlled row-level option data are intentionally excluded. The complete 17-student authorship and module inventory are in the [code README](code/README.md).

Persistence comparisons use 4,026 interpolation-domain observations, distinct from the 4,667 model-comparison targets; none of the models beats persistence. The NSS input is a continuously compounded par-yield proxy, not a bootstrapped zero curve.

## Reproduce the current run

```bash
cd code
python -m venv .venv
.venv/Scripts/python -m pip install -e ".[test]"
.venv/Scripts/python run_multimodel_valuation.py \
  --config config/multimodel_valuation_20260902_team8_refresh.json \
  --run-id <fresh-run-id>
.venv/Scripts/python -m pytest -q tests/test_team8_20260902_refresh.py
```

The runner refuses to overwrite an existing run. Use `code/data/manifests/valuation/AUTHORITATIVE_MULTIMODEL.json` to identify the promoted output.

## Interpretation boundary

Team 8 governs only the option-implied gold-path shape. Team 4 operating vectors and the unified DCF/WACC/equity bridge are common across the four engines. The GLD calibration is risk-neutral; no one-for-one GLD-to-ounce or validated risk-neutral-to-physical mapping is claimed.

## Citation

Barrick Gold Research Teams (2026), *Stochastic Valuation of Barrick Mining: Unified Thesis and Supporting Code*, Starting Finance Club PoliTo Research, working draft, release date TBD.
