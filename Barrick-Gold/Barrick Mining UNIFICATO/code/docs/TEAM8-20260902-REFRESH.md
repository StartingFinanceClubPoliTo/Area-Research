# Team 8 refresh - 2 September 2026

This note records the current gold-layer handoff used by the Barrick multi-model valuation.

## Frozen evidence

- Eligible GLD surface: 605 calls, 12 expiries, 146 distinct strikes, DTE 79--653.
- Calibration surface: fixed CC64 geometry with 64 actual contracts, 8 expiries and 20 strikes.
- Treasury curve: same-date NSS fit on 14 tenors; RMSE 2.0608 bp against the continuously compounded par-yield proxy.
- In-sample IV RMSE (bp): Black--Scholes 97.2107; Heston 49.4881; Bates 54.6273; Full Bates--Hawkes 52.9714.
- Full Bates--Hawkes branching ratio: 0.556903.
- Rolling OOS: 31 dense dates, 30 origins, 4,667 common forecasts.
- Date-equal rolling IV RMSE (bp): Black--Scholes 140.6528; Heston 80.1964; Bates 69.3055; Full Bates--Hawkes 65.0507.
- Date-equal OOS R-squared versus origin mean: -2.58%, 66.07%, 72.97%, 75.81%.
- Date-equal OOS R-squared versus persistence: -268.52%, -27.24%, -12.92%, -3.68%.

## Operational changes

The versioned configuration is `config/multimodel_valuation_20260902_team8_refresh.json`. A small adapter at `parity/sources/team-8-current/path_simulation.py` maps the current calibrated parameters to the unified valuation interface and uses full-truncation Euler dynamics, stepwise deterministic NSS forwards, risk-neutral jump compensation and a stationary exponential Hawkes intensity.

The authoritative run is `20260904T130000Z-team8-refresh-v4`. It uses 8,192 paths, 260 fine steps, 20 operating quarters and one common WACC shock matrix. The Team 4 production/cost vectors and the unified DCF/equity bridge remain identical across all four engines.

## Interpretation boundary

Full Bates--Hawkes is primary because it has the lowest rolling OOS IV RMSE. Heston remains the current-snapshot in-sample winner. The transfer from GLD risk-neutral shape to gold USD/oz is conditional and does not establish a physical forecast, fair value or target price.
