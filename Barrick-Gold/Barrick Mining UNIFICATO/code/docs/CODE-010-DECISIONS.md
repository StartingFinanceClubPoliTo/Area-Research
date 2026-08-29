# CODE-010 decisions

## D1 — Four models change only the gold-price layer

Production and costs are Team 4 ad hoc operating models. DCF, WACC and equity
bridge are Team 5 adaptations. The GBM/NSS block in v3 was a temporary gold
driver, not Team 4's production or cost model.

## D2 — Conditional GLD/Q shape bridge, not a physical forecast

The frozen Team 8 option parameters are the 2026-08-12T20:15:07.826914+00:00
LSE snapshot with a 2026-07-24 curve. They are applied to a common USD/oz
scenario start and common NSS drift solely as a controlled conditional
comparison. No recalibration, network refresh, GLD level conversion or
unvalidated Q-to-P inference is performed.

## D3 — Preserve Team 8 numerical engines

Import and call the frozen generators. Use their original five-year/260-step
grid and resample to quarterly operating dates. Do not copy, simplify or edit
their equations.

## D4 — Common WACC random numbers

Gold and WACC use separate seeds. The WACC shock matrix is generated once and
passed to all four valuation calls. This isolates differences to the gold
engine while preserving the v3 operating and DCF equations.

## D5 — Simulation size

Use 8,192 paths: a power of two suited to the frozen scrambled-Sobol engines,
twice the Team 8 publication diagnostic size, while keeping event-driven
Hawkes runtime and memory bounded. Quantiles remain research sensitivities.

## D6 — Full Bates--Hawkes selection language

Full Bates--Hawkes is the preferred structural/current-surface-fit scenario
because its in-sample IV RMSE is 61.0173 bp versus 78.1254 (Bates), 94.4391
(Heston) and 117.9272 (BS). It is not labelled the OOS predictive winner.

