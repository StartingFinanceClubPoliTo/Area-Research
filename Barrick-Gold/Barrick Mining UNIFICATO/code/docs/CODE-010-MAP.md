# CODE-010 system map

Status: implementation contract, 2026-08-26.

## Boundary and data flow

1. `config/multimodel_valuation_20260826.json` selects the immutable v3 base
   operating/DCF contract and the four frozen Team 8 parameter JSON files.
2. `multimodel_valuation.py` loads the frozen Team 8 `path_simulation.py`
   module without modifying it. Black--Scholes/GBM, Heston, Bates--Poisson and
   Full Bates--Hawkes generate the **gold-price layer only**.
3. Every engine starts from the same USD 4,677/oz Team 4 scenario value and
   receives the same Team 4 NSS common scenario-drift curve. The Team 8
   option-implied GLD/Q parameters determine conditional distribution shape;
   this is not a validated physical-price forecast or an approved Q-to-P map.
4. The 260-step, five-year Team 8 grid is resampled at every thirteenth step
   to the 20 Team 4 operating quarters.
5. `simulate_valuation_from_gold_paths` applies the same Team 4 production and
   cost arrays and the same Team 5 tax/growth/ROIC/WACC/terminal/equity bridge
   to each gold matrix. One separately seeded WACC shock matrix is generated
   once and reused byte-for-byte across the four valuations.
6. `multimodel_reporting.py` writes aggregate quantiles, model summaries,
   Barrick/share figures and a hash-complete manifest. No path-level data is
   published.

## Components and ownership

- Frozen Team 8 source: `parity/sources/team-8/**`; read-only input.
- Stable v3: `outputs/valuation/20260825T143500Z-provisional-v3/**`; read-only.
- New source: package modules, runner, config and tests under `Github-Branch`.
- New run: versioned output/figure/manifest directories; never overwrite v3.

## Failure surfaces

- Wrong measure claim: prevented by explicit conditional-bridge metadata.
- Model contamination: prevented by common-input fingerprints and WACC array
  equality tests.
- Grid mismatch: rejected unless 260 steps divide exactly into 20 quarters.
- Parameter drift: every Team 8 JSON and generator dependency is hashed.
- Invalid paths: shape, finite-value and strict-positivity checks fail closed.
- Corporate overclaim: status remains provisional/non-target and all unresolved
  accounting/equity-bridge limitations remain serialized.

