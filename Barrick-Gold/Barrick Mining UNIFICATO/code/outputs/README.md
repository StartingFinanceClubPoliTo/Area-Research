# Generated outputs

Only `thesis/20260828T180000Z-unificato-publication/` is the current unified
publication run.

The other retained output families are not alternative thesis versions:

- `current/` contains dated LSE research snapshots;
- `figures/team-8/` contains the frozen-aggregate OOS diagnostic regenerated
  by the autonomous figure runner;
- `valuation/20260825T143500Z-provisional-v3/` is an immutable regression
  fixture whose byte identity is checked by the test suite;
- `valuation/20260826T111412Z-conditional-multimodel-v4-final/` and
  `verification/20260826T205500Z-code012-current-runtime-parity/` are paired
  historical parity fixtures, not current Barrick valuation outputs.

Do not cite regression or parity fixtures as the unified valuation. New
publication figures must be produced under a fresh `thesis/<run-id>/` by
`run_all_thesis_figures.py` and pass the provenance audit before promotion.
