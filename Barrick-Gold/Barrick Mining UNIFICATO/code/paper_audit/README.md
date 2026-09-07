# Barrick Paper reproducibility 🔬

Audited operating-proxy sensitivity experiment supporting the September 7, 2026 Barrick Paper. Outputs are aggregate USD-billion proxies, not equity values, physical probabilities or market mispricing estimates.

## 👥 Authors
Research Teams 1–8, Starting Finance Club PoliTo. The full named author list is in the accompanying paper. Team 8 option models: Salvatore Gabriele Messina and Alessandro Coco.

## 🗂️ Structure
- `pricing/`: corrected Team 8 calibration and pricing modules.
- `src/`: operating-proxy and gold-path integration.
- `team8_data/data/processed/`: curated source data, with provenance in `DATA_MANIFEST.json`.
- `audit_inputs/`: frozen September 2 calibration and original rolling OOS inputs.
- `data/processed/team8/`: original and separately audited parameters.
- `audit_outputs/`: numerical audit tables, figures and samples.
- `tests/`: regression tests for candidate preservation and nested boundaries.

## ▶️ Run
From this directory, use Python 3.11 or later:

```sh
python -m pip install -r requirements.txt
python main.py
python -m pytest tests -q
```

`python main.py --recalibrate` also repeats the Hawkes local refinement. The default replays its saved audited candidate. Refinements are deterministic local improvements, not proofs of global minima. Input hashes are verified before every run.

## 📊 Interpretation
The current-snapshot refit and the historical OOS experiment are distinct. Historical OOS results retain the original origin parameters: changing the final-date fit does not retroactively validate a new calibration protocol. `oos_audit.csv` gives exact supports, denominators and two different RMSE aggregations. The bootstrap is exploratory with 30 dates.

The reported signed terminal preserves negative operating outcomes. `legacy_positive` is retained only to replay and quantify the earlier floor. `none` truncates at five years and is not a reserve-life estimate. Q-law plus assumed WACC remains a sensitivity operator; no accounting reconciliation or Q-to-P transformation is claimed. Calendar entries are frozen scenario inputs, not forecasts commencing at the option snapshot.

The option inputs were collected through the IB API; Treasury histories are separately identified. No account configuration, keys, private runtime files or environments are distributed. The user-provided curated dataset is copied without rewriting source observations.

## 📚 Audit
See `technical_audit.pdf`, `audit_outputs/review_status.json`, parameter JSON files and the tabular numerical outputs. New historical recalibration and an independently reconciled corporate valuation remain separate research tasks.
