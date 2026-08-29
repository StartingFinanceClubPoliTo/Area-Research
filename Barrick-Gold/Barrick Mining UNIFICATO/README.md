# Barrick Mining UNIFICATO 🪙

📄 Article PDFs are published on the Starting Finance Club PoliTo website: https://sfclubpolito.it/pubblicazioni. This GitHub repository contains code, data notes, and reproducibility assets.

Unified thesis and reproducibility companion for the Starting Finance Club PoliTo Research project on stochastic valuation of Barrick Mining Corporation.

> Educational and research material only. Model outputs are conditional
> sensitivities, not fair values, target prices or investment recommendations.

## 👥 Authors

Stefano Falcione, Marco Fracca, Filippo Triassi, Giorgio Zoccatelli, Andrea
Rostagno, Francesco Florio, Giacomo Scali, Jacopo Foralosso, Federico Vesco,
Lorenzo Pietra, Bader Moussaif, Davide Sisto, Matteo Armando, Davide D'Amico,
Pietro Weisz, Salvatore Gabriele Messina and Alessandro Coco.

## 🎯 Purpose

The project connects four stochastic gold-price engines to a common operating
forecast, DCF contract and equity bridge. It keeps calibration, provenance,
units, dates and unresolved corporate inputs explicit across the full workflow.

## 🗂️ Structure

| Path | Role |
| --- | --- |
| `thesis/` | Complete LaTeX thesis project and compiled `Articolo.pdf`. |
| `code/` | Class-based Python package, curated notebook, tests, public inputs, figures, tables and manifests. |

Licensed LSE row-level snapshots are intentionally excluded. Their manifests,
scope notes and redistribution-safe processed inputs remain available under
`code/data/`.

## ⚙️ Setup

```bash
cd code
python -m venv .venv
python -m pip install -e ".[test]"
```

## ▶️ Run

```bash
python main.py status
python main.py test -q
```

The status command audits the handoff structure and table links. Established
figure and valuation runners are documented in `code/README.md`.

## 📊 Outputs

The authoritative publication run is stored under
`code/outputs/thesis/20260828T180000Z-unificato-publication/`. Analytical
figures have PNG/PDF companions and machine-readable CSV/JSON sidecars.

## 📚 Citation

Barrick Gold Research Teams (2026), *Stochastic Valuation of Barrick Mining:
Unified Thesis and Supporting Code*, Starting Finance Club PoliTo Research.
