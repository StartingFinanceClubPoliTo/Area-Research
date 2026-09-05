# Barrick Mining PAPER 📄

Compact research paper for the integrated stochastic valuation of Barrick Mining Corporation. The paper is an **eight-page working draft**; release date: **TBD**.

## Current snapshot

- Team 8 option/Treasury date: September 2, 2026.
- Dense eligible surface: 605 GLD calls, 12 expiries, 146 distinct strikes, DTE 79–653.
- Calibration geometry: fixed CC64 sample of 64 actual contracts, 8 expiries and 20 strikes.
- Same-date NSS curve: 14 Treasury tenors, 2.0608 bp fit RMSE.
- In-sample IV RMSE winner: Heston, 49.4881 bp.
- Dense rolling OOS winner: Full Bates–Hawkes, 65.0507 bp date-equal IV RMSE over 30 origins and 4,667 common forecasts.
- Conditional valuation: 8,192 paths; Barrick close USD 44.13; cross-model medians USD 35.14–35.87.

The paper keeps the in-sample and out-of-sample rankings distinct and treats the GLD risk-neutral-to-gold transfer as a conditional modeling device, not a physical forecast or target price. The operating handoff deliberately retains a persistent cost trend and assumes no ex-ante benefit from future technology or efficiency gains; this is prudential relative to otherwise identical efficiency-improvement scenarios, not a mathematical lower bound.

The persistence benchmark uses 4,026 interpolation-domain observations, a different support from the 4,667 model targets; no candidate beats persistence. The paper also distinguishes option-implied jump clustering from causal news attribution and flags unreconciled finite reserves, replacement investment and closure costs.

## Structure

| Path | Role |
| --- | --- |
| `paper/Articolo.tex` | Main pdfLaTeX source; working-draft date is `TBD`. |
| `paper/Articolo.pdf` | Verified eight-page PDF, including cover and references. |
| `paper/sections/` | Ten modular paper sections. |
| `paper/img/` | Club logo and five clickable contact icons, with the Font Awesome license. |
| `paper/figures/` | The 21 Team 8, operating, DCF and valuation images actually referenced by the paper. |
| [`../Barrick Mining UNIFICATO/code/`](../Barrick%20Mining%20UNIFICATO/code/) | Executable companion, manifests, aggregate inputs and authoritative run. |

## Build

```bash
cd paper
pdflatex Articolo.tex
pdflatex Articolo.tex
```

No BibTeX step is required: the bibliography is embedded in `sections/07_references.tex`. Build by-products, nested archives and unused legacy figures are not committed.

## Authors

Stefano Falcione, Marco Fracca, Filippo Triassi, Giorgio Zoccatelli, Andrea Rostagno, Francesco Florio, Giacomo Scali, Jacopo Foralosso, Federico Vesco, Lorenzo Pietra, Bader Moussaif, Davide Sisto, Matteo Armando, Davide D'Amico, Pietro Weisz, Salvatore Gabriele Messina and Alessandro Coco.

## Citation

Barrick Gold Research Teams (2026), *Stochastic Valuation of Barrick Mining*, Starting Finance Club PoliTo Research, working draft, release date TBD.
