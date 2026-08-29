# Stochastic Valuation Paper Template — v4

Scientific-paper draft for condensing the Barrick Mining Corporation stochastic valuation project into at most 10 A4 pages.

## What is now source-grounded
- four gold engines: Black-Scholes/GBM, Heston, Bates-Poisson and Full Bates-Hawkes;
- exact Full Bates-Hawkes SDE plus the affine characteristic-function / Riccati system actually used by the supplied Team 8 implementation;
- vega-scaled calibration loss, admissibility constraints and separation of in-sample calibration from one-step-ahead OOS validation;
- frozen 12 August valuation calibration versus the separate 26 August Chebyshev sampling audit;
- Team 4 cost model: log-space ARIMA(3,1,0) with drift, AIC/BIC selection, residual diagnostics and 12-period hold-out evidence;
- Team 4 production model: physical decomposition, SARIMA specifications for ore and grade, bounded recovery process, and explicit note that AIC/BIC selection is not documented for these production orders;
- Team 5 stochastic DCF mapping and the current reconciliation limits;
- primary valuation distribution, cross-model comparison and OOS option-model evidence.

## Scientific compression rule
The paper is not a compressed chapter-by-chapter thesis. Keep equations that define the implemented models, tests that justify or reject them, one primary valuation distribution, one compact cross-model comparison and the interpretation-changing limitations. Remove duplicated audit tables and descriptive histories first.

## Project skill
Use `skills/stochastic-valuation-paper/SKILL.md` for every new source batch. The skill requires explicit source status (`accepted`, `diagnostic`, `placeholder`, `superseded`, `unverified`) and prevents conflicting snapshots from being silently merged.

## Compile
```bash
pdflatex Articolo.tex
pdflatex Articolo.tex
```

The bibliography is frozen in `main.bbl`; `references.bib` remains the editable source ledger.


Updated v5: added cover page, compact residual diagnostics, Welch--Goyal figure, jump-intensity comparison, multimodel distribution overlay, and affiliations for Universita di Torino and Universita di Padova.


V7 layout pass: three render/inspect/reflow iterations. Keywords removed; JEL moved to the left metadata column and updated to G12, G13, G31, C32, C53, C61, C63. Section headings forced ragged-right, centering leakage removed, operating figures rebalanced, OOS diagnostics condensed, bibliography pulled onto the final page, and the paper reduced to 7 pages including the cover.


Updated v8: introduction expanded to describe the thesis-style pedagogical companion, interdisciplinary rationale, and the distinct roles of Teams 1-8 in the integrated Barrick research architecture.

## v9 layout revision (28 August 2026)
- First research page masthead simplified to date at top-left and SFC PoliTo logo at top-right.
- Footer simplified globally to centered page number only.
- Section 5 receives explicit vertical spacing to prevent crowding with the preceding operating-evidence block.
- Section 7 rebuilt as two equal-width minipage columns for a true half-page / half-page composition.


### v10 layout micro-adjustments
- Added 0.5 cm before the Evidence status box in Section 5.1.
- Lowered Table 3 to align it horizontally with Table 4.
- Increased separation between Tables 3/4 and the following multimodel distribution figure.

## v11 handoff revision (29 August 2026)

- Figure 6 is now a readable 2×2 comparison: two residual heatmaps per row.
- The paper compiles to 9 A4 pages, including cover and references, with no
  errors, undefined references, overfull boxes or underfull boxes.
- The final project is packaged in the SF five-folder layout; build products
  and visual-QA files stay outside this `Overleaf/` source folder.
- The quantitative companion is `../../../UNIFICATO/Github-Branch/`; its
  handoff interface reports `READY` and the complete suite passes 60/60 tests.
