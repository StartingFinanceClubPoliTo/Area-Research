# Barrick Stochastic Valuation Paper - Team 8 refresh

Scientific paper for the integrated Barrick Mining stochastic-valuation project. The publication date remains **TBD** while the corporate bridge and final editorial review are unfinished.

## Current evidence snapshot

- Team 8 option and Treasury snapshot: 2 September 2026.
- Dense GLD surface before sampling: 605 eligible calls, 12 expiries, 146 distinct strikes, DTE 79--653.
- Official calibration geometry: fixed 8x8 Chebyshev--Chebyshev sample of 64 distinct actual contracts, spanning 8 expiries and 20 strikes.
- U.S. Treasury input: same-date NSS fit on 14 tenors, RMSE 2.0608 bp against the continuously compounded par-yield proxy.
- In-sample IV RMSE winner: Heston, 49.4881 bp.
- Dense-only rolling OOS winner: Full Bates--Hawkes, 65.0507 bp date-equal IV RMSE over 30 origins and 4,667 common forecasts.
- Hawkes branching ratio: 0.556903, stationary and economically material.
- Barrick conditional valuation: 8,192 paths; 2 September close USD 44.13; all four medians lie between USD 35.14 and USD 35.87.

## Implemented model stack

- Black--Scholes/GBM, Heston, Bates--Poisson and Full Bates--Hawkes;
- vega-scaled option calibration with admissibility constraints;
- fixed CC64 geometry and dense-only next-available-date rolling validation;
- Heston variance and Hawkes intensity state projection to the target date;
- Team 4 Q1--Q2 2026 actuals plus the frozen Q3 2026--2030 operating forecast;
- one common Team 5-style DCF, WACC shock array and equity bridge across all four gold engines.

The companion implementation and versioned run are maintained in the [Research repository](https://github.com/StartingFinanceClubPoliTo/Research/tree/main/Barrick-Gold/Barrick%20Mining%20UNIFICATO/code). Row-level option observations are intentionally excluded from the publication package.

## Scientific interpretation

The paper is a controlled attribution exercise. Heston is the current-snapshot calibration winner, whereas Full Bates--Hawkes is the rolling OOS winner. The GLD risk-neutral distributional shape is transferred to a separately sourced gold level without claiming a validated GLD-to-ounce or risk-neutral-to-physical mapping. Output distributions are research sensitivities, not fair values, confidence intervals, price targets or investment recommendations.

## Compile

```bash
pdflatex Articolo.tex
pdflatex Articolo.tex
```

The bibliography is frozen in `sections/07_references.tex`; no BibTeX step is required. For Overleaf, upload the project ZIP, choose `Articolo.tex` as the main document, and select pdfLaTeX.

The Overleaf upload ZIP contains 41 source/dependency files: two synchronized LaTeX entry points, ten included sections, the cover and logo, 21 scientific images and five contact icons with their Font Awesome license. The repository README and unused bibliography ledger are not part of that ZIP. Generated `.aux`, `.log`, `.out`, `.synctex`, nested ZIP and obsolete image files are excluded.

## Refresh revision - 6 September 2026

The Team 8 data, sampling, Treasury, calibration, residual, rolling OOS, Hawkes-intensity and Barrick valuation evidence use one synchronized snapshot. The article has seven body pages plus the cover, with unchanged font and margins and continuous two-column flow. The project/collaboration context is separate from the scientific introduction. Clickable institutional contacts appear before JEL classification. Figures 5 and 12 use independent vertical panels; all four Figure 6 residual maps fit vertically on body page 4, followed by the calibration RMSE table on page 5. Figure 13 stacks both benchmark charts. The cost assumption is prudential relative to otherwise identical efficiency-improvement scenarios, not a mathematical valuation lower bound. The article remains a working draft with release date TBD.

The expanded discussion clarifies conditional next-date scoring, margin units and the accounting bridge, economic dispersion versus Monte Carlo precision, and the interpretation of negative proxy tails. The final page develops layered validation, filing reconciliation, dependent operating risks, finite-resource closure and a sequenced research agenda. These are explanations and validation priorities drawn from the unified study, not newly completed empirical tests or accounting reconciliations.
