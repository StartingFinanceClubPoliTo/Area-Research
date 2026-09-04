# Stochastic Valuation of Barrick Mining — Team 8 refresh

Main file: `Articolo.tex`

Engine: pdfLaTeX

Release date: **TBD**

## Frozen evidence

- September 2, 2026 GLD surface: 605 eligible calls, 12 expiries, 146 strikes and DTE 79–653 before sampling.
- Fixed 8×8 Chebyshev–Chebyshev geometry: 64 distinct actual contracts, 8 expiries and 20 strikes.
- Same-date U.S. Treasury NSS fit: 14 tenors and 2.0608 bp RMSE.
- In-sample IV RMSE (bp): Black–Scholes 97.2107; Heston 49.4881; Bates 54.6273; Full Bates–Hawkes 52.9714.
- Date-equal rolling OOS IV RMSE (bp): 140.6528; 80.1964; 69.3055; 65.0507, respectively.
- Rolling validation: 31 dense dates, 30 origins and 4,667 common forecasts.
- Full Bates–Hawkes branching ratio: 0.556903.
- Barrick valuation run: `20260904T130000Z-team8-refresh-v4`, 8,192 paths, market reference USD 44.13.

## Scientific contract

The paper compares Black–Scholes/GBM, Heston, Bates–Poisson and Full Bates–Hawkes while holding the Team 4 operating layer and unified DCF/WACC/equity bridge fixed. Heston is the current-snapshot calibration winner; Full Bates–Hawkes is the rolling OOS winner and therefore the primary structural scenario. The option calibration is risk-neutral and its transfer to gold USD/oz is conditional.

Row-level option observations are excluded. The public executable companion stores source code, aggregate calibration/OOS evidence, immutable manifests and the authoritative valuation outputs under [`../../Barrick Mining UNIFICATO/code/`](../../Barrick%20Mining%20UNIFICATO/code/).

## Compile

```bash
pdflatex Articolo.tex
pdflatex Articolo.tex
```

The current result is eight A4 pages including cover and references. The full rendered document and the clean Overleaf package were checked on September 4, 2026.
