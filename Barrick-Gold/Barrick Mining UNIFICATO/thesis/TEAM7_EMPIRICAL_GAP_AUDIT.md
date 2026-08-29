# Team 7 empirical gap audit

Date: 27 August 2026  
Target: `chapters/07-gold-econometrics.tex`  
Source reviewed: `Team 7 - Gold Volatility Dynamics/Overleaf/Articolo.tex` and
its paired `Github-Branch` outputs.

## Classification result

### A. Restored in the thesis chapter

The following evidence was present in the standalone Team 7 article but had
been reduced to prose or omitted from the first unified Chapter 7:

- full descriptive statistics for the 4,694 synchronized observations;
- the static correlation heatmap and all six 252-day rolling-correlation
  charts for gold and silver versus the 10-year yield change, S&P 500 and DXY;
- all six original OLS result sets, consolidated into one gold and one silver
  table with HC1 standard errors;
- AR(1) level/return estimates, AR(p) diagnostics and ARMA/ARIMA selections;
- higher moments, histograms, normal Q--Q plots, rolling-volatility charts and
  the return/absolute-return/squared-return ACF grid;
- absolute-return persistence regressions;
- GARCH(1,1) parameters, conditional-volatility charts and the complete
  1--10-day conditional-variance forecast sequence;
- GPH long-memory estimates, the 1--100-lag absolute-return ACF and the two
  fractional-differencing illustrations.

Nineteen article PNGs are now frozen under `img/team7_empirical/` and are
referenced from Chapter 7. The TeX tables were transcribed and consolidated so
the chapter remains self-contained.

### B. Preserved as reproducibility evidence, not printed in the main text

The Team 7 repository contains four raw/processed data CSVs, 24 result CSVs,
15 code-output PNGs and 13 generated TeX tables. The main chapter reports the
economically relevant summaries, while the following high-volume material
remains repository-level evidence:

- complete ARMA and ARIMA order grids;
- the 4,694-row fitted GARCH conditional-volatility series;
- raw Yahoo downloads and intermediate aligned panels;
- coefficient exports duplicating the consolidated thesis tables.

This category is not an information deletion: it separates reader-facing
evidence from machine-facing replication output.

### C. Diagnostic-only figures not promoted to the thesis

Three code figures do not occur in the published Team 7 article and remain
diagnostic-only: `quality_check_levels_returns.png`,
`quality_check_ust10y_level.png`, and `quality_check_ust10y_change.png`. They
check transformations and units but add no independent empirical conclusion
once the sample definition and descriptive table are reported.

### D. Content already relocated elsewhere in the unified structure

- monetary regimes, gold history and the debasement-trade narrative are in
  Chapter 2;
- the CIR/Heston bridge, stochastic volatility and jumps are in Chapter 8;
- option-surface fitting and simulation are in Chapter 9;
- operating simulations and Barrick valuation are in Chapters 10--13.

These items were not missing from the project; they were deliberately moved to
avoid repeating theory or treating completed downstream work as future work.

## Interpretation and version caveats

- The historical Yahoo sample and the current LSE option sample remain
  separate evidence layers and are not concatenated.
- The article snapshot and current repository CSVs show minor version drift in
  some AR(1) robust standard errors. Chapter 7 labels its AR(1) table as the
  published article snapshot rather than silently presenting it as a current
  recomputation.
- The GPH estimates for absolute returns exceed 0.5. They support very strong
  persistence but do not establish a covariance-stationary elementary ARFIMA
  process.
- GARCH is retained as a valid discrete-time volatility model. The reason for
  moving to Heston/Bates/Hawkes is the option-pricing and valuation objective,
  not a claim that ARIMA or GARCH are intrinsically invalid.

## Limitations/next-steps rewrite

The standalone article's generic “next steps” were split into:

1. limitations of the historical empirical layer;
2. developments already completed by Chapters 8--13 of the unified thesis;
3. a genuinely residual agenda, including same-date data reconstruction,
   asymmetric/heavy-tailed/regime-switching volatility, causal macroeconomic
   identification, robust long-memory testing, the P-to-Q bridge and augmented
   filtrations using heterogeneous macro/geopolitical data.
