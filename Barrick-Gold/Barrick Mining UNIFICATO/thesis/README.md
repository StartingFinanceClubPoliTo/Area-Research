# Barrick unified thesis — working draft

Main file: `Articolo.tex`

Engine: pdfLaTeX

Release date: **TBD**

The self-contained Research working draft now integrates September calibration, dense OOS diagnostics and the conditional corporate valuation. Historical August examples remain explicitly dated; they are not current calibration evidence.

The current executable Team 8/Barrick experiment is maintained in the [Research companion](https://github.com/StartingFinanceClubPoliTo/Research/tree/main/Barrick-Gold/Barrick%20Mining%20UNIFICATO/code):

- snapshot date September 2, 2026;
- authoritative run `20260904T130000Z-team8-refresh-v4`;
- 605 eligible calls before CC64 sampling;
- Full Bates–Hawkes best rolling OOS IV RMSE at 65.0507 bp;
- Barrick market reference USD 44.13.

Heston has the best current in-sample IV RMSE. Full Bates-Hawkes leads the four-model dense OOS ranking, but no model beats observed-IV persistence. Model targets contain 4,667 observations; persistence comparisons have 4,026. The Treasury curve is a par-yield proxy, not a bootstrapped zero curve.

Compile `Articolo.tex` with pdfLaTeX. Upload `chapters/`, `img/` and the Research cover with the main file. Build by-products and continuity registers stay outside Overleaf. The measure transfer, accounting/equity bridge and corporate assumptions remain qualified research limitations, not a finished valuation product.
