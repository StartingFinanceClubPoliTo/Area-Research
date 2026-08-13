# Local LSE dataset reconstruction

`lse_dataset.py` fetches the current GLD call chain, historical GLD option
bars, US Treasury yields, and daily GLD candles from London Strategic Edge. It
keeps source-native local schemas, builds coherent maturity-specific rate
curves, filters calibration-compatible option surfaces, and selects
deterministic Chebyshev samples.

LSE is the sole calibration-data source. Responses are written by default to
`Data/lse_local/`, which is ignored by Git because the LSE data licence does
not permit redistribution. No LSE row-level data are committed.

## Run

Install the requirements, configure `LSE_API_KEY` in the environment, and run:

```powershell
python lse_dataset.py --max-dte 1000 --history-start 2021-01-01
```

The local directory contains the source-native option chain, the full eligible
surface, a Chebyshev sample, the Treasury curve, GLD daily history, metadata,
and a build audit. The workflow calls the official SDK's `options()`,
`bond_yields()`, and `candles()` methods. The curve selects the latest date
shared by 1M, 2M, 3M, 6M, 1Y, 2Y, 3Y, and 5Y US Treasury tenors. LSE par yields
are converted to the documented continuously compounded proxy
`log(1 + yield_decimal)` and linearly interpolated by option maturity.

LSE `last_price` can represent a trade older than the current underlying
snapshot. It is retained locally for audit but is not a calibration target.
The canonical surface uses contracts updated within seven days and transforms
LSE IV into a coherent call price and vega through `BnS.py`, with one snapshot
spot, one maturity clock, `q=0`, and the maturity-specific LSE rate. The
outputs are:

- `gld_lse_chain.csv`
- `gld_lse_calibration_full.csv`
- `gld_lse_calibration_chebyshev.csv`
- `usd_treasury_curve.csv`
- `gld_daily_history.csv`
- `gld_lse_meta.json`
- `lse_build_audit.json`

The default `Main.ipynb -> main.py` workflow additionally exports the most
recent year of GLD option bars at daily frequency and a complete panel of all
eight Treasury tenors. These local-only files support two distinct tests:

- primary rolling validation: calibrate at `t`, forecast the next available
  trading date, then shift `t` forward;
- fixed-cutoff stress test: freeze parameters six months before the last
  historical date and retain the entire next six months as a holdout.

The rolling target is a 25-node grid fixed before evaluation, with moneyness
0.95--1.05 and maturity 90--270 days. The model never receives the next day's
spot: that spot is used only after the forecast to normalize the realized IV
surface. Every forecast rate curve is dated no later than its origin.

The current-chain endpoint has no historical `as_of` argument, so current
valuation runs remain snapshots. Historical validation uses the vault's GLD
option-bar export instead. Row-level files and checkpoints remain local-only
and may not be redistributed.
