# Local LSE dataset reconstruction

`lse_dataset.py` fetches the current GLD call chain from London Strategic Edge,
keeps a source-native local schema, filters a calibration-compatible surface,
and selects a Chebyshev sample.

LSE is the sole calibration-data source. Responses are written by default to
`Data/lse_local/`, which is ignored by Git because the LSE data licence does
not permit redistribution. No LSE row-level data are committed.

## Run

Install the requirements, configure `LSE_API_KEY` in the environment, and run:

```powershell
python lse_dataset.py --max-dte 1000 --annual-rate 0.037
```

The local directory contains the source-native chain, the full eligible
surface, a Chebyshev sample, metadata, and a build audit. The API supplies
current chain values, IV, and Greeks but no risk-free curve, so
`--annual-rate` is an explicit assumption.

LSE `last_price` can represent a trade older than the current underlying
snapshot. It is retained locally for audit but is not a calibration target.
The canonical surface uses contracts updated within seven days and transforms
LSE IV into a coherent call price and vega through the protected `BnS.py`, with
one snapshot spot, one maturity clock, `q=0`, and the stated flat rate. The
outputs are:

- `gld_lse_chain.csv`
- `gld_lse_calibration_full.csv`
- `gld_lse_calibration_chebyshev.csv`
- `gld_lse_meta.json`
- `lse_build_audit.json`

The documented current-chain endpoint has no historical `as_of` argument;
each run is a new current snapshot.
