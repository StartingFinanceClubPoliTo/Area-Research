# Local LSE dataset reconstruction

`lse_dataset.py` fetches the current GLD call chain from London Strategic Edge,
maps it to the historical wide-chain columns, filters a calibration-compatible
surface, and selects a Chebyshev sample.

The original files in `Data/` are immutable baselines. New LSE responses are
written by default to `Data/lse_local/`, which is ignored by Git because the
LSE data licence does not permit redistribution.

## Run

Install the requirements, configure `LSE_API_KEY` in the environment, and run:

```powershell
python lse_dataset.py --max-dte 1000 --annual-rate 0.037
```

The local directory contains the wide chain, the full eligible surface, a
Chebyshev sample, metadata, and a build audit. The API supplies current chain
values and Greeks but no risk-free curve, so `--annual-rate` is an explicit
assumption. The documented current-chain endpoint has no historical `as_of`
argument; this workflow therefore creates a comparable current snapshot, not
an exact reconstruction of the 3 April 2026 baseline.
