# Advanced Stochastic Modeling for Gold Options

Supporting code for the Starting Finance Club PoliTo Research article on GLD option pricing, constrained calibration, and stochastic gold-price paths.

## Authors

- [0c0c](https://github.com/0c0c)
- [SalvatoreMessina11](https://github.com/SalvatoreMessina11)

## Purpose

The main model progression is Black-Scholes, Heston, Bates, and full affine Bates-Hawkes. The final Bates-Hawkes specification combines Heston stochastic variance with self-exciting price jumps and is calibrated directly to GLD call prices. Its exponential Hawkes kernel preserves a finite-dimensional Markov state and an affine characteristic function.

Two lighter Hawkes implementations remain as labelled method benchmarks:

- `BatesHawkes.py`: stationary-mean proxy, equivalent to Bates with a constant effective intensity.
- Constant-volatility routines in `BatesHawkesExact.py`: validation of the exact Hawkes jump transform without stochastic variance.

Neither benchmark replaces the full stochastic-volatility calibration used by the final diagnostics and simulations.

## Main Components

| Path | Role |
| --- | --- |
| `Main.ipynb` | Default one-cell project entry point; confirms the locally configured LSE key and delegates all work to Python modules. |
| `main.py` | Key-gated orchestration of current calibration, rolling/fixed historical validation, diagnostics, and Monte Carlo outputs. |
| `BnS.py` | Black-Scholes pricing, implied-volatility inversion, and vega. |
| `Heston.py` | Heston characteristic function, pricing, and calibration. |
| `Bates.py` | Heston stochastic variance plus constant-intensity lognormal jumps. |
| `calibration_core.py` | Validated immutable option-surface slices and serialisable calibration reports. |
| `fourier_pricing.py` | Shared Carr--Madan, Gil--Pelaez, and vectorised COS kernels. |
| `calibration_workflow.py` | Dataset loading and publication diagnostics used by thin notebooks. |
| `historical_validation.py` | No-look-ahead surface construction, OOS R-squared, Welch--Goyal loss differences, and HAC tests. |
| `online_validation.py` | Rolling `t -> t+1` calibration, state forecasting, model checkpoints, and deterministic result merging. |
| `path_simulation.py` | Deterministic GBM, Heston, Bates, and full Bates--Hawkes path engines. |
| `Hawkes.py` | Exponential, rough power-law, and exact affine Hawkes calibration classes, diagnostics, and simulation. |
| `Hawkes Calibration.ipynb` | Thin notebook for fitting both Hawkes kernels and producing comparison plots. |
| `workflow.md` | Hawkes event-data, calibration, model-selection, and diagnostics workflow. |
| `BatesHawkes.py` | Stationary-intensity proxy benchmark. |
| `BatesHawkesExact.py` | Full affine Heston-Hawkes characteristic function and COS/Fourier pricing engine. |
| `Data/` | Publication-safe aggregate parameters, metrics, manifests, and generated figures. |
| `Data/lse_local/` | Git-ignored current/historical LSE rows and rolling-validation checkpoints. |
| `lse_dataset.py` | Sole data-ingestion path: current and historical GLD options, US Treasury yields, and daily GLD candles from LSE. |
| `benchmarks/` | Deterministic scalar-versus-batch pricing benchmark. |
| `tools/` | Notebook and publication-output rebuild entry points. |
| `tests/test_hawkes_calibration.py` | Exponential and rough likelihood, fit, residual, and compatibility tests. |
| `tests/test_hawkes_exact.py` | Characteristic-function, limit, pricing, parity, and objective tests. |

Calibration logic belongs to the root model classes. Notebooks call those
classes and contain only data preparation, execution, diagnostics, and plots.

## Setup

```bash
python -m venv .venv
pip install -r requirements.txt
```

## Default run

Open `Main.ipynb` and execute its only code cell. It calls `main.py`, which
checks whether `LSE_API_KEY` is already configured on the computer and asks for
confirmation. If the key is absent or confirmation is declined, the run stops
before any network request or file modification. The command-line equivalent is:

```bash
python main.py
```

## Model notebooks

Open the notebook matching the model and run its cells in order:

- `Black and Scholes Calibration.ipynb`
- `Heston Calibration.ipynb`
- `Bates Calibration.ipynb`
- `Bates-Hawkes Proxy Calibration.ipynb`
- `Hawkes Calibration.ipynb`

The notebooks import the root `.py` modules; they do not define alternative
model implementations. Bates-family calibrations accept an optional random
seed and can return a `CalibrationReport` with objective values, status,
iterations, evaluations, elapsed time, and named parameters.

Run all regression tests and the pricing benchmark from the repository root:

```bash
python -m pytest tests -q
python benchmarks/benchmark_pricing.py
```

## LSE data build

LSE is the sole calibration-data source. Configure `LSE_API_KEY` and run:

```bash
python lse_dataset.py --max-dte 1000 --history-start 2021-01-01
```

The API's reported last trade may be older than its current underlying snapshot.
The workflow therefore calibrates to LSE implied volatility: it applies a
seven-day freshness filter, then reconstructs coherent call prices and vegas
with the Black--Scholes functions, the snapshot spot, `q=0`, and an
LSE-sourced USD Treasury curve interpolated by option maturity. Daily GLD
candles from LSE feed the empirical-return normality diagnostics. Raw and row-level LSE outputs stay under
the Git-ignored `Data/lse_local/`; only aggregate research outputs are committed.
See `LSE-DATASET.md` for the exact contract and licence boundary.

The primary predictive test is a rolling one-step-ahead exercise. On every
available date `t`, each model is recalibrated using only the surface and rate
curve known by `t`; current variance and Hawkes intensity are evolved to the
next trading date, and the forecast is scored on a fixed moneyness/maturity IV
grid. The four chronological model chains run in parallel Python processes,
while every chain remains sequential because `t-1` is used as the SLSQP warm
start at `t`. This is a computational initialization, not a Bayesian prior.
Surface caches and per-model checkpoints are bound to SHA-256 fingerprints of
the exact option, GLD, and rate inputs, so revised source data cannot silently
reuse stale rolling estimates.
The current snapshot calibration remains the separate input to today's
simulation/valuation layer.

## Interpretation

The calibration code enforces positivity, the Heston Feller condition,
correlation bounds, and Hawkes stationarity. Cross-sectional improvements are
reported separately from rolling predictive evidence. The latter is compared
with both a node-specific expanding mean and the previous day's observed IV
surface, with pairwise OOS R-squared, Welch--Goyal cumulative loss differences,
and Newey--West HAC tests.

The generated five-year paths use forward rates implied by the LSE Treasury
curve. The Bates-Hawkes paths combine calibrated Heston variance with
event-driven Hawkes jump arrivals. Separate outputs report model-wise return
percentiles 0--100, skewness, excess kurtosis, volatility states, and jump
intensity/counts.

The material is for research and education only and does not provide investment advice.
