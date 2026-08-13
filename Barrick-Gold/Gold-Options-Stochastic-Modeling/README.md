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
| `BnS.py` | Black-Scholes pricing, implied-volatility inversion, and vega. |
| `Heston.py` | Heston characteristic function, pricing, and calibration. |
| `Bates.py` | Heston stochastic variance plus constant-intensity lognormal jumps. |
| `calibration_core.py` | Validated immutable option-surface slices and serialisable calibration reports. |
| `fourier_pricing.py` | Shared Carr--Madan, Gil--Pelaez, and vectorised COS kernels. |
| `calibration_workflow.py` | Dataset loading and publication diagnostics used by thin notebooks. |
| `path_simulation.py` | Deterministic GBM, Heston, Bates, and full Bates--Hawkes path engines. |
| `Hawkes.py` | Exponential, rough power-law, and exact affine Hawkes calibration classes, diagnostics, and simulation. |
| `Hawkes Calibration.ipynb` | Thin notebook for fitting both Hawkes kernels and producing comparison plots. |
| `workflow.md` | Hawkes event-data, calibration, model-selection, and diagnostics workflow. |
| `BatesHawkes.py` | Stationary-intensity proxy benchmark. |
| `BatesHawkesExact.py` | Full affine Heston-Hawkes characteristic function and COS/Fourier pricing engine. |
| `Data/` | Publication-safe aggregate parameters, metrics, manifests, and generated figures. |
| `Data/lse_local/` | Git-ignored current LSE chain and sampled calibration rows. |
| `lse_dataset.py` | Sole data-ingestion path: current GLD options from the LSE API. |
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

## Calibration Notebooks

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
python lse_dataset.py --max-dte 1000 --annual-rate 0.037
```

The API's reported last trade may be older than its current underlying snapshot.
The workflow therefore calibrates to LSE implied volatility: it applies a
seven-day freshness filter, then reconstructs coherent call prices and vegas
with the protected Black--Scholes functions, the snapshot spot, `q=0`, and the
explicit 3.7% flat-rate assumption. Raw and row-level LSE outputs stay under
the Git-ignored `Data/lse_local/`; only aggregate research outputs are committed.
See `LSE-DATASET.md` for the exact contract and licence boundary.

## Interpretation

The calibration code enforces positivity, the Heston Feller condition,
correlation bounds, and Hawkes stationarity. Cross-sectional improvements are
reported as in-sample evidence only, not as proof of time-series clustering or
out-of-sample dominance.

The generated five-year Bates-Hawkes paths use the calibrated Heston variance state and event-driven Hawkes jump arrivals. Separate figures show stochastic volatility, Hawkes jump intensity/counts, and Bates Poisson intensity/counts on readable scales.

The material is for research and education only and does not provide investment advice.
