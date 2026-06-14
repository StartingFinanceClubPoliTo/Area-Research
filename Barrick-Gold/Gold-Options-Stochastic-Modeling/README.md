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
| `Hawkes.py` | Exponential, rough power-law, and exact affine Hawkes calibration classes, diagnostics, and simulation. |
| `Hawkes Calibration.ipynb` | Thin notebook for fitting both Hawkes kernels and producing comparison plots. |
| `workflow.md` | Hawkes event-data, calibration, model-selection, and diagnostics workflow. |
| `BatesHawkes.py` | Stationary-intensity proxy benchmark. |
| `BatesHawkesExact.py` | Full affine Heston-Hawkes characteristic function and COS/Fourier pricing engine. |
| `Data/` | GLD option inputs, calibrated parameters, metrics, and generated figures. |
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
model implementations.

Run the Hawkes tests from the repository root with:

```bash
python tests/test_hawkes_calibration.py
python tests/test_hawkes_exact.py
```

## Interpretation

The GLD call surface used here prefers the Bates boundary when self-excitation is left unconstrained. The reported full calibration therefore enforces and discloses a minimum branching ratio of `0.02`, while also reporting the unconstrained Bates-limit objective. This is an identification result, not evidence that Hawkes clustering always improves option-surface fit.

The generated five-year Bates-Hawkes paths use the calibrated Heston variance state and event-driven Hawkes jump arrivals. Separate figures show stochastic volatility, Hawkes jump intensity/counts, and Bates Poisson intensity/counts on readable scales.

The material is for research and education only and does not provide investment advice.
