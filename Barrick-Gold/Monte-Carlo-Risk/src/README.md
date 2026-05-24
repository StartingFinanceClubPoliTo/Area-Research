# Source Package

Reusable Python modules for the Monte Carlo risk article.

## Role

The `src/` folder contains the importable package used by `examples/run_reproducibility.py`. Each module keeps one analytical block separate so the examples can be inspected, reused, and tested without opening the original notebook.

## Modules

| Module | Role |
| --- | --- |
| `monte_carlo_risk/random_generators.py` | Pseudo-random generation, distribution transforms, multivariate normal sampling, and diagnostics. |
| `monte_carlo_risk/stochastic_processes.py` | Geometric Brownian motion and jump-diffusion path simulators. |
| `monte_carlo_risk/option_pricing.py` | Black-Scholes formulas, European option Monte Carlo, Asian options, quasi-Monte Carlo, and barrier-option examples. |
| `monte_carlo_risk/variance_reduction.py` | Control variates, antithetic variates, stratified sampling, and importance sampling estimators. |
| `monte_carlo_risk/returns.py` | Gaussian-mixture returns, empirical inverse CDF mapping, VaR, CVaR, and terminal-wealth utilities. |
| `monte_carlo_risk/__init__.py` | Package marker and public package description. |

## Usage

Use the root README for environment setup. The reproducibility script adds this folder to the Python path automatically:

```bash
python examples/run_reproducibility.py
```
