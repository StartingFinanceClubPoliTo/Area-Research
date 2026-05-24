# Monte Carlo Methods and Portfolio Simulation

| Field | Value |
| --- | --- |
| Project | [Barrick Gold](../) |
| Article | 4 |
| Authors | Andrea Rostagno, Francesco Florio |
| Reviewer | Salvatore Messina |
| Published | 2026 |

Code companion for the Starting Finance Club PoliTo Research article *"Monte Carlo Methods and Portfolio Simulation: Theory, Convergence, and Applications to Risk Analysis"*.

## Purpose

This folder contains the publication-facing Python implementation for Article 4 in the Barrick Gold research track. The project turns the original notebook workflow into reusable modules for random-number generation, stochastic process simulation, Monte Carlo option pricing, variance reduction, non-Gaussian returns, and portfolio risk metrics.

## Methodology

The codebase is organized around compact, reproducible examples:

- pseudo-random and transformed random-variable diagnostics;
- geometric Brownian motion and jump-diffusion path simulation;
- European, Asian, and barrier option pricing with Monte Carlo methods;
- variance-reduction techniques such as antithetic variates, control variates, stratified sampling, and importance sampling;
- Gaussian-mixture returns, VaR, CVaR, and terminal-wealth risk analysis.

## Structure

```text
Monte-Carlo-Risk/
|-- README.md
|-- requirements.txt
|-- src/
|   `-- monte_carlo_risk/
|       |-- __init__.py
|       |-- random_generators.py
|       |-- stochastic_processes.py
|       |-- option_pricing.py
|       |-- variance_reduction.py
|       `-- returns.py
|-- examples/
|   |-- README.md
|   `-- run_reproducibility.py
|-- figures/
|   |-- README.md
|   |-- asian_call_mc_vs_qmc.png
|   |-- box_muller_qq_plot.png
|   |-- european_call_convergence.png
|   |-- lcg_uniform_histogram.png
|   `-- terminal_wealth_distribution.png
`-- outputs/
    |-- README.md
    `-- monte_carlo_summary.csv
```

The local `Revisione/` folder stores uploaded review artifacts for traceability. It is ignored by Git and is not part of the publication-facing repository.

## Files

| Path | Role |
| --- | --- |
| `src/monte_carlo_risk/random_generators.py` | Pseudo-random generation, distribution transforms, multivariate normal sampling, and diagnostics. |
| `src/monte_carlo_risk/stochastic_processes.py` | GBM and jump-diffusion path simulators. |
| `src/monte_carlo_risk/option_pricing.py` | Black-Scholes formulas, Monte Carlo pricing, Asian options, quasi-Monte Carlo, and barrier-option examples. |
| `src/monte_carlo_risk/variance_reduction.py` | Control variates, antithetic variates, stratified sampling, and importance sampling. |
| `src/monte_carlo_risk/returns.py` | Gaussian-mixture returns, empirical inverse CDF mapping, VaR, CVaR, and terminal-wealth utilities. |
| `examples/run_reproducibility.py` | End-to-end script that regenerates the curated figures and CSV summary table. |

## Setup

Create and activate a virtual environment from this folder:

```bash
python -m venv .venv
```

On Windows:

```bash
.venv\Scripts\activate
```

On macOS/Linux:

```bash
source .venv/bin/activate
```

Install the required packages:

```bash
pip install -r requirements.txt
```

## How To Reproduce

Run the reproducibility script from the project root:

```bash
python examples/run_reproducibility.py
```

The script adds `src/` to the Python path automatically, so no editable package installation is required for local execution.

## Outputs

| Output | Description |
| --- | --- |
| `figures/lcg_uniform_histogram.png` | Uniform pseudo-random diagnostic from the linear congruential generator example. |
| `figures/box_muller_qq_plot.png` | Normality diagnostic for the Box-Muller transform. |
| `figures/european_call_convergence.png` | Monte Carlo convergence for a European call option. |
| `figures/asian_call_mc_vs_qmc.png` | Standard Monte Carlo versus quasi-Monte Carlo Asian call comparison. |
| `figures/terminal_wealth_distribution.png` | Simulated terminal-wealth distribution for portfolio risk analysis. |
| `outputs/monte_carlo_summary.csv` | Representative summary table for prices, confidence intervals, variance-reduction estimates, return moments, VaR, and CVaR. |

## Reproducibility Notes

- Random seeds are fixed inside the example script.
- The examples use synthetic inputs unless explicitly documented in code.
- Results can change if the number of paths, seeds, or model parameters are modified.
- The original uploaded PDF and notebook are intentionally kept out of Git; this folder contains the cleaned source code and curated reproducibility outputs.
- The analysis is for educational and research purposes only and does not provide investment advice.

## Authors

- Andrea Rostagno
- Francesco Florio

Reviewed by Salvatore Messina.

## Citation

Rostagno, A. and Florio, F., *Monte Carlo Methods and Portfolio Simulation: Theory, Convergence, and Applications to Risk Analysis*, Starting Finance Club PoliTo Research, 2026.
