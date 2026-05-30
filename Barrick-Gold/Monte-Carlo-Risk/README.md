# Monte Carlo Methods and Portfolio Simulation 🎲

📄 Article PDF: https://sfclubpolito.it/pdf-viewer.html?file=Pubblicazioni%2FMonte%20Carlo%20Methods%20and%20Portfolio%20Simulation.pdf&lang=it

Code companion for the Starting Finance Club PoliTo Research article *Monte Carlo Methods and Portfolio Simulation: Theory, Convergence, and Applications to Risk Analysis*.

## 👥 Authors

- [Andrea Rostagno](https://github.com/Andrea-Rostagno)
- [Francesco Florio](https://github.com/Francesco-Florio)

## 🎯 Purpose

This folder turns the original notebook workflow into reusable Python modules for random-number generation, stochastic process simulation, Monte Carlo option pricing, variance reduction, non-Gaussian returns, and portfolio risk metrics.

## 🧠 Methodology

The codebase is organized around compact, reproducible examples:

- pseudo-random and transformed random-variable diagnostics;
- geometric Brownian motion and jump-diffusion path simulation;
- European, Asian, and barrier option pricing with Monte Carlo methods;
- variance-reduction techniques such as antithetic variates, control variates, stratified sampling, and importance sampling;
- Gaussian-mixture returns, VaR, CVaR, and terminal-wealth risk analysis.

## 🗂️ Structure

```text
Monte-Carlo-Risk/
|-- README.md
|-- requirements.txt
|-- src/
|-- examples/
|-- figures/
`-- outputs/
```

## ⚙️ Setup

```bash
python -m venv .venv
pip install -r requirements.txt
```

On Windows, activate with `.venv\Scripts\activate`; on macOS/Linux, use `source .venv/bin/activate`.

## ▶️ Run

From the project root:

```bash
python examples/run_reproducibility.py
```

## 📊 Outputs

| Output | Description |
| --- | --- |
| `figures/lcg_uniform_histogram.png` | Uniform pseudo-random diagnostic. |
| `figures/box_muller_qq_plot.png` | Normality diagnostic for the Box-Muller transform. |
| `figures/european_call_convergence.png` | European call Monte Carlo convergence. |
| `figures/asian_call_mc_vs_qmc.png` | Standard Monte Carlo versus quasi-Monte Carlo comparison. |
| `figures/terminal_wealth_distribution.png` | Simulated terminal-wealth distribution. |
| `outputs/monte_carlo_summary.csv` | Summary table for prices, intervals, variance reduction, VaR, CVaR, and portfolio risk metrics. |

## 🧪 Reproducibility Notes

- Random seeds are fixed inside the example script.
- The examples use synthetic inputs unless explicitly documented in code.
- Results can change if the number of paths, seeds, or model parameters are modified.
- The material is for educational and research purposes only and does not provide investment advice.

## 📚 Citation

Rostagno, A. and Florio, F., *Monte Carlo Methods and Portfolio Simulation*, Starting Finance Club PoliTo Research, GitHub repository, published 01/05/2026, accessed 30/05/2026.
