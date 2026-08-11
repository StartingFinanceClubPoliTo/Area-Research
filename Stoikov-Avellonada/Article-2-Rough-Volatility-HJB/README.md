# Rough Volatility and End-of-Day Inventory Control

Companion code for the Starting Finance Club PoliTo Research article on rough-volatility market making and liquidation-aware inventory control.

## Purpose

The repository reproduces the numerical material used in the article: fractional Brownian motion diagnostics, causal Volterra simulations, Markovian lifting experiments, and the reduced HJB market-making comparison.

## Structure

- `src/rough_processes.py`: validated, cached classes for fBM, Volterra processes, and Markovian lifting; the original functional API remains available.
- `src/dashboard_base.py`: shared slider, redraw, and colorbar lifecycle for all dashboards.
- `src/generate_article_figures.py`: class-based static figure generator for article-style diagnostics.
- `src/interactive_fbm_dashboard.py`: Matplotlib dashboard with sliders for fBM parameters.
- `src/interactive_volterra_dashboard.py`: Matplotlib dashboard with sliders for Volterra causality diagnostics.
- `src/interactive_lift_dashboard.py`: Matplotlib dashboard with sliders for the lifted OU approximation.
- `src/solve_rough_hjb.py`: reduced HJB solver, reusable market environment, policy simulator, and experiment orchestrator.
- `tests/test_numerics.py`: deterministic regression, API compatibility, shared-environment, validation, and dashboard lifecycle tests.
- `img/`: reference article figures.
- `output/`: generated CSV, JSON, HTML, or PNG outputs.

## Setup

Use Python 3.10 or later.

```powershell
python -m venv .venv
.\.venv\Scripts\python -m pip install -r requirements.txt
```

## Usage

Regenerate article-style figures:

```powershell
python src/generate_article_figures.py --output output/generated_figures
```

Open the interactive dashboards:

```powershell
python src/interactive_fbm_dashboard.py
python src/interactive_volterra_dashboard.py
python src/interactive_lift_dashboard.py
```

Run the reduced HJB experiment:

```powershell
python src/solve_rough_hjb.py
```

Run the automated checks:

```powershell
python -m unittest discover -s tests -v
```

## Outputs

The HJB script writes `hjb_summary.json` and figure files. The static figure generator writes PNG files into the selected output folder. Interactive scripts open local Matplotlib windows with slider controls and do not overwrite article figures.

## Design and performance

Deterministic matrices are cached inside the model instances. Markovian-lift paths are batched across simulations, with only the time recursion left in Python. The Monte Carlo comparison constructs the policy-independent lifted state, volatility, and mid-price once per shock stream and shares them between the naive and HJB policies. Interactive redraws remove their old colorbars before adding new ones, so figure axes do not accumulate.

The automated regression suite checks that these optimizations preserve the numerical results. The experiments are stylized and data-free: they illustrate model mechanics rather than calibrate live market data.

## Project notes

- The article is scheduled for publication on June 26, 2026.
- All simulations use deterministic random seeds by default.
- The material is for educational and research purposes only and does not provide investment advice.

## Authors

Salvatore Gabriele Messina; Niccolò Soriano.

## Citation

Messina, S. G. and Soriano, N., *Rough Volatility and End-of-Day Inventory Control: supporting code*, Starting Finance Club PoliTo Research, GitHub repository, expected publication 26/06/2026.
