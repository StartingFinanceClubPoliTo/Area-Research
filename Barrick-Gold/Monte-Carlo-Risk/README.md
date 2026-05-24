# Monte Carlo Methods and Portfolio Simulation

Code companion for the Starting Finance Club PoliTo Research article "Monte Carlo Methods and Portfolio Simulation: Theory, Convergence, and Applications to Risk Analysis".

## Purpose

This folder contains the publication-facing Python implementation for Article 4 in the Barrick Gold research track. The code reorganizes the original notebook into reusable modules for Monte Carlo estimation, stochastic process simulation, option pricing, variance reduction, mixture returns, and risk metrics.

## Structure

```text
Monte-Carlo-Risk/
|-- README.md
|-- requirements.txt
|-- .gitignore
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
|   `-- README.md
`-- outputs/
    `-- README.md
```

The local `Revisione/` folder stores the uploaded PDF and original notebook for traceability. It is intentionally ignored by Git and should not be uploaded.

## Setup

Create a virtual environment from this folder and install the dependencies:

```bash
python -m venv .venv
.venv\Scripts\activate
pip install -r requirements.txt
```

On macOS/Linux, activate the environment with `source .venv/bin/activate`.

## Usage

Run the reproducibility example:

```bash
python examples/run_reproducibility.py
```

The example adds `src/` to the Python path automatically, so no package installation step is required for local execution.

## Outputs

The reproducibility script writes:

- diagnostic charts to `figures/`;
- summary tables to `outputs/`.

Generated outputs are reproducible examples tied to the article methodology. They are not investment signals and should be regenerated when code or parameters change.

## Reproducibility Notes

- Random seeds are fixed inside the example script.
- Simulations use synthetic inputs unless explicitly documented otherwise.
- The original PDF and notebook are kept out of Git because the repository should contain source code, README files, and curated reproducibility material rather than uploaded publication artifacts.
- Results can vary if users change the number of simulated paths, random seeds, or model parameters.

## Authors

- Andrea Rostagno
- Francesco Florio

Reviewed by Salvatore Messina.

## Citation

Rostagno, A. and Florio, F., "Monte Carlo Methods and Portfolio Simulation: Theory, Convergence, and Applications to Risk Analysis", Starting Finance Club PoliTo Research, 2026.
