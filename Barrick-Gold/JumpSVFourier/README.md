# Beyond Black-Scholes Synthetic Examples

This folder provides reproducible synthetic examples for the Starting Finance Club PoliTo Research article on jump processes, stochastic volatility, Fourier methods, and Hawkes dynamics.

## Purpose

The examples connect the theoretical discussion to small simulated illustrations. They do not calibrate market data and they are not intended as an option-pricing benchmark.

## Structure

| Path | Role |
| --- | --- |
| `1 Simulations/simulate_examples.py` | Generates synthetic data and article figures. |
| `1 Simulations/img/` | Figure files used by the revised article. |
| `1 Simulations/output/` | CSV files with the simulated paths and variance examples. |

## Setup

Create a Python environment and install:

```bash
pip install -r requirements.txt
```

## Run

From this folder:

```bash
python "1 Simulations/simulate_examples.py"
```

## Outputs

The script writes:

- `1 Simulations/img/jump_diffusion_paths.png`;
- `1 Simulations/img/stochastic_variance_examples.png`;
- `1 Simulations/output/jump_diffusion_paths.csv`;
- `1 Simulations/output/stochastic_variance_examples.csv`.

## Reproducibility Notes

The script fixes the random seed and documents all parameters near the top of the file. The paths are synthetic and use compact discretized examples to visualize mechanisms discussed in the article.

## Authors

- Davide Sisto
- Matteo Armando

## Citation

Sisto, D. and Armando, M., *Beyond Black-Scholes: Mathematical Foundations for Jump Processes, Stochastic Volatility, and Fourier Methods*, Starting Finance Club PoliTo Research article, supporting synthetic examples.
