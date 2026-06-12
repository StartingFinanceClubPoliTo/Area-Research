# Advanced Stochastic Modeling for Gold Options ⚙️

This folder supports the Starting Finance Club PoliTo Research article on stochastic models for GLD option pricing and calibration.

## 👥 Authors

- [0c0c](https://github.com/0c0c)
- [SalvatoreMessina11](https://github.com/SalvatoreMessina11)

## 🎯 Purpose

The project implements Black-Scholes, Heston, Bates, and Bates-Hawkes calibration material for GLD option modelling. Fourier-style pricing routines are used for Heston and Bates, while the Bates-Hawkes layer is presented as a stationary-intensity proxy and simulation/calibration extension for clustered jump risk.

## ✨ Key Features

- Nelson-Siegel-Svensson yield-curve calibration for a continuous risk-free-rate function.
- Uniform versus Chebyshev sampling for volatility-surface reconstruction.
- Vega-weighted calibration for stochastic-volatility and jump-diffusion models.
- Bates jump-diffusion extension built on the Heston implementation.
- Hawkes-process utilities for self-exciting jump intensity, event clustering diagnostics, and a Bates-Hawkes stationary-intensity calibration proxy.

## 🗂️ Structure

| Path | Role |
| --- | --- |
| `Black and Scholes Calibration.ipynb` | Black-Scholes calibration, option-chain cleaning, implied-volatility analysis, and sampling comparison. |
| `Heston Calibration.ipynb` | Heston calibration with global and local optimizers. |
| `Bates Calibration.ipynb` | Bates jump-diffusion calibration and diagnostics. |
| `BnS.py` | Black-Scholes pricing and implied-volatility utilities. |
| `Heston.py` | Heston characteristic functions, pricing, and calibration logic. |
| `Bates.py` | Bates characteristic function, pricing, and jump calibration logic. |
| `Hawkes.py` | Exponential Hawkes intensity, branching-ratio, and simulation utilities. |
| `BatesHawkes.py` | Stationary-intensity Bates-Hawkes calibration proxy. |
| `Sampling.py` | Sampling, filtering, and interpolation utilities. |
| `Hawkes-Calibration-Notes.md` | Conceptual bridge from Bates jump intensity to possible self-exciting jump-intensity calibration extensions. |
| `calibrations/` | Script entrypoints split by Black-Scholes, Heston, Bates, Bates-Hawkes, and Hawkes diagnostics. |
| `Data/` | GLD option datasets and generated visual diagnostics. |
| `requirements.txt` | Python dependencies inferred from the code and original README. |

## ⚙️ Setup

```bash
python -m venv .venv
pip install -r requirements.txt
```

On Windows, activate with `.venv\Scripts\activate`; on macOS/Linux, use `source .venv/bin/activate`.

## ▶️ Usage

Open the calibration notebook matching the model you want to inspect and run the cells in order:

```text
Black and Scholes Calibration.ipynb
Heston Calibration.ipynb
Bates Calibration.ipynb
```

The notebooks rely on the Python modules in the root folder and the data files in `Data/`. Script entrypoints are also available:

```bash
python calibrations/01_calibrate_black_scholes.py
python calibrations/02_calibrate_heston.py
python calibrations/03_calibrate_bates.py
python calibrations/04_calibrate_bates_hawkes_proxy.py --maxiter 80 --popsize 8
python calibrations/05_generate_hawkes_diagnostics.py
```

## 📊 Outputs

The `Data/` folder contains GLD option-chain inputs and generated diagnostics such as volatility smiles, volatility surfaces, sampling comparisons, residual heatmaps, and the Hawkes intensity comparison used in the article.

## 🧪 Reproducibility Notes

- The article is scheduled for publication on July 3, 2026.
- Notebook outputs can change if the option-chain data or calibration settings are modified.
- The Bates-Hawkes script is a stationary-intensity proxy: it is useful for calibration design and diagnostics, but it is not a full event-dependent Hawkes option-pricing engine.
- Python bytecode caches and nested Git metadata from the source repository are excluded from the publication folder.
- The material is for research and education only and does not provide investment advice.

## 📚 Citation

0c0c and SalvatoreMessina11, *Advanced Stochastic Modeling for Gold Options: supporting code*, Starting Finance Club PoliTo Research, GitHub repository, expected publication 03/07/2026, accessed 31/05/2026.
