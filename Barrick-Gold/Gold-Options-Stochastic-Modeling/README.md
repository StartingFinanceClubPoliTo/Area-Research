# Advanced Stochastic Modeling for Gold Options ⚙️

This folder supports the Starting Finance Club PoliTo Research article on stochastic models for GLD option pricing and calibration.

## 👥 Authors

- [0c0c](https://github.com/0c0c)
- [SalvatoreMessina11](https://github.com/SalvatoreMessina11)

## 🎯 Purpose

The project implements Black-Scholes, Heston, and Bates option-pricing models, with Fourier-style pricing routines, calibration notebooks, GLD option data, and diagnostics across implied-volatility smiles and surfaces.

## ✨ Key Features

- Nelson-Siegel-Svensson yield-curve calibration for a continuous risk-free-rate function.
- Uniform versus Chebyshev sampling for volatility-surface reconstruction.
- Vega-weighted calibration for stochastic-volatility and jump-diffusion models.
- Bates jump-diffusion extension built on the Heston implementation.

## 🗂️ Structure

| Path | Role |
| --- | --- |
| `Black and Scholes Calibration.ipynb` | Black-Scholes calibration, option-chain cleaning, implied-volatility analysis, and sampling comparison. |
| `Heston Calibration.ipynb` | Heston calibration with global and local optimizers. |
| `Bates Calibration.ipynb` | Bates jump-diffusion calibration and diagnostics. |
| `BnS.py` | Black-Scholes pricing and implied-volatility utilities. |
| `Heston.py` | Heston characteristic functions, pricing, and calibration logic. |
| `Bates.py` | Bates characteristic function, pricing, and jump calibration logic. |
| `Sampling.py` | Sampling, filtering, and interpolation utilities. |
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

The notebooks rely on the Python modules in the root folder and the data files in `Data/`.

## 📊 Outputs

The `Data/` folder contains GLD option-chain inputs and generated diagnostics such as volatility smiles, volatility surfaces, sampling comparisons, and residual heatmaps.

## 🧪 Reproducibility Notes

- The publication date for this article has been postponed.
- Notebook outputs can change if the option-chain data or calibration settings are modified.
- Python bytecode caches and nested Git metadata from the source repository are excluded from the publication folder.
- The material is for research and education only and does not provide investment advice.

## 📚 Citation

0c0c and SalvatoreMessina11, *Advanced Stochastic Modeling for Gold Options: supporting code*, Starting Finance Club PoliTo Research, GitHub repository, publication postponed, accessed 30/05/2026.
