# New Keynesian Cost-Push Shock Project ⚖️

📄 Article PDF: https://sfclubpolito.it/pdf-viewer.html?file=Pubblicazioni%2FThe%20Output%20Cost%20of%20Inflation%20Stabilization.pdf&lang=it

This project simulates impulse responses in a linear New Keynesian model after a cost-push shock. It supports a macroeconomic Research article on how monetary-policy intensity affects inflation and output-gap dynamics.

## 👥 Authors

- Emanuele Ravello
- Valentina Romeo

## 🎯 Purpose

The model compares inflation, output-gap, and nominal-interest-rate responses under three Taylor-rule regimes: accommodative, benchmark, and aggressive.

## 🗂️ Structure

| File | Role |
| --- | --- |
| `nk_costpush_three_regimes.mod` | Dynare model with the New Keynesian IS curve, Phillips curve, Taylor rule, and persistent cost-push shock. |
| `run_nk_irfs.m` | MATLAB runner that executes Dynare, extracts IRFs, creates the summary table, plots the three regimes, and saves outputs. |
| `Outputs/nk_irf_summary_table.csv` | Summary statistics for peak inflation, output-gap loss, peak nominal rate, and cumulative output loss. |
| `Outputs/nk_irfs_three_regimes.png` | Raster version of the IRF chart. |
| `Outputs/nk_irfs_three_regimes.pdf` | Vector version of the IRF chart. |
| `Outputs/nk_irfs_results.mat` | MATLAB results file with the plotted IRF series. |

## 🧠 Model

| Regime | Taylor-rule inflation coefficient |
| --- | ---: |
| Accommodative | `phi_pi = 1.10` |
| Benchmark | `phi_pi = 1.50` |
| Aggressive | `phi_pi = 2.00` |

Other key calibration choices include `beta = 0.99`, `sigma = 1`, `varphi = 1`, `theta = 0.75`, cost-push shock persistence `rho_u = 0.50`, and shock standard error `0.01`.

## ▶️ Run

Requirements:

- MATLAB;
- Dynare installed and available on the MATLAB path.

From MATLAB:

```matlab
cd Macro/NK_cost_push_prj
run_nk_irfs
```

## 📊 Outputs

The script creates `Outputs/` if missing and writes the summary table, PNG/PDF IRF chart, and MATLAB results file.

## 🧪 Reproducibility Notes

Curated outputs in `Outputs/` are part of the reproducibility material. Dynare-generated temporary files, local logs, and MATLAB autosave files are ignored through `.gitignore`.

## 📚 Citation

Ravello, E. and Romeo, V., *The Output Cost of Inflation Stabilization: supporting code*, Starting Finance Club PoliTo Research, GitHub repository, publication PDF at https://sfclubpolito.it/pdf-viewer.html?file=Pubblicazioni%2FThe%20Output%20Cost%20of%20Inflation%20Stabilization.pdf&lang=it, accessed 30/05/2026.
