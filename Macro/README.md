# Macro 🌍

📄 Article PDF: https://sfclubpolito.it/pdf-viewer.html?file=Pubblicazioni%2FThe%20Output%20Cost%20of%20Inflation%20Stabilization.pdf&lang=it

Macroeconomic research material for Starting Finance Club PoliTo.

## 👥 Authors

- Emanuele Ravello
- Valentina Romeo

## 📌 Article

| Project | Topic | Materials |
| --- | --- | --- |
| [New Keynesian Cost-Push Shock Project](./NK_cost_push_prj/) | Monetary policy responses to a cost-push shock under alternative Taylor-rule regimes | Article PDF, Dynare model, MATLAB runner, exported IRF figures, results file, and CSV summary table. |

## 🎯 Purpose

`NK_cost_push_prj` implements a linear New Keynesian model with a persistent cost-push shock. The same shock is simulated under accommodative, benchmark, and aggressive Taylor-rule regimes.

## 🗂️ Structure

```text
Macro/
|-- README.md
`-- NK_cost_push_prj/
    |-- README.md
    |-- run_nk_irfs.m
    |-- nk_costpush_three_regimes.mod
    `-- Outputs/
```

## ▶️ Reproduce

Requirements:

- MATLAB;
- Dynare available on the MATLAB path.

Run from MATLAB:

```matlab
cd Macro/NK_cost_push_prj
run_nk_irfs
```

## 🧪 Notes

- The included `Outputs/` folder contains curated results extracted with the project.
- Dynare-generated temporary files and local logs are ignored through the project `.gitignore`.
- The material is for research and education only and does not provide investment advice.
