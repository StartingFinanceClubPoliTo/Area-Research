# Macro

This folder contains macroeconomic research material for Starting Finance Club PoliTo.

## Articles

| Project | Topic | Materials |
| --- | --- | --- |
| [New Keynesian Cost-Push Shock Project](./NK_cost_push_prj/) | Monetary policy responses to a cost-push shock under alternative Taylor-rule regimes | Dynare model, MATLAB runner, exported IRF figures, results file, and CSV summary table |

## Current Project

`NK_cost_push_prj` implements a linear New Keynesian model with a persistent cost-push shock. The same shock is simulated under three policy regimes:

- accommodative Taylor rule: `phi_pi = 1.10`;
- benchmark Taylor rule: `phi_pi = 1.50`;
- aggressive Taylor rule: `phi_pi = 2.00`.

The model compares the responses of inflation, the output gap, and the nominal interest rate. The MATLAB runner calls Dynare, extracts the impulse responses, builds a summary table, and exports the chart and data files.

## Repository Structure

```text
Macro/
|-- README.md
`-- NK_cost_push_prj/
    |-- README.md
    |-- run_nk_irfs.m
    |-- nk_costpush_three_regimes.mod
    |-- .gitignore
    `-- Outputs/
        |-- nk_irf_summary_table.csv
        |-- nk_irfs_three_regimes.png
        |-- nk_irfs_three_regimes.pdf
        `-- nk_irfs_results.mat
```

## How To Reproduce

Requirements:

- MATLAB;
- Dynare available on the MATLAB path.

Run from MATLAB:

```matlab
cd Macro/NK_cost_push_prj
run_nk_irfs
```

The runner writes all reproducibility outputs to `Outputs/`.

## Notes

- The included `Outputs/` folder contains the curated results extracted with the project.
- Dynare-generated temporary files and local logs are ignored through the project `.gitignore`.
- Local zip archives are ignored at repository level; version control should contain the extracted project folder.
