# Area Research

Research code, reproducibility material, and article companions by **Starting Finance Club PoliTo**.

This repository collects the GitHub-facing material produced by the Research division at Politecnico di Torino. Each folder is organized around an article series or research project, with README files, scripts, notebooks, requirements, curated outputs, and reproducibility notes when code is available.

## Barrick Gold Article Code Index

Only articles with a supplied code folder are listed here. Articles without executable code are intentionally omitted from the table.

| Article | Code Folder | Authors | Material |
| --- | --- | --- | --- |
| Article 1 | [Component-Driven EBITDA Forecast for Barrick Gold](./Barrick-Gold/Component-Driven-EBITDA-Barrick/) | [Giacomo Scali](https://www.linkedin.com/in/giacomo-scali-abp01/), [Jacopo Foralosso](https://www.linkedin.com/in/jacopo-foralosso-6753a2256) | EBITDA forecasting workflow with cost-of-sales, production, and gold-price Monte Carlo components. |
| Article 4 | [Monte Carlo Methods and Portfolio Simulation](./Barrick-Gold/Monte-Carlo-Risk/) | Andrea Rostagno, Francesco Florio | Python package for Monte Carlo convergence, stochastic processes, option pricing, variance reduction, and portfolio risk examples. |
| Article 7 | [Beyond Black-Scholes Synthetic Examples](./Barrick-Gold/JumpSVFourier/) | Davide Sisto, Matteo Armando | Synthetic simulations and figures for jump processes, stochastic volatility, Fourier methods, and Hawkes dynamics. |

## Other Research Projects

| Project | Folder | Material |
| --- | --- | --- |
| Stoikov-Avellonada | [Stoikov-Avellonada](./Stoikov-Avellonada/) | Market-making notebook for Article 1, with an Avellaneda-Stoikov simulation pipeline. |
| Markoviz | [Markoviz](./Markoviz/) | Markowitz portfolio optimization code, data-source documentation, and GitHub reference material. |
| Macro | [Macro](./Macro/) | New Keynesian cost-push shock simulations with Dynare/MATLAB and exported IRF outputs. |

## Repository Map

```text
Research/
|-- README.md
|-- Barrick-Gold/
|   |-- Component-Driven-EBITDA-Barrick/
|   |-- Monte-Carlo-Risk/
|   |-- JumpSVFourier/
|   `-- OLS-Dynamic-Corr/
|-- Markoviz/
|   `-- markowitz-portfolio-optimization-github/
|-- Stoikov-Avellonada/
|   `-- Article-1/
`-- Macro/
    `-- NK_cost_push_prj/
```

## How To Use This Repository

1. Open the project folder that matches the article or research topic.
2. Read the local `README.md` for setup, run commands, required packages, outputs, and data notes.
3. Use committed scripts and curated outputs as the publication-facing reproducibility material.

## Version-Control Notes

- Local `.zip` archives, virtual environments, cache folders, IDE settings, and local logs are excluded from version control.
- Publication-facing folders should contain extracted source files, README files, requirements, scripts, notebooks when needed, and curated outputs.
- Raw external data are included only when redistribution is allowed. Otherwise, the README explains where the data come from and how to regenerate the workflow.

## Contacts

- LinkedIn: [Starting Finance Club PoliTo](https://www.linkedin.com/company/startingfinance-club-polito)
- Instagram: [@sfclubpolito](https://www.instagram.com/sfclubpolito)
- Website: [sfclubpolito.it](https://sfclubpolito.it/)
