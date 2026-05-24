# Barrick Gold

Research code companions for the Barrick Gold article track by **Starting Finance Club PoliTo**.

This folder collects the publication-facing GitHub material for the Barrick Gold series. The index below lists only articles that currently include code or reproducibility assets; article folders without executable code are left out of the public code table.

## Articles With Code

| Article | Project | Authors | Materials |
| --- | --- | --- | --- |
| Article 1 | [Component-Driven EBITDA Forecast for Barrick Gold](./Component-Driven-EBITDA-Barrick/) | [Giacomo Scali](https://www.linkedin.com/in/giacomo-scali-abp01/), [Jacopo Foralosso](https://www.linkedin.com/in/jacopo-foralosso-6753a2256) | Cost-of-sales model, production forecast notebooks, GLD option-implied volatility work, and EBITDA Monte Carlo simulation. |
| Article 4 | [Monte Carlo Methods and Portfolio Simulation](./Monte-Carlo-Risk/) | Andrea Rostagno, Francesco Florio | Reusable Python package, reproducibility script, generated figures, and CSV summary outputs. |
| Article 7 | [Beyond Black-Scholes Synthetic Examples](./JumpSVFourier/) | Davide Sisto, Matteo Armando | Synthetic simulations, article figures, CSV paths, and compact reproducibility notes. |

## Folder Structure

```text
Barrick-Gold/
|-- README.md
|-- Component-Driven-EBITDA-Barrick/
|   |-- cost_of_sales/
|   |-- production/
|   `-- ebitda_montecarlo/
|-- Monte-Carlo-Risk/
|   |-- src/
|   |-- examples/
|   |-- figures/
|   `-- outputs/
|-- JumpSVFourier/
|   `-- 1 Simulations/
`-- OLS-Dynamic-Corr/
```

## Reproducibility Notes

- Each project folder contains its own setup and run instructions.
- Committed outputs are curated examples used to document the workflow, not investment signals.
- External market, option-chain, or company data are redistributed only when the source allows it. When raw data are not committed, the relevant README explains how to reconstruct the input.
- `OLS-Dynamic-Corr/` is kept as a placeholder for the Team 2 article, but it is not listed above because no executable code was supplied in the current folder.

## Citation

When citing code from this folder, cite the article title, authors, Starting Finance Club PoliTo Research, and the specific GitHub folder used for reproduction.
