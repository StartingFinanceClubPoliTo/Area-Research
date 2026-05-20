# Area-Research

Research projects, articles, and supporting code by **Starting Finance Club PoliTo**.

This repository hosts codebases, datasets, and reproducibility
materials for research projects developed by the Research division
of the Starting Finance Club at Politecnico di Torino.

## Projects

| Project | Status | Contents |
|---------|--------|----------|
| [Barrick Gold](./Barrick-Gold/) | Active | Barrick Gold valuation and operating model research. |
| [Stoikov-Avellonada](./Stoikov-Avellonada/) | Active | Market making research. The uploaded notebook is Article 1. |
| [Markoviz](./Markoviz/) | Active | Markowitz portfolio optimization code, reproducibility notes, and data-source documentation. |
| [Macro](./Macro/) | Active | New Keynesian cost-push shock simulations with Dynare/MATLAB and exported IRF results. |

## Project Index

### Barrick Gold

- Articles and code are indexed in [Barrick Gold](./Barrick-Gold/).

### Stoikov-Avellonada

- Article 1 is indexed in [Stoikov-Avellonada](./Stoikov-Avellonada/).

### Markoviz

- [Markowitz Portfolio Optimization](./Markoviz/markowitz-portfolio-optimization-github/) contains the Python code supporting a mean-variance portfolio optimization article.
- The script downloads market data with `yfinance`, builds long-only Markowitz portfolios, compares a classical ETF universe with a Bitcoin-augmented universe, and runs fixed and rolling out-of-sample tests.

### Macro

- [New Keynesian Cost-Push Shock Project](./Macro/NK_cost_push_prj/) contains a Dynare model and MATLAB runner for impulse-response simulations under accommodative, benchmark, and aggressive Taylor-rule regimes.
- The project includes curated output files in `Outputs/`: IRF chart exports, a MATLAB results file, and a CSV summary table.

## Repository Structure

Each top-level folder is a research project. A project folder can contain one
or more article folders, notebooks, scripts, datasets, and a local `README.md`
describing the available materials.

Article folders can include:

- A `README.md` describing the article, authors, and the role of each script.
- One subfolder per analytical component, when needed.
- A `data/` folder when public datasets are committed.

## GitHub Notes

- Local `.zip` archives are treated as staging files and ignored through the root `.gitignore`.
- Commit the extracted project folders, README files, source code, and curated reproducibility outputs instead of the uploaded archives.
- Generated folders such as Python virtual environments, cache folders, IDE settings, and local logs are excluded from version control.

## Contacts

- LinkedIn: [Starting Finance Club PoliTo](https://www.linkedin.com/company/startingfinance-club-polito)
- Instagram: [@sfclubpolito](https://www.instagram.com/sfclubpolito)
- Website: [sfclubpolito.it](https://sfclubpolito.it/)
