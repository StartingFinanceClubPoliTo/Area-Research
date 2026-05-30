# Markoviz 📐

📄 Article PDFs are published on the Starting Finance Club PoliTo website: https://sfclubpolito.it/pubblicazioni. This GitHub repository contains code, data notes, and reproducibility assets.

Portfolio-optimization research material for Starting Finance Club PoliTo.

## 👥 Author

- [Antonio Guarini](https://github.com/Anthony27-x)

## 📌 Article

| Article | Topic | Materials |
| --- | --- | --- |
| [Markowitz Portfolio Optimization](./markowitz-portfolio-optimization-github/) | Mean-variance portfolio optimization, efficient frontiers, and out-of-sample portfolio tests | Python script, requirements, data notice, and GitHub reference text. |

## 🎯 Purpose

The uploaded material supports a Research article on Markowitz portfolio optimization. The code studies a long-only, fully invested portfolio across a liquid ETF universe with an optional Bitcoin-augmented comparison.

## 🗂️ Structure

```text
Markoviz/
|-- README.md
`-- markowitz-portfolio-optimization-github/
    |-- portfolio_optimization_markowitz.py
    |-- requirements.txt
    |-- README.md
    |-- DATA_NOTICE.md
    |-- GITHUB_REFERENCE.txt
    `-- .gitignore
```

## ▶️ Reproduce

```bash
cd Markoviz/markowitz-portfolio-optimization-github
python -m venv .venv
pip install -r requirements.txt
python portfolio_optimization_markowitz.py
```

On Windows, activate with `.venv\Scripts\activate`; on macOS/Linux, use `source .venv/bin/activate`.

## 🧪 Notes

- The analysis uses Yahoo Finance data through `yfinance`.
- Outputs can change if the provider revises historical data.
- The code is for educational and research use only and does not provide investment advice.
