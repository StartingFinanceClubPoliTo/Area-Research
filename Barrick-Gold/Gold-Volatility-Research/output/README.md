# Outputs 📊

Generated figures, tables, and CSV outputs for the gold, silver, and macro-financial driver analysis.

## 📄 Contents

| Folder | Role |
| --- | --- |
| `csv/` | Machine-readable regression, AR/ARMA/ARIMA, GARCH, ARFIMA/GPH, stylized-fact, and summary outputs. |
| `figures/` | Correlation, quality-check, conditional-volatility, and long-memory diagnostic figures. |
| `tables/` | LaTeX-ready regression, AR/ARMA/ARIMA, descriptive, correlation, GARCH, and ARFIMA tables for article integration. |

## 🔁 Regeneration

From the project root:

```bash
python src/main.py
```

Generated files can change if upstream market data or model settings are updated.
