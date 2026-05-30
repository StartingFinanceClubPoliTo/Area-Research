# Examples ▶️

Runnable scripts that regenerate the curated figures and tables for the Monte Carlo risk article.

## 📄 Key File

| File | Role |
| --- | --- |
| `run_reproducibility.py` | Generates random-number diagnostics, option-pricing convergence charts, variance-reduction summaries, Gaussian-mixture return statistics, and terminal-wealth risk metrics. |

## 🚀 How To Run

From the `Monte-Carlo-Risk/` root folder:

```bash
python examples/run_reproducibility.py
```

The script writes charts to `figures/` and summary tables to `outputs/`.

## 🧪 Notes

- The script uses deterministic seeds for the publication-facing examples.
- It imports local modules from `src/`, so no package installation step is required beyond installing `requirements.txt`.
