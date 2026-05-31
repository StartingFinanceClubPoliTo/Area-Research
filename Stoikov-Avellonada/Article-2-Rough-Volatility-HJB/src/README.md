# Source Code ⚙️

Python source for the Rough Volatility and End-of-Day Inventory Control companion folder.

## Files

| File | Role |
| --- | --- |
| `rough_processes.py` | Shared stochastic-process helpers for fBM, Volterra, and Markovian-lifting simulations. |
| `generate_article_figures.py` | Batch generator for static article-style figures. |
| `solve_rough_hjb.py` | Reduced HJB solver and Monte Carlo comparison between naive and inventory-aware policies. |
| `interactive_fbm_dashboard.py` | Local dashboard for fBM covariance and path diagnostics. |
| `interactive_volterra_dashboard.py` | Local dashboard for Volterra memory and filtration diagnostics. |
| `interactive_lift_dashboard.py` | Local dashboard for Markovian-lifting diagnostics. |

Run scripts from the article folder root so relative output paths remain stable.
