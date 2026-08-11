# Source Code ⚙️

Python source for the Rough Volatility and End-of-Day Inventory Control companion folder.

## Files

| File | Role |
| --- | --- |
| `rough_processes.py` | `SimulationGrid`, `FractionalBrownianMotion`, `VolterraProcess`, and `MarkovianLift`, plus backward-compatible functional wrappers. |
| `dashboard_base.py` | Common Matplotlib slider wiring, redraw lifecycle, and bounded colorbar management. |
| `generate_article_figures.py` | `ArticleFigureGenerator` and `FigureConfig` for deterministic batch output. |
| `solve_rough_hjb.py` | HJB solve, shared `MarketEnvironment`, `PolicySimulator`, and `RoughHJBExperiment`. |
| `interactive_fbm_dashboard.py` | Local fBM covariance and path dashboard. |
| `interactive_volterra_dashboard.py` | Local Volterra memory and filtration dashboard. |
| `interactive_lift_dashboard.py` | Local Markovian-lifting dashboard. |

Run scripts from the article folder root so relative output paths remain stable. Stateless numerical kernels remain functions; classes own validated configuration, cached matrices, reusable state, or orchestration.
