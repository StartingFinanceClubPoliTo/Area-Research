# Source Code ⚙️

Python source for the Rough Volatility and End-of-Day Inventory Control companion folder.

## Files

| File | Role |
| --- | --- |
| `rough_processes.py` | `SimulationGrid`, Davies--Harte and Cholesky fBM classes, `VolterraProcess`, `MarkovianLift`, lag-wise covariance diagnostics, and functional wrappers. |
| `Rough_Volatility_HJB_Standalone.ipynb` | Executed, downloadable notebook with all source classes embedded directly; it imports no local project module and exposes `generate_all_article_images(...)` for the complete 16-image set. |
| `dashboard_base.py` | Common Matplotlib slider wiring, redraw lifecycle, and bounded colorbar management. |
| `generate_article_figures.py` | `ArticleFigureGenerator` and `FigureConfig` for deterministic batch output. |
| `solve_rough_hjb.py` | HJB solve, shared `MarketEnvironment`, `PolicySimulator`, and `RoughHJBExperiment`. |
| `interactive_fbm_dashboard.py` | Local fBM covariance and path dashboard. |
| `interactive_volterra_dashboard.py` | Local Volterra memory and filtration dashboard. |
| `interactive_lift_dashboard.py` | Local Markovian-lifting dashboard. |

Run scripts from the article folder root so relative output paths remain stable. Stateless numerical kernels remain functions; classes own validated configuration, cached matrices, reusable state, or orchestration.

The standalone notebook was also executed from an isolated folder containing no project modules. Its embedded smoke checks cover Davies--Harte fBM, lag-wise covariance errors, the Markovian lift, and a reduced HJB run. Calling `generate_all_article_images("standalone_outputs")` creates the 12 Hurst-regime diagnostics, `img_4.png`, `img_5.png`, `hjb_surfaces.png`, and `hjb_simulation.png`.
