# Data and Diagnostics 📊

This folder contains GLD option-chain inputs and generated visual diagnostics for the stochastic option-modelling article.

## 📄 Files

| File | Description |
| --- | --- |
| `gld_chain_wide_chain.csv` | Wide option-chain dataset used by the calibration notebooks. |
| `gld_chain_wide_meta.json` | Metadata for the option-chain dataset. |
| `gld_iv_dataset_chebyshev.csv` | Implied-volatility dataset sampled with Chebyshev nodes. |
| `volatility_smiles_2d.png` | Two-dimensional implied-volatility smile diagnostics. |
| `volatility_surface_3d.png` | Three-dimensional implied-volatility surface. |
| `sampling_comparison.png` | Uniform versus Chebyshev sampling comparison. |
| `heston_residual_heatmap.png` | Heston calibration residual heatmap. |
| `bates_residual_heatmap.png` | Bates calibration residual heatmap. |
| `bates_volatility_smile.png` | Bates model volatility-smile diagnostic. |
| `error_heatmaps_comparison.png` | Interpolation or calibration error comparison. |

## 🧪 Notes

These files are publication-facing inputs and diagnostics. If the notebooks are rerun with different dates, filters, or calibration settings, regenerated figures may differ.
