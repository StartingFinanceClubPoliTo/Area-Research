# Simulations

Simulation stage for the Article 7 synthetic examples.

## Role

This folder generates synthetic price-path and variance-path examples for the article figures. Run the script from the `JumpSVFourier/` root folder so output paths remain stable.

## Files

| Path | Role |
| --- | --- |
| `simulate_examples.py` | Runs both simulations and writes data plus images. |
| `img/` | Stores article-ready raster figures. |
| `output/` | Stores the CSV data used to draw the figures. |

## Run

From the `JumpSVFourier/` root folder:

```bash
python "1 Simulations/simulate_examples.py"
```
