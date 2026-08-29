# Team 4 frozen source references

This folder contains three byte-identical, hash-locked source files referenced
by the historical 25–26 August 2026 valuation regression configurations:

- `ebitda_montecarlo/ebitda_mc.ipynb`;
- `production/production_forecast.ipynb`;
- `cost_of_sales/cost_of_sales.R`.

They are preserved as provenance and regression inputs. The unified current
valuation does not use Team 4's legacy gold-price or illustrative EBITDA
simulation layer; it uses only the separated operating forecast described in
the current configuration and project README.

The files retain their original SHA-256 values. Keeping them locally removes a
filesystem dependency on the separate Team 4 workspace and makes the public
test suite self-contained.
