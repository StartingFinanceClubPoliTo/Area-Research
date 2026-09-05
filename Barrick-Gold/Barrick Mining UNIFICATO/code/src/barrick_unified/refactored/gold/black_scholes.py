"""Frozen Team 8 Black--Scholes/GBM path adapter."""

from __future__ import annotations

import numpy as np

from .base import GoldPriceModel, GoldSimulationContext


class BlackScholesGoldModel(GoldPriceModel):
    model_id = "black_scholes"
    label = "Black-Scholes / GBM"

    def _simulate_fine(self, context: GoldSimulationContext) -> np.ndarray:
        return context.team8_module.simulate_gbm_paths(
            context.start_usd_per_oz,
            context.fine_drift,
            float(context.parameters["sigma"]),
            context.grid.fine_dt_years,
            context.n_paths,
            context.seed,
        )
