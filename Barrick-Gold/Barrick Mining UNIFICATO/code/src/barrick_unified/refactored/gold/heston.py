"""Frozen Team 8 Heston path adapter."""

from __future__ import annotations

import numpy as np

from .base import GoldPriceModel, GoldSimulationContext


class HestonGoldModel(GoldPriceModel):
    model_id = "heston"
    label = "Heston"

    def _simulate_fine(self, context: GoldSimulationContext) -> np.ndarray:
        parameters = tuple(
            float(context.parameters[name])
            for name in ("v0", "kappa", "theta", "xi", "rho")
        )
        return context.team8_module.simulate_heston_paths(
            context.start_usd_per_oz,
            context.fine_drift,
            parameters,
            context.grid.fine_dt_years,
            context.n_paths,
            context.seed,
        )[0]
