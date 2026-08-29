"""Frozen Team 8 Full Bates--Hawkes path adapter."""

from __future__ import annotations

import numpy as np

from .base import GoldPriceModel, GoldSimulationContext


class FullBatesHawkesGoldModel(GoldPriceModel):
    model_id = "full_bates_hawkes"
    label = "Full Bates–Hawkes"

    def _simulate_fine(self, context: GoldSimulationContext) -> np.ndarray:
        return context.team8_module.simulate_full_hawkes_paths(
            context.start_usd_per_oz,
            context.fine_drift,
            context.parameters,
            context.grid.fine_dt_years,
            context.n_paths,
            context.seed,
        )[0]
