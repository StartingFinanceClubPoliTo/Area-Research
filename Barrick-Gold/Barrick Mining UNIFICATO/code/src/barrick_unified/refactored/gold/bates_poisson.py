"""Frozen Team 8 Bates--Poisson path adapter."""

from __future__ import annotations

import numpy as np

from .base import GoldPriceModel, GoldSimulationContext


class BatesPoissonGoldModel(GoldPriceModel):
    model_id = "bates_poisson"
    label = "Bates-Poisson"

    def _simulate_fine(self, context: GoldSimulationContext) -> np.ndarray:
        parameters = tuple(
            float(context.parameters[name])
            for name in (
                "v0",
                "kappa",
                "theta",
                "sigma",
                "rho",
                "lambd",
                "mu_J",
                "sigma_J",
            )
        )
        return context.team8_module.simulate_bates_paths(
            context.start_usd_per_oz,
            context.fine_drift,
            parameters,
            context.grid.fine_dt_years,
            context.n_paths,
            context.seed,
        )[0]
