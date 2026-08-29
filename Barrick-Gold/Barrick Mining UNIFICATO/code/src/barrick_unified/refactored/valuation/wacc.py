"""Team 5 mean-reverting WACC process."""

from __future__ import annotations

from dataclasses import dataclass

import numpy as np

from barrick_unified.valuation import ValuationInputs

from ..domain.units import Unit, UnitArray


@dataclass(frozen=True)
class WACCProjection:
    annual_rates: UnitArray


class WACCProcess:
    def __init__(self, inputs: ValuationInputs) -> None:
        self._inputs = inputs

    def simulate(self, shocks: np.ndarray) -> WACCProjection:
        inputs = self._inputs
        shocks = np.asarray(shocks, dtype=float)
        expected = (inputs.n_simulations, inputs.n_years)
        if shocks.shape != expected or not np.isfinite(shocks).all():
            raise ValueError(f"WACC shocks must be finite with shape {expected}")
        wacc = np.empty((inputs.n_simulations, inputs.n_years + 1), dtype=float)
        wacc[:, 0] = inputs.wacc_0
        mean_reversion = np.exp(-inputs.wacc_reversion)
        conditional_std = inputs.wacc_volatility * np.sqrt(
            (1.0 - np.exp(-2.0 * inputs.wacc_reversion))
            / (2.0 * inputs.wacc_reversion)
        )
        for year in range(inputs.n_years):
            wacc[:, year + 1] = (
                inputs.wacc_long_run
                + (wacc[:, year] - inputs.wacc_long_run) * mean_reversion
                + conditional_std * shocks[:, year]
            )
        minimum = inputs.stable_growth + inputs.terminal_spread_floor
        annual = np.clip(wacc[:, 1:], minimum, 0.25)
        return WACCProjection(UnitArray(annual, Unit.DECIMAL_RATE, "annual WACC"))
