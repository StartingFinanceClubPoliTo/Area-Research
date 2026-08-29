"""Cost models are independent of every gold-price engine."""

from __future__ import annotations

from dataclasses import dataclass
from typing import Protocol

import numpy as np

from barrick_unified.valuation import ValuationInputs

from ..domain.units import Unit, UnitArray


@dataclass(frozen=True)
class CostForecast:
    quarterly_usd_per_oz: UnitArray
    source_model: str
    data_status: str

    @property
    def values(self) -> np.ndarray:
        return self.quarterly_usd_per_oz.values


class CostModel(Protocol):
    def forecast(self) -> CostForecast: ...


class FrozenTeam4CostModel:
    """Compatibility adapter for the accepted 20 Team 4 output values."""

    def __init__(self, inputs: ValuationInputs) -> None:
        self._inputs = inputs

    def forecast(self) -> CostForecast:
        values = UnitArray(
            np.array(self._inputs.cost_usd_per_oz, copy=True),
            Unit.COST_USD_PER_TROY_OZ,
            "Team 4 quarterly unit costs",
        ).require_shape((self._inputs.n_quarters,)).require_positive()
        return CostForecast(
            quarterly_usd_per_oz=values,
            source_model="Team 4 ad hoc cost model output adapter",
            data_status="FROZEN_CODE010_OUTPUT_VECTOR_NOT_REFRESHED",
        )
