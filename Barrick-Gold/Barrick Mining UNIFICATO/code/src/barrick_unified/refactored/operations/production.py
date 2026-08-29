"""Production models are independent of every gold-price engine."""

from __future__ import annotations

from dataclasses import dataclass
from typing import Protocol

import numpy as np

from barrick_unified.valuation import ValuationInputs

from ..domain.units import Unit, UnitArray


@dataclass(frozen=True)
class ProductionForecast:
    quarterly_koz: UnitArray
    source_model: str
    data_status: str

    @property
    def values(self) -> np.ndarray:
        return self.quarterly_koz.values


class ProductionModel(Protocol):
    def forecast(self) -> ProductionForecast: ...


class FrozenTeam4ProductionModel:
    """Compatibility adapter for the accepted 20 Team 4 output values."""

    def __init__(self, inputs: ValuationInputs) -> None:
        self._inputs = inputs

    def forecast(self) -> ProductionForecast:
        values = UnitArray(
            np.array(self._inputs.production_koz, copy=True),
            Unit.PRODUCTION_KOZ,
            "Team 4 quarterly production",
        ).require_shape((self._inputs.n_quarters,)).require_positive()
        return ProductionForecast(
            quarterly_koz=values,
            source_model="Team 4 ad hoc production model output adapter",
            data_status="FROZEN_CODE010_OUTPUT_VECTOR_NOT_REFRESHED",
        )
