"""Component-margin identity extracted from CODE-010."""

from __future__ import annotations

from dataclasses import dataclass

import numpy as np

from ..domain.units import Unit, UnitArray
from .projection import OperatingProjection


@dataclass(frozen=True)
class MarginProjection:
    quarterly_usd_mn: UnitArray
    annual_usd_mn: UnitArray


class ComponentMarginEngine:
    """Revenue less modeled cost of sales before accounting reconciliation."""

    def calculate(
        self,
        *,
        gold_paths_usd_per_oz: np.ndarray,
        operations: OperatingProjection,
    ) -> MarginProjection:
        gold = np.asarray(gold_paths_usd_per_oz, dtype=float)
        if gold.ndim != 2 or gold.shape[1] != operations.quarters:
            raise ValueError("gold paths must align with operating quarters")
        if not np.isfinite(gold).all() or np.any(gold <= 0.0):
            raise ValueError("gold paths must be finite and positive")
        quarterly = (
            (gold - operations.costs.values[None, :])
            * operations.production.values[None, :]
            / 1000.0
        )
        annual = quarterly.reshape(gold.shape[0], operations.years, 4).sum(axis=2)
        return MarginProjection(
            quarterly_usd_mn=UnitArray(
                quarterly, Unit.USD_MN, "quarterly component margin"
            ),
            annual_usd_mn=UnitArray(annual, Unit.USD_MN, "annual component margin"),
        )
