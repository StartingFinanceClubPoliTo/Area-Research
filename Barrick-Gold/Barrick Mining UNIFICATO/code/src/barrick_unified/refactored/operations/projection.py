"""Aligned Team 4 operating projection."""

from __future__ import annotations

from dataclasses import dataclass

import numpy as np

from .costs import CostForecast
from .production import ProductionForecast


@dataclass(frozen=True)
class OperatingProjection:
    production: ProductionForecast
    costs: CostForecast

    def __post_init__(self) -> None:
        if self.production.values.shape != self.costs.values.shape:
            raise ValueError("production and costs must use the same quarter grid")
        if self.production.values.ndim != 1:
            raise ValueError("operating projections must be one-dimensional")
        if self.production.values.size < 4 or self.production.values.size % 4:
            raise ValueError("operating projections must cover complete years")

    @property
    def quarters(self) -> int:
        return int(self.production.values.size)

    @property
    def years(self) -> int:
        return self.quarters // 4

    @property
    def data_status(self) -> str:
        statuses = {self.production.data_status, self.costs.data_status}
        return next(iter(statuses)) if len(statuses) == 1 else "MIXED"

    def annual_production_koz(self) -> np.ndarray:
        return self.production.values.reshape(self.years, 4).sum(axis=1)

    def production_weighted_cost_usd_per_oz(self) -> np.ndarray:
        production = self.production.values.reshape(self.years, 4)
        costs = self.costs.values.reshape(self.years, 4)
        return np.sum(production * costs, axis=1) / np.sum(production, axis=1)
