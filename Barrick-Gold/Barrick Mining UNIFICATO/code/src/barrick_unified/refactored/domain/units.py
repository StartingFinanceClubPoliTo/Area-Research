"""Small unit-aware array boundary.

NumPy arrays stay the computational representation. ``UnitArray`` is used at
layer boundaries so production, costs, gold and per-share values cannot be
silently interchanged.
"""

from __future__ import annotations

from dataclasses import dataclass
from enum import Enum

import numpy as np


class Unit(str, Enum):
    GOLD_USD_PER_TROY_OZ = "USD/troy oz"
    COST_USD_PER_TROY_OZ = "USD/troy oz cost"
    PRODUCTION_KOZ = "koz"
    USD_MN = "USD mn"
    USD_PER_SHARE = "USD/share"
    DECIMAL_RATE = "decimal rate"


@dataclass(frozen=True)
class UnitArray:
    values: np.ndarray
    unit: Unit
    name: str

    def __post_init__(self) -> None:
        values = np.asarray(self.values, dtype=float)
        if not values.size or not np.isfinite(values).all():
            raise ValueError(f"{self.name} must be finite and non-empty")
        object.__setattr__(self, "values", values)

    def require_shape(self, shape: tuple[int, ...]) -> "UnitArray":
        if self.values.shape != shape:
            raise ValueError(
                f"{self.name} must have shape {shape}, received {self.values.shape}"
            )
        return self

    def require_positive(self) -> "UnitArray":
        if np.any(self.values <= 0.0):
            raise ValueError(f"{self.name} must be strictly positive")
        return self
