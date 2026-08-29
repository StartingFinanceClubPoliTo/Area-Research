"""Common gold-engine contract."""

from __future__ import annotations

from abc import ABC, abstractmethod
from dataclasses import dataclass
from pathlib import Path
from types import ModuleType
from typing import Any

import numpy as np

from ..domain.provenance import array_sha256
from ..domain.time import QuarterGrid
from ..domain.units import Unit, UnitArray


@dataclass(frozen=True)
class GoldSimulationContext:
    grid: QuarterGrid
    n_paths: int
    seed: int
    start_usd_per_oz: float
    fine_drift: np.ndarray
    parameters: dict[str, Any]
    parameter_path: Path
    team8_module: ModuleType

    def __post_init__(self) -> None:
        drift = np.asarray(self.fine_drift, dtype=float)
        if self.n_paths < 1:
            raise ValueError("n_paths must be positive")
        if self.seed < 0:
            raise ValueError("seed must be non-negative")
        if not np.isfinite(self.start_usd_per_oz) or self.start_usd_per_oz <= 0.0:
            raise ValueError("gold start must be finite and positive")
        if drift.shape != (self.grid.fine_steps,) or not np.isfinite(drift).all():
            raise ValueError("fine_drift must align with the fine grid")
        if not self.parameter_path.is_file():
            raise FileNotFoundError(self.parameter_path)
        object.__setattr__(self, "fine_drift", drift)


@dataclass(frozen=True)
class GoldPathResult:
    model_id: str
    label: str
    quarterly_levels: UnitArray
    seed: int
    parameter_path: Path

    @property
    def quarterly_gold_paths(self) -> np.ndarray:
        return self.quarterly_levels.values

    @property
    def sha256(self) -> str:
        return array_sha256(self.quarterly_gold_paths)


class GoldPriceModel(ABC):
    """A model whose only output is a conditional gold-price layer."""

    model_id: str
    label: str

    @abstractmethod
    def _simulate_fine(self, context: GoldSimulationContext) -> np.ndarray:
        raise NotImplementedError

    def simulate(self, context: GoldSimulationContext) -> GoldPathResult:
        fine = np.asarray(self._simulate_fine(context), dtype=float)
        expected = (context.n_paths, context.grid.fine_steps + 1)
        if fine.shape != expected:
            raise ValueError(
                f"{self.model_id} fine paths must have shape {expected}, received {fine.shape}"
            )
        if not np.isfinite(fine).all() or np.any(fine <= 0.0):
            raise ValueError(f"{self.model_id} generated invalid gold paths")
        quarterly = context.grid.resample_levels(fine)
        unit_values = UnitArray(
            quarterly,
            Unit.GOLD_USD_PER_TROY_OZ,
            f"{self.model_id} quarterly gold paths",
        ).require_positive()
        return GoldPathResult(
            model_id=self.model_id,
            label=self.label,
            quarterly_levels=unit_values,
            seed=context.seed,
            parameter_path=context.parameter_path,
        )
