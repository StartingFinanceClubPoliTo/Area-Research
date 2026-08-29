"""Time-grid value objects for the 20-quarter operating horizon."""

from __future__ import annotations

from dataclasses import dataclass

import numpy as np


@dataclass(frozen=True)
class QuarterGrid:
    """Validated mapping from a fine simulation grid to operating quarters."""

    quarters: int
    dt_years: float
    fine_steps: int

    def __post_init__(self) -> None:
        if self.quarters < 4 or self.quarters % 4:
            raise ValueError("quarters must be a positive multiple of four")
        if not np.isclose(self.dt_years, 0.25):
            raise ValueError("the Team 4 operating grid requires dt_years=0.25")
        if self.fine_steps <= 0 or self.fine_steps % self.quarters:
            raise ValueError("fine_steps must be a positive multiple of quarters")

    @property
    def years(self) -> int:
        return self.quarters // 4

    @property
    def horizon_years(self) -> float:
        return self.quarters * self.dt_years

    @property
    def steps_per_quarter(self) -> int:
        return self.fine_steps // self.quarters

    @property
    def fine_dt_years(self) -> float:
        return self.horizon_years / self.fine_steps

    @property
    def quarterly_indices(self) -> np.ndarray:
        return np.arange(
            self.steps_per_quarter,
            self.fine_steps + 1,
            self.steps_per_quarter,
            dtype=int,
        )

    def resample_levels(self, fine_paths: np.ndarray) -> np.ndarray:
        values = np.asarray(fine_paths, dtype=float)
        if values.ndim != 2 or values.shape[1] != self.fine_steps + 1:
            raise ValueError(
                "fine paths must have shape (n_paths, fine_steps + 1)"
            )
        quarterly = np.ascontiguousarray(values[:, self.quarterly_indices])
        if quarterly.shape[1] != self.quarters:
            raise AssertionError("quarterly resampling produced the wrong shape")
        return quarterly

    def as_dict(self) -> dict[str, int | float]:
        return {
            "fine_steps": self.fine_steps,
            "steps_per_quarter": self.steps_per_quarter,
            "quarters": self.quarters,
            "horizon_years": self.horizon_years,
            "fine_dt_years": self.fine_dt_years,
        }
