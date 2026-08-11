"""Reusable, class-based rough-process simulators.

The public functions at the bottom preserve the original command-line API.
The classes centralize validation and cache deterministic matrices so callers
that redraw paths on the same grid do not rebuild covariance or kernel data.
"""

from __future__ import annotations

from dataclasses import dataclass
from functools import cached_property
from math import gamma
from pathlib import Path

import numpy as np


def _positive_int(value: int, name: str, minimum: int = 1) -> int:
    result = int(value)
    if result < minimum:
        raise ValueError(f"{name} must be at least {minimum}")
    return result


def _validate_hurst(hurst: float) -> float:
    result = float(hurst)
    if not 0.0 < result < 1.0:
        raise ValueError("hurst must lie strictly between 0 and 1")
    return result


@dataclass(frozen=True)
class SimulationGrid:
    """Validated time discretization shared by all process models."""

    steps: int = 300
    horizon: float = 1.0

    def __post_init__(self) -> None:
        object.__setattr__(self, "steps", _positive_int(self.steps, "steps", 3))
        horizon = float(self.horizon)
        if horizon <= 0.0:
            raise ValueError("horizon must be positive")
        object.__setattr__(self, "horizon", horizon)

    @cached_property
    def time(self) -> np.ndarray:
        return np.linspace(0.0, self.horizon, self.steps)

    @property
    def dt(self) -> float:
        return float(self.time[1] - self.time[0])


@dataclass(frozen=True)
class FractionalBrownianMotion:
    """fBM covariance model with a cached Cholesky factor."""

    grid: SimulationGrid
    hurst: float = 0.5
    jitter: float = 1e-10

    def __post_init__(self) -> None:
        object.__setattr__(self, "hurst", _validate_hurst(self.hurst))

    @cached_property
    def covariance(self) -> np.ndarray:
        time = self.grid.time
        tt = time[:, None]
        ss = time[None, :]
        exponent = 2.0 * self.hurst
        return 0.5 * (tt**exponent + ss**exponent - np.abs(tt - ss) ** exponent)

    @cached_property
    def cholesky(self) -> np.ndarray:
        return np.linalg.cholesky(self.covariance + self.jitter * np.eye(self.grid.steps))

    def simulate(self, paths: int = 10, seed: int = 42) -> np.ndarray:
        count = _positive_int(paths, "paths")
        draws = np.random.default_rng(seed).standard_normal((self.grid.steps, count))
        return (self.cholesky @ draws).T


@dataclass(frozen=True)
class VolterraProcess:
    """Causal Volterra representation with a vectorized kernel build."""

    grid: SimulationGrid
    hurst: float = 0.5

    def __post_init__(self) -> None:
        object.__setattr__(self, "hurst", _validate_hurst(self.hurst))

    @cached_property
    def kernel(self) -> np.ndarray:
        steps = self.grid.steps
        if abs(self.hurst - 0.5) < 1e-12:
            kernel = np.zeros((steps, steps), dtype=float)
            kernel[np.tril_indices(steps, -1)] = 1.0
            return kernel

        time = self.grid.time
        lag = time[:, None] - time[None, :]
        causal = lag > 0.0
        coefficient = gamma(2.0 * self.hurst + 1.0) / (
            2.0 * gamma(self.hurst + 0.5) ** 2
        )
        kernel = np.zeros_like(lag)
        kernel[causal] = np.sqrt(coefficient) * np.maximum(
            lag[causal], self.grid.dt
        ) ** (self.hurst - 0.5)
        return kernel

    def simulate(self, paths: int = 10, seed: int = 77) -> tuple[np.ndarray, np.ndarray]:
        count = _positive_int(paths, "paths")
        rng = np.random.default_rng(seed)
        increments = rng.normal(
            0.0, np.sqrt(self.grid.dt), size=(count, self.grid.steps)
        )
        process = increments @ self.kernel.T
        brownian = np.cumsum(increments, axis=1)
        return process, brownian


@dataclass(frozen=True)
class MarkovianLift:
    """Finite OU lifting with cached rates, weights, and innovation factor."""

    grid: SimulationGrid
    hurst: float = 0.5
    factors: int = 10
    lambda_min: float | None = None
    lambda_max: float | None = None
    jitter: float = 1e-12

    def __post_init__(self) -> None:
        object.__setattr__(self, "hurst", _validate_hurst(self.hurst))
        object.__setattr__(self, "factors", _positive_int(self.factors, "factors"))
        lower = 2.0 / self.grid.horizon if self.lambda_min is None else float(self.lambda_min)
        upper = 400.0 / self.grid.horizon if self.lambda_max is None else float(self.lambda_max)
        if not 0.0 < lower < upper:
            raise ValueError("lambda bounds must satisfy 0 < lambda_min < lambda_max")
        object.__setattr__(self, "lambda_min", lower)
        object.__setattr__(self, "lambda_max", upper)

    @cached_property
    def rates(self) -> np.ndarray:
        return np.logspace(
            np.log10(float(self.lambda_min)),
            np.log10(float(self.lambda_max)),
            self.factors,
        )

    @cached_property
    def weights(self) -> np.ndarray:
        weights = self.rates ** (-(self.hurst + 0.5))
        return weights / weights.sum()

    @cached_property
    def decay(self) -> np.ndarray:
        return np.exp(-self.rates * self.grid.dt)

    @cached_property
    def innovation_covariance(self) -> np.ndarray:
        rate_sum = self.rates[:, None] + self.rates[None, :]
        return (1.0 - np.exp(-rate_sum * self.grid.dt)) / rate_sum

    @cached_property
    def innovation_cholesky(self) -> np.ndarray:
        return np.linalg.cholesky(
            self.innovation_covariance + self.jitter * np.eye(self.factors)
        )

    def simulate(
        self,
        paths: int = 10,
        seed: int = 101,
        match_marginals: bool = True,
    ) -> np.ndarray:
        """Simulate all paths together, leaving only the time recursion in Python.

        Complexity is O(N P M^2) for N steps, P paths, and M OU factors, but
        the former P-by-N Python loop is reduced to N batched BLAS operations.
        """

        count = _positive_int(paths, "paths")
        rng = np.random.default_rng(seed)
        draws = rng.standard_normal((count, self.grid.steps - 1, self.factors))
        innovations = draws @ self.innovation_cholesky.T
        state = np.zeros((count, self.factors), dtype=float)
        lifted = np.zeros((count, self.grid.steps), dtype=float)

        for step in range(self.grid.steps - 1):
            state = self.decay * state + innovations[:, step, :]
            lifted[:, step + 1] = state @ self.weights

        if match_marginals and count > 1:
            target_variance = self.grid.time ** (2.0 * self.hurst)
            empirical_variance = np.var(lifted, axis=0, ddof=1)
            mask = (self.grid.time > 0.0) & (empirical_variance > 1e-14)
            scale = np.ones_like(self.grid.time)
            scale[mask] = np.sqrt(target_variance[mask] / empirical_variance[mask])
            lifted *= scale

        return lifted


def time_grid(steps: int, T: float = 1.0) -> np.ndarray:
    return SimulationGrid(steps, T).time.copy()


def fbm_covariance(t: np.ndarray, hurst: float) -> np.ndarray:
    values = np.asarray(t, dtype=float)
    h = _validate_hurst(hurst)
    tt = values[:, None]
    ss = values[None, :]
    exponent = 2.0 * h
    return 0.5 * (tt**exponent + ss**exponent - np.abs(tt - ss) ** exponent)


def simulate_fbm_cholesky(
    steps: int = 300,
    hurst: float = 0.5,
    paths: int = 10,
    T: float = 1.0,
    seed: int = 42,
) -> tuple[np.ndarray, np.ndarray, np.ndarray]:
    model = FractionalBrownianMotion(SimulationGrid(steps, T), hurst)
    return model.simulate(paths, seed), model.grid.time.copy(), model.covariance.copy()


def covariance_error(paths: np.ndarray, target_cov: np.ndarray) -> tuple[np.ndarray, float]:
    empirical = np.cov(np.asarray(paths, dtype=float), rowvar=False)
    difference = empirical - np.asarray(target_cov, dtype=float)
    return difference, float(np.sqrt(np.mean(difference**2)))


def fractional_kernel_matrix(
    steps: int = 300,
    hurst: float = 0.5,
    T: float = 1.0,
) -> tuple[np.ndarray, np.ndarray]:
    model = VolterraProcess(SimulationGrid(steps, T), hurst)
    return model.grid.time.copy(), model.kernel.copy()


def simulate_volterra(
    steps: int = 300,
    hurst: float = 0.5,
    paths: int = 10,
    T: float = 1.0,
    seed: int = 77,
) -> tuple[np.ndarray, np.ndarray, np.ndarray, np.ndarray]:
    model = VolterraProcess(SimulationGrid(steps, T), hurst)
    process, brownian = model.simulate(paths, seed)
    return process, brownian, model.grid.time.copy(), model.kernel.copy()


def lift_rates_weights(
    factors: int = 10,
    hurst: float = 0.5,
    T: float = 1.0,
    lambda_min: float | None = None,
    lambda_max: float | None = None,
) -> tuple[np.ndarray, np.ndarray]:
    model = MarkovianLift(
        SimulationGrid(3, T), hurst, factors, lambda_min, lambda_max
    )
    return model.rates.copy(), model.weights.copy()


def ou_innovation_cov(lam: np.ndarray, dt: float) -> np.ndarray:
    rates = np.asarray(lam, dtype=float)
    rate_sum = rates[:, None] + rates[None, :]
    return (1.0 - np.exp(-rate_sum * float(dt))) / rate_sum


def simulate_markov_lift(
    steps: int = 300,
    hurst: float = 0.5,
    paths: int = 10,
    factors: int = 10,
    T: float = 1.0,
    seed: int = 101,
    match_marginals: bool = True,
) -> tuple[np.ndarray, np.ndarray, np.ndarray, np.ndarray]:
    model = MarkovianLift(SimulationGrid(steps, T), hurst, factors)
    lifted = model.simulate(paths, seed, match_marginals)
    return lifted, model.grid.time.copy(), model.rates.copy(), model.weights.copy()


def ensure_dir(path: str | Path) -> Path:
    output = Path(path)
    output.mkdir(parents=True, exist_ok=True)
    return output


__all__ = [
    "FractionalBrownianMotion",
    "MarkovianLift",
    "SimulationGrid",
    "VolterraProcess",
    "covariance_error",
    "ensure_dir",
    "fbm_covariance",
    "fractional_kernel_matrix",
    "lift_rates_weights",
    "ou_innovation_cov",
    "simulate_fbm_cholesky",
    "simulate_markov_lift",
    "simulate_volterra",
    "time_grid",
]
