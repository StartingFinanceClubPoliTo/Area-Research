"""Variance-reduction estimators for Monte Carlo experiments."""

from __future__ import annotations

from dataclasses import dataclass
from typing import Callable

import numpy as np


@dataclass(frozen=True)
class ControlVariateResult:
    estimate: float
    beta: float
    standard_error: float
    variance_plain: float
    variance_reduced: float


@dataclass(frozen=True)
class MultiControlVariateResult:
    estimate: float
    beta: np.ndarray
    standard_error: float
    variance_plain: float
    variance_reduced: float


@dataclass(frozen=True)
class AntitheticResult:
    estimate: float
    standard_error: float
    variance_plain: float
    variance_reduced: float


@dataclass(frozen=True)
class StratifiedResult:
    estimate: float
    standard_error: float
    variance_plain: float
    variance_reduced: float


@dataclass(frozen=True)
class ImportanceSamplingResult:
    estimate_plain: float
    estimate_importance: float
    standard_error_plain: float
    standard_error_importance: float
    variance_plain: float
    variance_importance: float


def control_variate_single(y: np.ndarray, x: np.ndarray, expected_x: float) -> ControlVariateResult:
    """Apply a single control variate with known expectation."""
    y = np.asarray(y, dtype=float).reshape(-1)
    x = np.asarray(x, dtype=float).reshape(-1)
    if y.size != x.size:
        raise ValueError("y and x must have the same length")
    covariance = np.cov(y, x, ddof=1)[0, 1]
    variance_x = np.var(x, ddof=1)
    beta = 0.0 if variance_x == 0 else covariance / variance_x
    adjusted = y - beta * (x - expected_x)
    return ControlVariateResult(
        estimate=float(adjusted.mean()),
        beta=float(beta),
        standard_error=float(adjusted.std(ddof=1) / np.sqrt(y.size)),
        variance_plain=float(np.var(y, ddof=1)),
        variance_reduced=float(np.var(adjusted, ddof=1)),
    )


def control_variates_multi(y: np.ndarray, x: np.ndarray, expected_x: np.ndarray) -> MultiControlVariateResult:
    """Apply multiple control variates with known expectations."""
    y = np.asarray(y, dtype=float).reshape(-1)
    x = np.asarray(x, dtype=float)
    expected_x = np.asarray(expected_x, dtype=float).reshape(-1)
    if x.ndim != 2 or x.shape[0] != y.size or x.shape[1] != expected_x.size:
        raise ValueError("x must have shape (len(y), len(expected_x))")

    centered_x = x - x.mean(axis=0)
    centered_y = y - y.mean()
    sigma_x = centered_x.T @ centered_x / (y.size - 1)
    sigma_xy = centered_x.T @ centered_y / (y.size - 1)
    beta = np.linalg.solve(sigma_x, sigma_xy)
    adjusted = y - (x - expected_x) @ beta
    return MultiControlVariateResult(
        estimate=float(adjusted.mean()),
        beta=beta,
        standard_error=float(adjusted.std(ddof=1) / np.sqrt(y.size)),
        variance_plain=float(np.var(y, ddof=1)),
        variance_reduced=float(np.var(adjusted, ddof=1)),
    )


def antithetic_variates(
    payoff: Callable[[np.ndarray], np.ndarray],
    n_paths: int,
    dim: int = 1,
    seed: int | None = None,
) -> AntitheticResult:
    """Estimate an expectation using pairs of shocks z and -z."""
    rng = np.random.default_rng(seed)
    shocks = rng.standard_normal(n_paths if dim == 1 else (n_paths, dim))
    y = np.asarray(payoff(shocks), dtype=float).reshape(-1)
    y_anti = np.asarray(payoff(-shocks), dtype=float).reshape(-1)
    if y.size != n_paths or y_anti.size != n_paths:
        raise ValueError("payoff must return one value per path")
    paired = 0.5 * (y + y_anti)
    return AntitheticResult(
        estimate=float(paired.mean()),
        standard_error=float(paired.std(ddof=1) / np.sqrt(n_paths)),
        variance_plain=float(np.var(y, ddof=1)),
        variance_reduced=float(np.var(paired, ddof=1)),
    )


def stratified_sampling_uniform(
    payoff: Callable[[np.ndarray], np.ndarray],
    n_paths: int,
    n_strata: int,
    seed: int | None = None,
) -> StratifiedResult:
    """Estimate an expectation by stratifying the unit interval."""
    if n_strata <= 1 or n_paths < n_strata:
        raise ValueError("use at least two strata and no fewer paths than strata")
    rng = np.random.default_rng(seed)
    crude_u = rng.random(n_paths)
    crude_y = np.asarray(payoff(crude_u), dtype=float).reshape(-1)

    counts = np.full(n_strata, n_paths // n_strata, dtype=int)
    counts[: n_paths % n_strata] += 1
    means = np.empty(n_strata, dtype=float)
    variances = np.empty(n_strata, dtype=float)
    for k, count in enumerate(counts):
        u = (k + rng.random(count)) / n_strata
        y = np.asarray(payoff(u), dtype=float).reshape(-1)
        means[k] = y.mean()
        variances[k] = y.var(ddof=1) if count > 1 else 0.0

    weights = np.full(n_strata, 1.0 / n_strata)
    estimate = float(np.sum(weights * means))
    variance_estimator = float(np.sum((weights**2) * variances / counts))
    return StratifiedResult(
        estimate=estimate,
        standard_error=float(np.sqrt(variance_estimator)),
        variance_plain=float(np.var(crude_y, ddof=1)),
        variance_reduced=variance_estimator * n_paths,
    )


def importance_sampling_normal_shift(
    payoff: Callable[[np.ndarray], np.ndarray],
    n_paths: int,
    shift: float,
    seed: int | None = None,
) -> ImportanceSamplingResult:
    """Estimate E[h(Z)] for Z standard normal using a shifted normal proposal."""
    rng = np.random.default_rng(seed)
    z_plain = rng.standard_normal(n_paths)
    y_plain = np.asarray(payoff(z_plain), dtype=float).reshape(-1)

    z_shifted = rng.normal(loc=shift, scale=1.0, size=n_paths)
    likelihood_ratio = np.exp(-shift * z_shifted + 0.5 * shift**2)
    weighted = np.asarray(payoff(z_shifted), dtype=float).reshape(-1) * likelihood_ratio

    return ImportanceSamplingResult(
        estimate_plain=float(y_plain.mean()),
        estimate_importance=float(weighted.mean()),
        standard_error_plain=float(y_plain.std(ddof=1) / np.sqrt(n_paths)),
        standard_error_importance=float(weighted.std(ddof=1) / np.sqrt(n_paths)),
        variance_plain=float(np.var(y_plain, ddof=1)),
        variance_importance=float(np.var(weighted, ddof=1)),
    )
