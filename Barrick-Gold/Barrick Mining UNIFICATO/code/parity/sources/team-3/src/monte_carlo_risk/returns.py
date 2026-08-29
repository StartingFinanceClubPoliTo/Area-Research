"""Return simulation and portfolio risk helpers."""

from __future__ import annotations

from typing import Tuple

import numpy as np
from scipy.special import ndtr
from scipy.stats import kurtosis, skew


def simulate_gaussian_mixture_returns(
    n: int,
    weights: np.ndarray | None = None,
    means: np.ndarray | None = None,
    volatilities: np.ndarray | None = None,
    seed: int | None = None,
) -> Tuple[np.ndarray, dict[str, float]]:
    """Simulate returns from a finite mixture of normal distributions."""
    rng = np.random.default_rng(seed)
    if weights is None:
        weights = np.array([0.85, 0.15], dtype=float)
    if means is None:
        means = np.array([0.0005, -0.0020], dtype=float)
    if volatilities is None:
        volatilities = np.array([0.01, 0.04], dtype=float)

    weights = np.asarray(weights, dtype=float)
    means = np.asarray(means, dtype=float)
    volatilities = np.asarray(volatilities, dtype=float)
    if not (weights.size == means.size == volatilities.size):
        raise ValueError("weights, means, and volatilities must have the same length")
    if np.any(weights <= 0.0) or np.any(volatilities <= 0.0):
        raise ValueError("weights and volatilities must be positive")
    weights = weights / weights.sum()

    regimes = rng.choice(weights.size, size=n, p=weights)
    returns = means[regimes] + volatilities[regimes] * rng.standard_normal(n)
    moments = {
        "mean": float(returns.mean()),
        "variance": float(returns.var(ddof=0)),
        "skewness": float(skew(returns, bias=False)),
        "excess_kurtosis": float(kurtosis(returns, fisher=True, bias=False)),
    }
    return returns, moments


def empirical_inverse_cdf(uniform_values: np.ndarray, sample: np.ndarray) -> np.ndarray:
    """Map uniforms to a return sample through the empirical inverse CDF."""
    uniform_values = np.asarray(uniform_values, dtype=float)
    sample = np.sort(np.asarray(sample, dtype=float))
    probabilities = np.linspace(0.0, 1.0, sample.size)
    return np.interp(np.clip(uniform_values, 0.0, 1.0), probabilities, sample)


def value_at_risk(returns: np.ndarray, alpha: float = 0.05) -> float:
    """Return the lower-tail empirical Value-at-Risk at level alpha."""
    if not 0.0 < alpha < 1.0:
        raise ValueError("alpha must be between 0 and 1")
    return float(np.quantile(np.asarray(returns, dtype=float), alpha))


def conditional_value_at_risk(returns: np.ndarray, alpha: float = 0.05) -> float:
    """Return the average return conditional on being below VaR_alpha."""
    returns = np.asarray(returns, dtype=float)
    var = value_at_risk(returns, alpha)
    return float(returns[returns <= var].mean())


def simulate_terminal_wealth(
    returns_1: np.ndarray,
    returns_2: np.ndarray,
    weights: np.ndarray,
    dependence: float,
    initial_value: float = 100.0,
    n_paths: int = 5_000,
    n_days: int = 252,
    seed: int | None = None,
) -> np.ndarray:
    """Simulate terminal wealth by mapping Gaussian-copula uniforms to empirical returns."""
    weights = np.asarray(weights, dtype=float)
    if weights.shape != (2,):
        raise ValueError("weights must contain exactly two entries")
    rng = np.random.default_rng(seed)
    covariance = np.array([[1.0, dependence], [dependence, 1.0]], dtype=float)
    terminal = np.empty(n_paths, dtype=float)

    for path in range(n_paths):
        z = rng.multivariate_normal(np.zeros(2), covariance, size=n_days)
        uniforms = ndtr(z)
        r1 = empirical_inverse_cdf(uniforms[:, 0], returns_1)
        r2 = empirical_inverse_cdf(uniforms[:, 1], returns_2)
        portfolio_returns = weights[0] * r1 + weights[1] * r2
        terminal[path] = initial_value * np.exp(portfolio_returns.sum())
    return terminal
