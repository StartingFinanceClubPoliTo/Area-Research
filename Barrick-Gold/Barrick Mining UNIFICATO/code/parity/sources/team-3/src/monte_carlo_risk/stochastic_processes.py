"""Stochastic process simulators used in the article examples."""

from __future__ import annotations

from typing import Tuple

import numpy as np


def simulate_gbm_exact(
    s0: float,
    drift: float,
    volatility: float,
    maturity: float,
    n_steps: int,
    seed: int | None = None,
) -> Tuple[np.ndarray, np.ndarray]:
    """Simulate one exact geometric Brownian motion path on a time grid."""
    if s0 <= 0 or volatility < 0 or maturity <= 0 or n_steps <= 0:
        raise ValueError("invalid GBM parameters")
    rng = np.random.default_rng(seed)
    dt = maturity / n_steps
    time = np.linspace(0.0, maturity, n_steps + 1)
    shocks = rng.standard_normal(n_steps)
    increments = (drift - 0.5 * volatility**2) * dt + volatility * np.sqrt(dt) * shocks
    prices = s0 * np.exp(np.r_[0.0, np.cumsum(increments)])
    return time, prices


def simulate_gbm_multivariate(
    s0: np.ndarray,
    drift: np.ndarray,
    covariance: np.ndarray,
    maturity: float,
    n_steps: int,
    seed: int | None = None,
) -> Tuple[np.ndarray, np.ndarray]:
    """Simulate correlated geometric Brownian motion paths."""
    s0 = np.asarray(s0, dtype=float)
    drift = np.asarray(drift, dtype=float)
    covariance = np.asarray(covariance, dtype=float)
    if s0.ndim != 1 or drift.shape != s0.shape:
        raise ValueError("s0 and drift must be one-dimensional arrays with matching shape")
    if covariance.shape != (s0.size, s0.size):
        raise ValueError("covariance shape must match the number of assets")

    rng = np.random.default_rng(seed)
    dt = maturity / n_steps
    time = np.linspace(0.0, maturity, n_steps + 1)
    factor = np.linalg.cholesky(covariance)
    prices = np.empty((n_steps + 1, s0.size), dtype=float)
    prices[0] = s0
    variance = np.diag(covariance)

    for step in range(n_steps):
        shock = factor @ rng.standard_normal(s0.size)
        prices[step + 1] = prices[step] * np.exp((drift - 0.5 * variance) * dt + np.sqrt(dt) * shock)
    return time, prices


def risk_neutral_jump_drift(rate: float, dividend_yield: float, intensity: float, jump_mean: float, jump_volatility: float) -> float:
    """Compute the risk-neutral drift for a Merton jump-diffusion log process."""
    expected_relative_jump = np.exp(jump_mean + 0.5 * jump_volatility**2) - 1.0
    return rate - dividend_yield - intensity * expected_relative_jump


def simulate_jump_diffusion_grid(
    s0: float,
    drift: float,
    volatility: float,
    intensity: float,
    jump_mean: float,
    jump_volatility: float,
    maturity: float,
    n_steps: int,
    seed: int | None = None,
) -> Tuple[np.ndarray, np.ndarray]:
    """Simulate a Merton jump-diffusion path on a fixed time grid."""
    rng = np.random.default_rng(seed)
    dt = maturity / n_steps
    time = np.linspace(0.0, maturity, n_steps + 1)
    log_price = np.empty(n_steps + 1, dtype=float)
    log_price[0] = np.log(s0)

    for step in range(n_steps):
        n_jumps = rng.poisson(intensity * dt)
        jump_component = 0.0
        if n_jumps > 0:
            jump_component = n_jumps * jump_mean + np.sqrt(n_jumps) * jump_volatility * rng.standard_normal()
        diffusion = (drift - 0.5 * volatility**2) * dt + volatility * np.sqrt(dt) * rng.standard_normal()
        log_price[step + 1] = log_price[step] + diffusion + jump_component

    return time, np.exp(log_price)


def simulate_jump_diffusion_jump_times(
    s0: float,
    drift: float,
    volatility: float,
    intensity: float,
    jump_mean: float,
    jump_volatility: float,
    maturity: float,
    seed: int | None = None,
) -> Tuple[np.ndarray, np.ndarray]:
    """Simulate a Merton jump-diffusion path by drawing explicit jump times."""
    rng = np.random.default_rng(seed)
    time = 0.0
    price = float(s0)
    times = [time]
    prices = [price]

    while time < maturity:
        wait = rng.exponential(1.0 / intensity) if intensity > 0 else np.inf
        next_time = min(time + wait, maturity)
        dt = next_time - time
        if dt > 0:
            price *= np.exp((drift - 0.5 * volatility**2) * dt + volatility * np.sqrt(dt) * rng.standard_normal())
            time = next_time
            times.append(time)
            prices.append(price)
        if time >= maturity:
            break
        price *= np.exp(jump_mean + jump_volatility * rng.standard_normal())
        times.append(time)
        prices.append(price)

    return np.asarray(times), np.asarray(prices)
