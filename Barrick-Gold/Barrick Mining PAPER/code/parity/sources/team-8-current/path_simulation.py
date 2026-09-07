"""Team 8 2026-09-02 path adapter for the unified Barrick experiment.

The calibrated option models live in the vendored Team 8 publication source.
This small integration layer exposes the four path functions expected by the
Barrick valuation runner.  It follows the full-truncation Euler dynamics used
by the current Team 8 diagnostic simulations and accepts a deterministic
stepwise risk-neutral rate vector.
"""

from __future__ import annotations

from typing import Mapping, Sequence

import numpy as np


def _inputs(
    start: float,
    rates: Sequence[float],
    dt: float,
    n_paths: int,
) -> tuple[np.ndarray, int]:
    rate_array = np.asarray(rates, dtype=float)
    if start <= 0 or dt <= 0 or n_paths <= 0 or rate_array.ndim != 1:
        raise ValueError("invalid path-simulation inputs")
    if not np.all(np.isfinite(rate_array)):
        raise ValueError("rates must be finite")
    return rate_array, rate_array.size


def _correlated_normals(
    rng: np.random.Generator, n_paths: int, rho: float
) -> tuple[np.ndarray, np.ndarray]:
    if not -1.0 <= rho <= 1.0:
        raise ValueError("rho must lie in [-1, 1]")
    z_price = rng.standard_normal(n_paths)
    z_independent = rng.standard_normal(n_paths)
    z_variance = rho * z_price + np.sqrt(max(0.0, 1.0 - rho * rho)) * z_independent
    return z_price, z_variance


def simulate_gbm_paths(
    start: float,
    rates: Sequence[float],
    sigma: float,
    dt: float,
    n_paths: int,
    seed: int,
) -> np.ndarray:
    rate_array, n_steps = _inputs(start, rates, dt, n_paths)
    if sigma < 0:
        raise ValueError("sigma must be non-negative")
    rng = np.random.default_rng(seed)
    paths = np.empty((n_paths, n_steps + 1), dtype=float)
    paths[:, 0] = start
    for step, rate in enumerate(rate_array):
        shock = rng.standard_normal(n_paths)
        paths[:, step + 1] = paths[:, step] * np.exp(
            (rate - 0.5 * sigma * sigma) * dt + sigma * np.sqrt(dt) * shock
        )
    return paths


def simulate_heston_paths(
    start: float,
    rates: Sequence[float],
    parameters: Sequence[float],
    dt: float,
    n_paths: int,
    seed: int,
) -> tuple[np.ndarray, np.ndarray]:
    rate_array, n_steps = _inputs(start, rates, dt, n_paths)
    v0, kappa, theta, vol_of_vol, rho = map(float, parameters)
    rng = np.random.default_rng(seed)
    paths = np.empty((n_paths, n_steps + 1), dtype=float)
    variance = np.empty_like(paths)
    paths[:, 0] = start
    variance[:, 0] = max(v0, 0.0)
    root_dt = np.sqrt(dt)
    for step, rate in enumerate(rate_array):
        z_price, z_variance = _correlated_normals(rng, n_paths, rho)
        v_pos = np.maximum(variance[:, step], 0.0)
        paths[:, step + 1] = paths[:, step] * np.exp(
            (rate - 0.5 * v_pos) * dt + np.sqrt(v_pos) * root_dt * z_price
        )
        variance[:, step + 1] = np.maximum(
            variance[:, step]
            + kappa * (theta - v_pos) * dt
            + vol_of_vol * np.sqrt(v_pos) * root_dt * z_variance,
            0.0,
        )
    return paths, variance


def simulate_bates_paths(
    start: float,
    rates: Sequence[float],
    parameters: Sequence[float],
    dt: float,
    n_paths: int,
    seed: int,
) -> tuple[np.ndarray, np.ndarray]:
    rate_array, n_steps = _inputs(start, rates, dt, n_paths)
    v0, kappa, theta, vol_of_vol, rho, intensity, mu_jump, sigma_jump = map(
        float, parameters
    )
    if intensity < 0 or sigma_jump < 0:
        raise ValueError("jump intensity and dispersion must be non-negative")
    rng = np.random.default_rng(seed)
    paths = np.empty((n_paths, n_steps + 1), dtype=float)
    variance = np.empty_like(paths)
    paths[:, 0] = start
    variance[:, 0] = max(v0, 0.0)
    root_dt = np.sqrt(dt)
    compensator = np.exp(mu_jump + 0.5 * sigma_jump * sigma_jump) - 1.0
    for step, rate in enumerate(rate_array):
        z_price, z_variance = _correlated_normals(rng, n_paths, rho)
        jump_count = rng.poisson(intensity * dt, n_paths)
        jump_log = jump_count * mu_jump + np.sqrt(jump_count) * sigma_jump * rng.standard_normal(n_paths)
        v_pos = np.maximum(variance[:, step], 0.0)
        paths[:, step + 1] = paths[:, step] * np.exp(
            (rate - intensity * compensator - 0.5 * v_pos) * dt
            + np.sqrt(v_pos) * root_dt * z_price
            + jump_log
        )
        variance[:, step + 1] = np.maximum(
            variance[:, step]
            + kappa * (theta - v_pos) * dt
            + vol_of_vol * np.sqrt(v_pos) * root_dt * z_variance,
            0.0,
        )
    return paths, variance


def simulate_full_hawkes_paths(
    start: float,
    rates: Sequence[float],
    parameters: Mapping[str, float],
    dt: float,
    n_paths: int,
    seed: int,
) -> tuple[np.ndarray, np.ndarray, np.ndarray]:
    rate_array, n_steps = _inputs(start, rates, dt, n_paths)
    p = {name: float(value) for name, value in parameters.items()}
    branching = p["branching_ratio"]
    if not 0.0 <= branching < 1.0:
        raise ValueError("stationary Hawkes branching_ratio must lie in [0, 1)")
    beta = p["beta"]
    alpha = p.get("alpha", branching * beta)
    if beta <= 0 or alpha < 0 or p["sigma_J"] < 0:
        raise ValueError("invalid Hawkes or jump parameters")
    rng = np.random.default_rng(seed)
    paths = np.empty((n_paths, n_steps + 1), dtype=float)
    variance = np.empty_like(paths)
    intensity = np.empty_like(paths)
    paths[:, 0] = start
    variance[:, 0] = max(p["v0"], 0.0)
    intensity[:, 0] = max(p["lambda0"], 0.0)
    root_dt = np.sqrt(dt)
    compensator = np.exp(p["mu_J"] + 0.5 * p["sigma_J"] ** 2) - 1.0
    for step, rate in enumerate(rate_array):
        z_price, z_variance = _correlated_normals(rng, n_paths, p["rho"])
        lam_pos = np.maximum(intensity[:, step], 0.0)
        jump_count = rng.poisson(lam_pos * dt)
        jump_log = (
            jump_count * p["mu_J"]
            + np.sqrt(jump_count) * p["sigma_J"] * rng.standard_normal(n_paths)
        )
        v_pos = np.maximum(variance[:, step], 0.0)
        paths[:, step + 1] = paths[:, step] * np.exp(
            (rate - lam_pos * compensator - 0.5 * v_pos) * dt
            + np.sqrt(v_pos) * root_dt * z_price
            + jump_log
        )
        variance[:, step + 1] = np.maximum(
            variance[:, step]
            + p["kappa"] * (p["theta"] - v_pos) * dt
            + p["xi"] * np.sqrt(v_pos) * root_dt * z_variance,
            0.0,
        )
        intensity[:, step + 1] = np.maximum(
            intensity[:, step]
            + beta * (p["lambda_bar"] - lam_pos) * dt
            + alpha * jump_count,
            0.0,
        )
    return paths, variance, intensity
