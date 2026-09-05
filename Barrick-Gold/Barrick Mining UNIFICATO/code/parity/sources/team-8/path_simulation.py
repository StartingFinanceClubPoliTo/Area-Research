"""Deterministic path engines for the publication's Bates-family diagnostics."""

import math

import numpy as np
from scipy.stats import kurtosis, norm, poisson, qmc, skew

from BatesHawkesExact import BatesHawkesExact
from Hawkes import Hawkes


def forward_rates_from_zero_curve(maturities, zero_rates, horizon, n_steps):
    """Convert an interpolated zero curve to stepwise deterministic forwards.

    This preserves the discount integral ``R(T)T`` used by pricing rather than
    feeding a five-year simulation one median spot rate.
    """
    maturities = np.asarray(maturities, dtype=float)
    zero_rates = np.asarray(zero_rates, dtype=float)
    if maturities.ndim != 1 or zero_rates.shape != maturities.shape:
        raise ValueError("maturities and zero_rates must be aligned vectors")
    if len(maturities) < 2 or np.any(np.diff(maturities) <= 0.0):
        raise ValueError("zero-curve maturities must be strictly increasing")
    if horizon <= 0.0 or n_steps < 1:
        raise ValueError("horizon and n_steps must be positive")
    edges = np.linspace(0.0, float(horizon), int(n_steps) + 1)
    edge_zero = np.interp(
        edges, maturities, zero_rates, left=zero_rates[0], right=zero_rates[-1]
    )
    integrated = edges * edge_zero
    integrated[0] = 0.0
    return np.diff(integrated) / np.diff(edges)


def normal_draws(n_paths, n_factors, seed):
    """Scrambled Sobol standard normals with column-wise moment matching."""
    power = int(math.ceil(math.log2(max(n_paths, 2))))
    uniforms = qmc.Sobol(d=n_factors, scramble=True, seed=seed).random_base2(power)
    draws = norm.ppf(np.clip(uniforms[:n_paths], 1e-10, 1.0 - 1e-10))
    standard_deviation = draws.std(axis=0)
    standard_deviation[standard_deviation < 1e-12] = 1.0
    return (draws - draws.mean(axis=0)) / standard_deviation


def uniform_draws(n_paths, n_factors, seed):
    power = int(math.ceil(math.log2(max(n_paths, 2))))
    uniforms = qmc.Sobol(d=n_factors, scramble=True, seed=seed).random_base2(power)
    return np.clip(uniforms[:n_paths], 1e-10, 1.0 - 1e-10)


def simulate_gbm_paths(spot, rates, volatility, dt, n_paths, seed, q=0.0):
    rates = np.asarray(rates, dtype=float)
    shocks = normal_draws(n_paths, rates.size, seed)
    paths = np.empty((n_paths, rates.size + 1))
    paths[:, 0] = spot
    for step, rate in enumerate(rates):
        paths[:, step + 1] = paths[:, step] * np.exp(
            (rate - q - 0.5 * volatility**2) * dt
            + volatility * math.sqrt(dt) * shocks[:, step]
        )
    return paths


def _diffusion_shocks(n_paths, n_steps, rho, seed):
    shocks = normal_draws(n_paths, 2 * n_steps, seed)
    variance_shocks = shocks[:, :n_steps]
    independent = shocks[:, n_steps:]
    price_shocks = (
        rho * variance_shocks
        + math.sqrt(max(1.0 - rho**2, 0.0)) * independent
    )
    return price_shocks, variance_shocks


def simulate_heston_paths(spot, rates, parameters, dt, n_paths, seed, q=0.0):
    """Full-truncation Euler Heston paths and variance states."""
    v0, kappa, theta, xi, rho = (float(value) for value in parameters)
    rates = np.asarray(rates, dtype=float)
    price_shocks, variance_shocks = _diffusion_shocks(
        n_paths, rates.size, rho, seed
    )
    prices = np.empty((n_paths, rates.size + 1))
    variances = np.empty_like(prices)
    prices[:, 0], variances[:, 0] = spot, v0
    for step, rate in enumerate(rates):
        variance = np.maximum(variances[:, step], 0.0)
        variances[:, step + 1] = np.maximum(
            variances[:, step]
            + kappa * (theta - variance) * dt
            + xi * np.sqrt(variance * dt) * variance_shocks[:, step],
            0.0,
        )
        prices[:, step + 1] = prices[:, step] * np.exp(
            (rate - q - 0.5 * variance) * dt
            + np.sqrt(variance * dt) * price_shocks[:, step]
        )
    return prices, variances


def simulate_bates_paths(spot, rates, parameters, dt, n_paths, seed, q=0.0):
    """Heston diffusion with constant-intensity Poisson lognormal jumps."""
    v0, kappa, theta, xi, rho, intensity, mu_j, sigma_j = (
        float(value) for value in parameters
    )
    rates = np.asarray(rates, dtype=float)
    price_shocks, variance_shocks = _diffusion_shocks(
        n_paths, rates.size, rho, seed
    )
    jump_uniforms = uniform_draws(n_paths, rates.size, seed + 101)
    jump_shocks = normal_draws(n_paths, rates.size, seed + 202)
    prices = np.empty((n_paths, rates.size + 1))
    variances = np.empty_like(prices)
    counts = np.zeros((n_paths, rates.size + 1), dtype=int)
    prices[:, 0], variances[:, 0] = spot, v0
    compensator = BatesHawkesExact.kappa_J(mu_j, sigma_j)
    for step, rate in enumerate(rates):
        variance = np.maximum(variances[:, step], 0.0)
        variances[:, step + 1] = np.maximum(
            variances[:, step]
            + kappa * (theta - variance) * dt
            + xi * np.sqrt(variance * dt) * variance_shocks[:, step],
            0.0,
        )
        new_counts = poisson.ppf(
            jump_uniforms[:, step], intensity * dt
        ).astype(int)
        counts[:, step + 1] = counts[:, step] + new_counts
        jump_sum = (
            new_counts * mu_j
            + np.sqrt(new_counts) * sigma_j * jump_shocks[:, step]
        )
        prices[:, step + 1] = prices[:, step] * np.exp(
            (rate - q - intensity * compensator - 0.5 * variance) * dt
            + np.sqrt(variance * dt) * price_shocks[:, step]
            + jump_sum
        )
    return prices, variances, counts


def _hawkes_jump_blocks(parameters, horizon, dt, n_paths, seed):
    lambda0 = float(parameters["lambda0"])
    lambda_bar = float(parameters["lambda_bar"])
    alpha = float(parameters["alpha"])
    beta = float(parameters["beta"])
    mu_j = float(parameters["mu_J"])
    sigma_j = float(parameters["sigma_J"])
    n_steps = int(round(horizon / dt))
    times = np.linspace(0.0, horizon, n_steps + 1)
    jump_sums = np.zeros((n_paths, n_steps))
    integrated = np.zeros((n_paths, n_steps))
    counts = np.zeros((n_paths, n_steps + 1), dtype=int)
    intensities = np.zeros((n_paths, n_steps + 1))
    rng = np.random.default_rng(seed)

    for path_index in range(n_paths):
        events = Hawkes.simulate_exponential(
            lambda_bar,
            alpha,
            beta,
            horizon,
            seed=int(rng.integers(1, 2**31 - 1)),
            initial_intensity=lambda0,
        )
        marks = rng.normal(mu_j, sigma_j, events.size)
        intensities[path_index] = Hawkes.intensity_on_grid(
            times,
            events,
            lambda_bar,
            alpha,
            beta,
            initial_intensity=lambda0,
        )
        for step in range(n_steps):
            start, end = times[step], times[step + 1]
            previous = events[events <= start]
            new_mask = (events > start) & (events <= end)
            new_events = events[new_mask]
            integral = lambda_bar * dt
            integral += ((lambda0 - lambda_bar) / beta) * (
                math.exp(-beta * start) - math.exp(-beta * end)
            )
            if previous.size:
                integral += (alpha / beta) * float(
                    np.sum(
                        np.exp(-beta * (start - previous))
                        - np.exp(-beta * (end - previous))
                    )
                )
            if new_events.size:
                integral += (alpha / beta) * float(
                    np.sum(1.0 - np.exp(-beta * (end - new_events)))
                )
            integrated[path_index, step] = integral
            jump_sums[path_index, step] = float(np.sum(marks[new_mask]))
            counts[path_index, step + 1] = (
                counts[path_index, step] + int(new_events.size)
            )
    return jump_sums, integrated, counts, intensities


def simulate_full_hawkes_paths(
    spot, rates, parameters, dt, n_paths, seed, q=0.0
):
    """Heston variance with event-driven exponential-Hawkes jump arrivals."""
    rates = np.asarray(rates, dtype=float)
    horizon = rates.size * dt
    jump_sums, integrated, counts, intensities = _hawkes_jump_blocks(
        parameters, horizon, dt, n_paths, seed + 303
    )
    diffusion = (
        parameters["v0"], parameters["kappa"], parameters["theta"],
        parameters["xi"], parameters["rho"]
    )
    v0, kappa, theta, xi, rho = diffusion
    price_shocks, variance_shocks = _diffusion_shocks(
        n_paths, rates.size, rho, seed
    )
    prices = np.empty((n_paths, rates.size + 1))
    variances = np.empty_like(prices)
    prices[:, 0], variances[:, 0] = spot, v0
    compensator = BatesHawkesExact.kappa_J(
        parameters["mu_J"], parameters["sigma_J"]
    )
    for step, rate in enumerate(rates):
        variance = np.maximum(variances[:, step], 0.0)
        variances[:, step + 1] = np.maximum(
            variances[:, step]
            + kappa * (theta - variance) * dt
            + xi * np.sqrt(variance * dt) * variance_shocks[:, step],
            0.0,
        )
        prices[:, step + 1] = prices[:, step] * np.exp(
            (rate - q - 0.5 * variance) * dt
            + np.sqrt(variance * dt) * price_shocks[:, step]
            - compensator * integrated[:, step]
            + jump_sums[:, step]
        )
    return prices, variances, counts, intensities


def terminal_statistics(
    model, spot, paths, counts=None, status="calibrated", simulation_method=""
):
    terminal = np.asarray(paths)[:, -1]
    log_returns = np.log(np.maximum(terminal, 1e-12) / spot)
    simple_returns_pct = (terminal / spot - 1.0) * 100.0
    return {
        "model": model,
        "simulation_method": simulation_method,
        "terminal_mean": float(np.mean(terminal)),
        "terminal_std": float(np.std(terminal, ddof=1)),
        "p05": float(np.percentile(terminal, 5)),
        "p50": float(np.percentile(terminal, 50)),
        "p95": float(np.percentile(terminal, 95)),
        "logret_skew": float(skew(log_returns)),
        "logret_excess_kurtosis": float(kurtosis(log_returns, fisher=True)),
        "return_mean_pct": float(np.mean(simple_returns_pct)),
        "return_std_pct": float(np.std(simple_returns_pct, ddof=1)),
        "return_p00_pct": float(np.percentile(simple_returns_pct, 0)),
        "return_p05_pct": float(np.percentile(simple_returns_pct, 5)),
        "return_p25_pct": float(np.percentile(simple_returns_pct, 25)),
        "return_p50_pct": float(np.percentile(simple_returns_pct, 50)),
        "return_p75_pct": float(np.percentile(simple_returns_pct, 75)),
        "return_p95_pct": float(np.percentile(simple_returns_pct, 95)),
        "return_p100_pct": float(np.percentile(simple_returns_pct, 100)),
        "return_skewness": float(skew(simple_returns_pct, bias=False)),
        "return_excess_kurtosis": float(
            kurtosis(simple_returns_pct, fisher=True, bias=False)
        ),
        "mean_jump_count": 0.0 if counts is None else float(np.mean(counts[:, -1])),
        "calibration_status": status,
        "missing_work": "",
    }
