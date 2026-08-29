"""Option-pricing routines for Monte Carlo examples."""

from __future__ import annotations

from typing import Tuple

import numpy as np
from scipy.stats import norm, qmc


def black_scholes_call(s0: float, strike: float, rate: float, volatility: float, maturity: float) -> float:
    """Return the Black-Scholes price of a European call."""
    d1 = (np.log(s0 / strike) + (rate + 0.5 * volatility**2) * maturity) / (volatility * np.sqrt(maturity))
    d2 = d1 - volatility * np.sqrt(maturity)
    return float(s0 * norm.cdf(d1) - strike * np.exp(-rate * maturity) * norm.cdf(d2))


def black_scholes_put(s0: float, strike: float, rate: float, volatility: float, maturity: float) -> float:
    """Return the Black-Scholes price of a European put."""
    d1 = (np.log(s0 / strike) + (rate + 0.5 * volatility**2) * maturity) / (volatility * np.sqrt(maturity))
    d2 = d1 - volatility * np.sqrt(maturity)
    return float(strike * np.exp(-rate * maturity) * norm.cdf(-d2) - s0 * norm.cdf(-d1))


def monte_carlo_european_call(
    s0: float,
    strike: float,
    rate: float,
    volatility: float,
    maturity: float,
    n_paths: int,
    seed: int | None = None,
) -> Tuple[float, float]:
    """Estimate a European call price and standard error under risk-neutral GBM."""
    rng = np.random.default_rng(seed)
    shocks = rng.standard_normal(n_paths)
    terminal = s0 * np.exp((rate - 0.5 * volatility**2) * maturity + volatility * np.sqrt(maturity) * shocks)
    discounted = np.exp(-rate * maturity) * np.maximum(terminal - strike, 0.0)
    return float(discounted.mean()), float(discounted.std(ddof=1) / np.sqrt(n_paths))


def price_asian_call_mc(
    s0: float,
    strike: float,
    rate: float,
    volatility: float,
    maturity: float,
    n_steps: int,
    n_paths: int,
    seed: int | None = None,
) -> Tuple[float, float]:
    """Price an arithmetic Asian call with standard Monte Carlo."""
    rng = np.random.default_rng(seed)
    dt = maturity / n_steps
    shocks = rng.standard_normal((n_paths, n_steps))
    increments = (rate - 0.5 * volatility**2) * dt + volatility * np.sqrt(dt) * shocks
    paths = s0 * np.exp(np.cumsum(increments, axis=1))
    average = paths.mean(axis=1)
    discounted = np.exp(-rate * maturity) * np.maximum(average - strike, 0.0)
    return float(discounted.mean()), float(discounted.std(ddof=1) / np.sqrt(n_paths))


def price_asian_call_qmc(
    s0: float,
    strike: float,
    rate: float,
    volatility: float,
    maturity: float,
    n_steps: int,
    n_paths_power: int,
    seed: int | None = None,
) -> Tuple[float, float]:
    """Price an arithmetic Asian call using Sobol quasi-Monte Carlo points."""
    sampler = qmc.Sobol(d=n_steps, scramble=True, seed=seed)
    uniforms = sampler.random_base2(m=n_paths_power)
    shocks = norm.ppf(np.clip(uniforms, np.finfo(float).eps, 1.0 - np.finfo(float).eps))
    dt = maturity / n_steps
    increments = (rate - 0.5 * volatility**2) * dt + volatility * np.sqrt(dt) * shocks
    paths = s0 * np.exp(np.cumsum(increments, axis=1))
    average = paths.mean(axis=1)
    discounted = np.exp(-rate * maturity) * np.maximum(average - strike, 0.0)
    return float(discounted.mean()), float(discounted.std(ddof=1) / np.sqrt(discounted.size))


def price_up_and_out_call(
    s0: float,
    strike: float,
    barrier: float,
    rate: float,
    volatility: float,
    maturity: float,
    n_steps: int,
    n_paths: int,
    seed: int | None = None,
) -> Tuple[float, float]:
    """Estimate an up-and-out call using a Brownian-bridge crossing correction."""
    if barrier <= max(s0, strike):
        raise ValueError("barrier should be above s0 and strike for this example")

    rng = np.random.default_rng(seed)
    dt = maturity / n_steps
    log_barrier = np.log(barrier)
    payoffs = np.empty(n_paths, dtype=float)

    for path in range(n_paths):
        log_s = np.log(s0)
        alive_probability = 1.0
        knocked_out = False
        for _ in range(n_steps):
            log_next = log_s + (rate - 0.5 * volatility**2) * dt + volatility * np.sqrt(dt) * rng.standard_normal()
            if log_s >= log_barrier or log_next >= log_barrier:
                knocked_out = True
                break
            crossing_probability = np.exp(
                -2.0 * (log_barrier - log_s) * (log_barrier - log_next) / (volatility**2 * dt)
            )
            alive_probability *= 1.0 - crossing_probability
            log_s = log_next
        terminal = np.exp(log_s)
        payoff = 0.0 if knocked_out else alive_probability * max(terminal - strike, 0.0)
        payoffs[path] = np.exp(-rate * maturity) * payoff

    return float(payoffs.mean()), float(payoffs.std(ddof=1) / np.sqrt(n_paths))
