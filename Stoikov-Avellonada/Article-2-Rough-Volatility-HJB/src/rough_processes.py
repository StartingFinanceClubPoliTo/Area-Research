"""Reusable rough-process simulators for the HFT Article 2 companion code."""

from __future__ import annotations

from math import gamma
from pathlib import Path

import numpy as np


def time_grid(steps: int, T: float = 1.0) -> np.ndarray:
    if steps < 3:
        raise ValueError("steps must be at least 3")
    return np.linspace(0.0, T, int(steps))


def fbm_covariance(t: np.ndarray, hurst: float) -> np.ndarray:
    tt = t[:, None]
    ss = t[None, :]
    return 0.5 * (tt ** (2 * hurst) + ss ** (2 * hurst) - np.abs(tt - ss) ** (2 * hurst))


def simulate_fbm_cholesky(
    steps: int = 300,
    hurst: float = 0.5,
    paths: int = 10,
    T: float = 1.0,
    seed: int = 42,
) -> tuple[np.ndarray, np.ndarray, np.ndarray]:
    t = time_grid(steps, T)
    cov = fbm_covariance(t, hurst)
    rng = np.random.default_rng(seed)
    L = np.linalg.cholesky(cov + 1e-10 * np.eye(len(t)))
    z = rng.standard_normal((len(t), int(paths)))
    return (L @ z).T, t, cov


def covariance_error(paths: np.ndarray, target_cov: np.ndarray) -> tuple[np.ndarray, float]:
    emp = np.cov(paths, rowvar=False)
    diff = emp - target_cov
    return diff, float(np.sqrt(np.mean(diff ** 2)))


def fractional_kernel_matrix(steps: int = 300, hurst: float = 0.5, T: float = 1.0) -> tuple[np.ndarray, np.ndarray]:
    t = time_grid(steps, T)
    dt = t[1] - t[0]
    K = np.zeros((steps, steps))
    if abs(hurst - 0.5) < 1e-12:
        K[np.tril_indices(steps, -1)] = 1.0
        return t, K
    c_h = gamma(2 * hurst + 1) / (2.0 * gamma(hurst + 0.5) ** 2)
    for i in range(1, steps):
        lag = np.maximum(t[i] - t[:i], dt)
        K[i, :i] = np.sqrt(c_h) * lag ** (hurst - 0.5)
    return t, K


def simulate_volterra(
    steps: int = 300,
    hurst: float = 0.5,
    paths: int = 10,
    T: float = 1.0,
    seed: int = 77,
) -> tuple[np.ndarray, np.ndarray, np.ndarray, np.ndarray]:
    t, K = fractional_kernel_matrix(steps, hurst, T)
    dt = t[1] - t[0]
    rng = np.random.default_rng(seed)
    dW = rng.normal(0.0, np.sqrt(dt), size=(int(paths), steps))
    X = dW @ K.T
    W = np.cumsum(dW, axis=1)
    return X, W, t, K


def lift_rates_weights(
    factors: int = 10,
    hurst: float = 0.5,
    T: float = 1.0,
    lambda_min: float | None = None,
    lambda_max: float | None = None,
) -> tuple[np.ndarray, np.ndarray]:
    lo = 2.0 / T if lambda_min is None else lambda_min
    hi = 400.0 / T if lambda_max is None else lambda_max
    lam = np.logspace(np.log10(lo), np.log10(hi), int(factors))
    w = lam ** (-(hurst + 0.5))
    w /= w.sum()
    return lam, w


def ou_innovation_cov(lam: np.ndarray, dt: float) -> np.ndarray:
    lam_sum = lam[:, None] + lam[None, :]
    return (1.0 - np.exp(-lam_sum * dt)) / lam_sum


def simulate_markov_lift(
    steps: int = 300,
    hurst: float = 0.5,
    paths: int = 10,
    factors: int = 10,
    T: float = 1.0,
    seed: int = 101,
    match_marginals: bool = True,
) -> tuple[np.ndarray, np.ndarray, np.ndarray, np.ndarray]:
    t = time_grid(steps, T)
    dt = t[1] - t[0]
    lam, w = lift_rates_weights(factors, hurst, T)
    chol = np.linalg.cholesky(ou_innovation_cov(lam, dt) + 1e-12 * np.eye(len(lam)))
    a = np.exp(-lam * dt)
    rng = np.random.default_rng(seed)

    Y = np.zeros((int(paths), steps))
    for p in range(int(paths)):
        U = np.zeros(len(lam))
        for n in range(steps - 1):
            U = a * U + chol @ rng.standard_normal(len(lam))
            Y[p, n + 1] = w @ U

    if match_marginals and paths > 1:
        target_var = t ** (2 * hurst)
        emp_var = np.var(Y, axis=0, ddof=1)
        mask = (t > 0.0) & (emp_var > 1e-14)
        scale = np.ones_like(t)
        scale[mask] = np.sqrt(target_var[mask] / emp_var[mask])
        Y = Y * scale

    return Y, t, lam, w


def ensure_dir(path: str | Path) -> Path:
    out = Path(path)
    out.mkdir(parents=True, exist_ok=True)
    return out
