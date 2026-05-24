"""Random-number generation and diagnostic helpers."""

from __future__ import annotations

from typing import Tuple

import numpy as np
from scipy.special import erfinv
from scipy.stats import chi2


def lcg_uniform(
    n: int,
    seed: int,
    a: int = 1_103_515_245,
    c: int = 12_345,
    m: int = 2**31 - 1,
) -> np.ndarray:
    """Generate pseudo-random uniforms in (0, 1) with a linear congruential generator."""
    if n <= 0:
        raise ValueError("n must be positive")
    x = int(seed) % m
    u = np.empty(n, dtype=float)
    for i in range(n):
        x = (a * x + c) % m
        u[i] = x / m
    return np.clip(u, np.finfo(float).tiny, 1.0 - np.finfo(float).eps)


def inverse_exponential(u: np.ndarray, mean: float) -> np.ndarray:
    """Transform uniforms into exponential samples with the given mean."""
    if mean <= 0:
        raise ValueError("mean must be positive")
    u = np.asarray(u, dtype=float)
    return -mean * np.log(np.clip(u, np.finfo(float).tiny, 1.0))


def box_muller_from_uniforms(u1: np.ndarray, u2: np.ndarray) -> Tuple[np.ndarray, np.ndarray]:
    """Transform two independent uniform arrays into independent standard normals."""
    u1 = np.asarray(u1, dtype=float)
    u2 = np.asarray(u2, dtype=float)
    if u1.shape != u2.shape:
        raise ValueError("u1 and u2 must have the same shape")
    radius = np.sqrt(-2.0 * np.log(np.clip(u1, np.finfo(float).tiny, 1.0)))
    angle = 2.0 * np.pi * u2
    return radius * np.cos(angle), radius * np.sin(angle)


def sample_mvnormal_cholesky(
    n: int,
    mean: np.ndarray,
    covariance: np.ndarray,
    seed: int | None = None,
    z_std: np.ndarray | None = None,
) -> np.ndarray:
    """Sample a multivariate normal distribution using Cholesky factorization."""
    mean = np.asarray(mean, dtype=float).reshape(-1)
    covariance = np.asarray(covariance, dtype=float)
    if covariance.shape != (mean.size, mean.size):
        raise ValueError("covariance shape must match the mean vector")
    lower = np.linalg.cholesky(covariance)
    if z_std is None:
        rng = np.random.default_rng(seed)
        z_std = rng.standard_normal((n, mean.size))
    else:
        z_std = np.asarray(z_std, dtype=float)
        if z_std.shape != (n, mean.size):
            raise ValueError("z_std must have shape (n, len(mean))")
    return mean + z_std @ lower.T


def sample_mvnormal_pca(
    n: int,
    mean: np.ndarray,
    covariance: np.ndarray,
    keep_variance: float = 0.85,
    seed: int | None = None,
) -> Tuple[np.ndarray, int, np.ndarray]:
    """Sample a multivariate normal distribution using a PCA covariance factor."""
    mean = np.asarray(mean, dtype=float).reshape(-1)
    covariance = np.asarray(covariance, dtype=float)
    if keep_variance > 1.0:
        keep_variance = keep_variance / 100.0
    if not 0.0 < keep_variance <= 1.0:
        raise ValueError("keep_variance must be in (0, 1] or a percentage in (0, 100]")

    eigenvalues, eigenvectors = np.linalg.eigh(covariance)
    order = np.argsort(eigenvalues)[::-1]
    eigenvalues = np.clip(eigenvalues[order], 0.0, None)
    eigenvectors = eigenvectors[:, order]

    explained = np.cumsum(eigenvalues) / np.sum(eigenvalues)
    k = int(np.searchsorted(explained, keep_variance) + 1)
    factor = eigenvectors[:, :k] @ np.diag(np.sqrt(eigenvalues[:k]))
    rng = np.random.default_rng(seed)
    shocks = rng.standard_normal((n, k))
    return mean + shocks @ factor.T, k, explained


def chi_square_uniform_test(u: np.ndarray, bins: int = 30) -> Tuple[float, int, float]:
    """Return the chi-square statistic, degrees of freedom, and p-value for uniformity."""
    u = np.asarray(u, dtype=float)
    if bins < 2:
        raise ValueError("bins must be at least 2")
    observed, _ = np.histogram(u, bins=bins, range=(0.0, 1.0))
    expected = len(u) / bins
    statistic = float(np.sum((observed - expected) ** 2 / expected))
    dof = bins - 1
    p_value = float(chi2.sf(statistic, dof))
    return statistic, dof, p_value


def qq_data_standard_normal(z: np.ndarray, q: int = 101) -> Tuple[np.ndarray, np.ndarray, float]:
    """Build theoretical and empirical quantiles for a standard-normal QQ plot."""
    z = np.asarray(z, dtype=float)
    probabilities = np.linspace(0.01, 0.99, q)
    theoretical = np.sqrt(2.0) * erfinv(2.0 * probabilities - 1.0)
    empirical = np.quantile(z, probabilities)
    rmse = float(np.sqrt(np.mean((empirical - theoretical) ** 2)))
    return theoretical, empirical, rmse


def empirical_correlation(x: np.ndarray) -> np.ndarray:
    """Compute the sample correlation matrix of the columns in x."""
    x = np.asarray(x, dtype=float)
    if x.ndim != 2:
        raise ValueError("x must be a two-dimensional array")
    return np.corrcoef(x, rowvar=False)
