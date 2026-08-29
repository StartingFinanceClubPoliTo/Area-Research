"""Deterministic Nelson--Siegel--Svensson fitting for the LSE Treasury curve.

The original Team 8 notebooks used an NSS curve with hard-coded Treasury
observations.  This module preserves the method while replacing those values
with one coherent, versioned LSE observation date.  It deliberately has no
dependency on the standalone Team 4 demonstrator.
"""

from __future__ import annotations

from dataclasses import dataclass

import numpy as np
from scipy.optimize import least_squares


PARAMETER_NAMES = ("beta0", "beta1", "beta2", "beta3", "tau1", "tau2")


def _factor(maturities: np.ndarray, tau: float) -> tuple[np.ndarray, np.ndarray]:
    x = np.asarray(maturities, dtype=float) / float(tau)
    level = -np.expm1(-x) / x
    curvature = level - np.exp(-x)
    return level, curvature


def nss_zero_rates(maturities, parameters) -> np.ndarray:
    """Evaluate continuously compounded NSS zero-rate proxies."""

    times = np.asarray(maturities, dtype=float)
    if times.ndim != 1 or times.size == 0:
        raise ValueError("NSS maturities must be a non-empty vector")
    if np.any(~np.isfinite(times)) or np.any(times <= 0.0):
        raise ValueError("NSS maturities must be positive and finite")
    values = {name: float(parameters[name]) for name in PARAMETER_NAMES}
    if values["tau1"] <= 0.0 or values["tau2"] <= values["tau1"]:
        raise ValueError("NSS requires 0 < tau1 < tau2")
    level1, curvature1 = _factor(times, values["tau1"])
    _, curvature2 = _factor(times, values["tau2"])
    return (
        values["beta0"]
        + values["beta1"] * level1
        + values["beta2"] * curvature1
        + values["beta3"] * curvature2
    )


def _unpack(vector: np.ndarray) -> dict[str, float]:
    tau1 = float(np.exp(vector[4]))
    tau2 = tau1 + float(np.exp(vector[5]))
    return {
        "beta0": float(vector[0]),
        "beta1": float(vector[1]),
        "beta2": float(vector[2]),
        "beta3": float(vector[3]),
        "tau1": tau1,
        "tau2": tau2,
    }


@dataclass(frozen=True)
class NSSFit:
    parameters: dict[str, float]
    fitted_rates: np.ndarray
    residuals: np.ndarray
    rmse_bp: float
    max_abs_error_bp: float
    observations: int
    optimizer_cost: float

    def as_dict(self) -> dict[str, object]:
        return {
            "method": "Nelson-Siegel-Svensson nonlinear least squares",
            "rate_definition": "continuous proxy log(1 + Treasury par yield)",
            "parameters": dict(self.parameters),
            "observations": int(self.observations),
            "rmse_bp": float(self.rmse_bp),
            "max_abs_error_bp": float(self.max_abs_error_bp),
            "optimizer_cost": float(self.optimizer_cost),
            "constraints": {
                "beta_bounds": [-0.25, 0.25],
                "tau_order": "0 < tau1 < tau2",
            },
        }


def fit_nss_curve(maturities, observed_rates) -> NSSFit:
    """Fit NSS with deterministic multi-start constrained least squares."""

    times = np.asarray(maturities, dtype=float)
    rates = np.asarray(observed_rates, dtype=float)
    if times.ndim != 1 or rates.shape != times.shape:
        raise ValueError("NSS maturities and rates must be aligned vectors")
    if times.size < 8:
        raise ValueError("At least eight Treasury tenors are required for NSS")
    order = np.argsort(times)
    times, rates = times[order], rates[order]
    if np.any(np.diff(times) <= 0.0):
        raise ValueError("NSS maturities must be strictly increasing")
    if np.any(~np.isfinite(rates)) or np.any(np.abs(rates) > 0.25):
        raise ValueError("NSS rates must be finite and economically bounded")

    def residual(vector: np.ndarray) -> np.ndarray:
        return (nss_zero_rates(times, _unpack(vector)) - rates) * 10_000.0

    beta0 = float(rates[-1])
    beta1 = float(rates[0] - rates[-1])
    lower = np.array([-0.25, -0.25, -0.25, -0.25, np.log(0.02), np.log(0.02)])
    upper = np.array([0.25, 0.25, 0.25, 0.25, np.log(10.0), np.log(40.0)])
    candidates = []
    for tau1 in (0.10, 0.35, 0.75, 1.50, 3.00):
        for tau2 in (2.0, 5.0, 10.0, 20.0):
            if tau2 <= tau1:
                continue
            initial = np.array(
                [beta0, beta1, 0.01, 0.01, np.log(tau1), np.log(tau2 - tau1)]
            )
            candidates.append(
                least_squares(
                    residual,
                    initial,
                    bounds=(lower, upper),
                    max_nfev=20_000,
                    xtol=1e-13,
                    ftol=1e-13,
                    gtol=1e-13,
                )
            )
    best = min(candidates, key=lambda result: (float(np.sum(result.fun**2)), result.nfev))
    parameters = _unpack(best.x)
    fitted = nss_zero_rates(times, parameters)
    errors = fitted - rates
    rmse_bp = float(np.sqrt(np.mean(errors**2)) * 10_000.0)
    max_error_bp = float(np.max(np.abs(errors)) * 10_000.0)
    if not best.success or rmse_bp > 25.0:
        raise ValueError(
            f"NSS calibration failed quality control: success={best.success}, RMSE={rmse_bp:.3f} bp"
        )
    return NSSFit(
        parameters=parameters,
        fitted_rates=fitted,
        residuals=errors,
        rmse_bp=rmse_bp,
        max_abs_error_bp=max_error_bp,
        observations=int(times.size),
        optimizer_cost=float(best.cost),
    )


def quarterly_forward_rates(parameters, quarters: int, dt_years: float = 0.25) -> np.ndarray:
    """Convert the NSS zero curve into integral-preserving period forwards."""

    if quarters < 1 or dt_years <= 0.0:
        raise ValueError("quarters and dt_years must be positive")
    edges = np.arange(quarters + 1, dtype=float) * float(dt_years)
    positive_edges = edges.copy()
    positive_edges[0] = min(float(dt_years) / 1000.0, 1e-4)
    zero_rates = nss_zero_rates(positive_edges, parameters)
    integrated = edges * zero_rates
    integrated[0] = 0.0
    return np.diff(integrated) / float(dt_years)
