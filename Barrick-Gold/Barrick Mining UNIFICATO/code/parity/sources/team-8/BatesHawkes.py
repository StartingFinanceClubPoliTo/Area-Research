"""Stationary-intensity Bates--Hawkes proxy and calibration workflow."""

from dataclasses import dataclass
from time import perf_counter

import numpy as np
from scipy.optimize import differential_evolution, minimize

from Bates import Bates
from Hawkes import Hawkes
from calibration_core import (
    CalibrationReport,
    OptionSurface,
    feller_feasible_population,
)


@dataclass(frozen=True)
class BatesHawkesParameters:
    """Named, immutable parameter vector for the stationary Hawkes proxy."""

    v0: float
    kappa: float
    theta: float
    sigma: float
    rho: float
    lambda0: float
    alpha: float
    beta: float
    mu_J: float
    sigma_J: float

    @classmethod
    def from_array(cls, values):
        values = tuple(float(value) for value in values)
        if len(values) != 10:
            raise ValueError("Bates-Hawkes proxy requires ten parameters.")
        return cls(*values)

    def as_tuple(self):
        return tuple(getattr(self, name) for name in BatesHawkes.PARAMETER_NAMES)


class BatesHawkes(Bates):
    """Bates proxy using the stationary mean of a Hawkes intensity.

    This class deliberately remains distinct from ``BatesHawkesExact``: it is a
    transparent benchmark/proxy, not an event-dependent Hawkes option model.
    """

    PARAMETER_NAMES = (
        "v0", "kappa", "theta", "sigma", "rho",
        "lambda0", "alpha", "beta", "mu_J", "sigma_J",
    )
    BOUNDS = (
        (1e-4, 1.0),
        (0.1, 10.0),
        (1e-4, 1.0),
        (0.01, 15.0),
        (-0.99, 0.99),
        (0.01, 3.0),
        (0.01, 4.0),
        (0.05, 6.0),
        (-0.5, 0.5),
        (1e-4, 0.5),
    )
    MAX_BRANCHING = 0.98
    INVALID_OBJECTIVE = 1e8

    @staticmethod
    def effective_intensity(lambda0, alpha, beta):
        """Return the stationary mean intensity, requiring stationarity."""
        return Hawkes.stationary_mean_intensity(lambda0, alpha, beta)

    @staticmethod
    def price_proxy_fast(
        S0, K, T, v0, kappa, theta, sigma, rho,
        lambda0, alpha, beta, mu_J, sigma_J, r, q=0.0
    ):
        """Reference scalar proxy price."""
        lambda_eff = BatesHawkes.effective_intensity(lambda0, alpha, beta)
        return Bates.bates_price_fast(
            S0, K, T, v0, kappa, theta, sigma, rho,
            lambda_eff, mu_J, sigma_J, r, q
        )

    @staticmethod
    def prices_proxy_cos(
        S0, K, T, v0, kappa, theta, sigma, rho,
        lambda0, alpha, beta, mu_J, sigma_J, r, q=0.0, N=256, L=12.0
    ):
        """Maturity-batched proxy prices through the Bates COS engine."""
        lambda_eff = BatesHawkes.effective_intensity(lambda0, alpha, beta)
        return Bates.bates_prices_cos(
            S0, K, T, v0, kappa, theta, sigma, rho,
            lambda_eff, mu_J, sigma_J, r, q, N=N, L=L
        )

    @classmethod
    def _admissible(cls, p):
        branching = p.alpha / p.beta if p.beta > 0.0 else np.inf
        return (
            p.v0 > 0.0
            and p.kappa > 0.0
            and p.theta > 0.0
            and p.sigma > 0.0
            and 2.0 * p.kappa * p.theta - p.sigma**2 >= 0.0
            and -0.999 <= p.rho <= 0.999
            and p.lambda0 > 0.0
            and p.alpha >= 0.0
            and p.beta > 0.0
            and p.alpha < p.beta
            and branching < cls.MAX_BRANCHING
            and p.sigma_J > 0.0
        )

    @classmethod
    def bates_hawkes_proxy_objective(
        cls, params, df_market, S0, q=0.0, pricing="cos", cos_N=256
    ):
        """Vega-weighted proxy objective on a validated market surface."""
        try:
            p = BatesHawkesParameters.from_array(params)
        except (TypeError, ValueError):
            return cls.INVALID_OBJECTIVE
        if not cls._admissible(p):
            return cls.INVALID_OBJECTIVE

        surface = OptionSurface.from_frame(df_market)
        squared_error = 0.0
        try:
            for maturity_slice in surface.slices:
                if pricing == "quad":
                    model_prices = np.asarray(
                        [
                            cls.price_proxy_fast(
                                S0, strike, maturity_slice.maturity,
                                *p.as_tuple(), maturity_slice.rate, q
                            )
                            for strike in maturity_slice.strikes
                        ]
                    )
                elif pricing == "cos":
                    model_prices = cls.prices_proxy_cos(
                        S0,
                        maturity_slice.strikes,
                        maturity_slice.maturity,
                        *p.as_tuple(),
                        maturity_slice.rate,
                        q,
                        N=cos_N,
                    )
                else:
                    raise ValueError("pricing must be 'cos' or 'quad'.")
                if not np.all(np.isfinite(model_prices)):
                    return cls.INVALID_OBJECTIVE
                residuals = (
                    (model_prices - maturity_slice.market_prices)
                    / maturity_slice.safe_vegas
                )
                squared_error += float(np.sum(residuals**2))
        except (FloatingPointError, OverflowError, ValueError):
            return cls.INVALID_OBJECTIVE

        branching = p.alpha / p.beta
        regularisation = 0.01 * branching**2 / max(1.0 - branching, 1e-4)
        return squared_error / surface.size + regularisation

    @classmethod
    def calibrate_bates_hawkes_proxy(
        cls,
        df_market,
        S0,
        q=0.0,
        maxiter=80,
        popsize=8,
        seed=None,
        pricing="cos",
        cos_N=256,
        disp=True,
        return_report=False,
    ):
        """Calibrate the proxy with deterministic optional seed and diagnostics."""
        started_at = perf_counter()
        surface = OptionSurface.from_frame(df_market)
        objective_args = (surface, S0, q, pricing, cos_N)
        initial_population = feller_feasible_population(
            cls.BOUNDS, popsize, seed
        )
        result_global = differential_evolution(
            cls.bates_hawkes_proxy_objective,
            bounds=cls.BOUNDS,
            args=objective_args,
            maxiter=maxiter,
            popsize=popsize,
            tol=1e-3,
            polish=False,
            seed=seed,
            init=initial_population,
            disp=disp,
        )
        constraints = (
            {"type": "ineq", "fun": lambda x: x[7] - x[6] - 1e-4},
            {"type": "ineq", "fun": lambda x: 2.0 * x[1] * x[2] - x[3] ** 2},
        )
        result_local = minimize(
            cls.bates_hawkes_proxy_objective,
            x0=result_global.x,
            args=objective_args,
            method="SLSQP",
            bounds=cls.BOUNDS,
            constraints=constraints,
            options={"ftol": 1e-6, "maxiter": 120, "disp": disp},
        )
        report = CalibrationReport.from_optimizer(
            "Bates-Hawkes proxy",
            cls.PARAMETER_NAMES,
            result_local,
            result_global,
            started_at,
        )
        return report if return_report else report.x
