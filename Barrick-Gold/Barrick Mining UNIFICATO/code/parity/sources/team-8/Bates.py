"""Bates option pricing and reproducible surface calibration.

The scalar Fourier price remains the compatibility/reference implementation.
Calibration uses a vectorised COS batch by default and validates the market
surface once, outside the optimiser's hot loop.
"""

from dataclasses import dataclass
from time import perf_counter

import numpy as np
from scipy.optimize import differential_evolution, minimize

from Heston import Heston
from calibration_core import (
    CalibrationReport,
    OptionSurface,
    feller_feasible_population,
)
from fourier_pricing import adaptive_cos_call_prices, carr_madan_call


@dataclass(frozen=True)
class BatesParameters:
    """Named, immutable Bates parameter vector."""

    v0: float
    kappa: float
    theta: float
    sigma: float
    rho: float
    lambd: float
    mu_J: float
    sigma_J: float

    @classmethod
    def from_array(cls, values):
        values = tuple(float(value) for value in values)
        if len(values) != 8:
            raise ValueError("Bates requires exactly eight parameters.")
        return cls(*values)

    def as_tuple(self):
        return tuple(getattr(self, name) for name in Bates.PARAMETER_NAMES)


class Bates(Heston):
    """Bates (1996) model with scalar and maturity-batched pricing APIs."""

    PARAMETER_NAMES = (
        "v0", "kappa", "theta", "sigma", "rho", "lambd", "mu_J", "sigma_J"
    )
    BOUNDS = (
        (1e-4, 1.0),
        (0.1, 10.0),
        (1e-4, 1.0),
        (0.01, 15.0),
        (-0.99, 0.99),
        (0.0, 5.0),
        (-0.5, 0.5),
        (1e-4, 0.5),
    )
    INVALID_OBJECTIVE = 1e8

    @staticmethod
    def bates_charfunc(
        u, S0, v0, kappa, theta, sigma, rho, lambd, mu_J, sigma_J, T, r, q=0.0
    ):
        """Characteristic function of log-price under Bates."""
        cf_heston = Heston.heston_charfunc(
            u, S0, v0, kappa, theta, sigma, rho, T, r, q
        )
        omega = -lambd * (np.exp(mu_J + 0.5 * sigma_J**2) - 1.0)
        jump_term = lambd * (
            np.exp(1j * u * mu_J - 0.5 * u**2 * sigma_J**2) - 1.0
        )
        return cf_heston * np.exp((1j * u * omega + jump_term) * T)

    @staticmethod
    def _charfunc_for(S0, T, r, q, parameters):
        return lambda u: Bates.bates_charfunc(
            u, S0, *parameters.as_tuple(), T, r, q
        )

    @staticmethod
    def _integrand_fast_bates(
        u, S0, K, v0, kappa, theta, sigma, rho, lambd, mu_J, sigma_J, T, r, q=0.0
    ):
        """Compatibility integrand used by the historical notebooks."""
        parameters = BatesParameters(
            v0, kappa, theta, sigma, rho, lambd, mu_J, sigma_J
        )
        charfunc = Bates._charfunc_for(S0, T, r, q, parameters)
        numerator = np.exp(-r * T) * (charfunc(u - 1j) - K * charfunc(u))
        return numerator / (1j * u * K ** (1j * u))

    @staticmethod
    def bates_price_fast(
        S0, K, T, v0, kappa, theta, sigma, rho, lambd, mu_J, sigma_J, r, q=0.0
    ):
        """Reference European-call price via scalar Fourier inversion."""
        parameters = BatesParameters(
            v0, kappa, theta, sigma, rho, lambd, mu_J, sigma_J
        )
        return carr_madan_call(
            Bates._charfunc_for(S0, T, r, q, parameters), S0, K, T, r, q
        )

    @staticmethod
    def bates_prices_cos(
        S0, K, T, v0, kappa, theta, sigma, rho, lambd, mu_J, sigma_J,
        r, q=0.0, N=256, L=12.0
    ):
        """Price all strikes for one maturity from one COS transform grid."""
        strikes = np.atleast_1d(np.asarray(K, dtype=float))
        if T <= 0.0:
            return np.maximum(S0 - strikes, 0.0)
        parameters = BatesParameters(
            v0, kappa, theta, sigma, rho, lambd, mu_J, sigma_J
        )
        return adaptive_cos_call_prices(
            Bates._charfunc_for(S0, T, r, q, parameters),
            strikes,
            T,
            r,
            terms=N,
            width_scale=L,
        )

    @staticmethod
    def _admissible(parameters):
        p = parameters
        return (
            p.v0 > 0.0
            and p.kappa > 0.0
            and p.theta > 0.0
            and p.sigma > 0.0
            and -0.999 <= p.rho <= 0.999
            and 2.0 * p.kappa * p.theta - p.sigma**2 >= 0.0
            and p.lambd >= 0.0
            and p.sigma_J > 0.0
        )

    @staticmethod
    def bates_objective(params, df_market, S0, q=0.0, pricing="cos", cos_N=256):
        """Vega-weighted MSE on a DataFrame or prepared ``OptionSurface``."""
        try:
            parameters = BatesParameters.from_array(params)
        except (TypeError, ValueError):
            return Bates.INVALID_OBJECTIVE
        if not Bates._admissible(parameters):
            return Bates.INVALID_OBJECTIVE

        surface = OptionSurface.from_frame(df_market)
        squared_error = 0.0
        try:
            for maturity_slice in surface.slices:
                if pricing == "quad":
                    model_prices = np.asarray(
                        [
                            Bates.bates_price_fast(
                                S0, strike, maturity_slice.maturity,
                                *parameters.as_tuple(), maturity_slice.rate, q
                            )
                            for strike in maturity_slice.strikes
                        ]
                    )
                elif pricing == "cos":
                    model_prices = Bates.bates_prices_cos(
                        S0,
                        maturity_slice.strikes,
                        maturity_slice.maturity,
                        *parameters.as_tuple(),
                        maturity_slice.rate,
                        q,
                        N=cos_N,
                    )
                else:
                    raise ValueError("pricing must be 'cos' or 'quad'.")
                if not np.all(np.isfinite(model_prices)):
                    return Bates.INVALID_OBJECTIVE
                residuals = (
                    (model_prices - maturity_slice.market_prices)
                    / maturity_slice.safe_vegas
                )
                squared_error += float(np.sum(residuals**2))
        except (FloatingPointError, OverflowError, ValueError):
            return Bates.INVALID_OBJECTIVE
        return squared_error / surface.size

    @staticmethod
    def calibrate_bates(
        df_market,
        S0,
        q=0.0,
        maxiter=150,
        popsize=12,
        seed=None,
        pricing="cos",
        cos_N=256,
        disp=True,
        return_report=False,
    ):
        """Calibrate with differential evolution followed by constrained SLSQP.

        The default return value remains the historical NumPy parameter vector.
        Set ``return_report=True`` for diagnostics and reproducibility metadata.
        """
        started_at = perf_counter()
        surface = OptionSurface.from_frame(df_market)
        objective_args = (surface, S0, q, pricing, cos_N)
        initial_population = feller_feasible_population(
            Bates.BOUNDS, popsize, seed
        )

        result_global = differential_evolution(
            Bates.bates_objective,
            bounds=Bates.BOUNDS,
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
            {
                "type": "ineq",
                "fun": lambda x: 2.0 * x[1] * x[2] - x[3] ** 2,
            },
        )
        result_local = minimize(
            Bates.bates_objective,
            x0=result_global.x,
            args=objective_args,
            method="SLSQP",
            bounds=Bates.BOUNDS,
            constraints=constraints,
            options={"ftol": 1e-6, "maxiter": 100, "disp": disp},
        )
        report = CalibrationReport.from_optimizer(
            "Bates", Bates.PARAMETER_NAMES, result_local, result_global, started_at
        )
        return report if return_report else report.x
