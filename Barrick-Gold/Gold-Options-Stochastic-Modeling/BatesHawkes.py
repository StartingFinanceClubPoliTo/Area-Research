import numpy as np
from scipy.optimize import differential_evolution, minimize

from Bates import Bates
from Hawkes import Hawkes


class BatesHawkes(Bates):
    """
    Bates-Hawkes calibration utilities.

    A full Bates-Hawkes option-pricing engine requires an event-dependent
    self-exciting jump intensity. The pricing approximation implemented here
    uses the stationary mean Hawkes intensity as an effective Bates intensity:

        lambda_eff = lambda0 / (1 - alpha / beta).

    This is a calibration-oriented proxy. It lets the same dataset, vega-weighted
    objective, and SLSQP workflow compare a constant-intensity Bates jump layer
    with a stationary self-exciting jump-intensity candidate, while clearly
    leaving exact Hawkes pricing to future work.
    """

    @staticmethod
    def effective_intensity(lambda0, alpha, beta):
        """Map a stationary Hawkes process to its long-run mean intensity."""
        return Hawkes.stationary_mean_intensity(lambda0, alpha, beta)

    @staticmethod
    def price_proxy_fast(S0, K, T, v0, kappa, theta, sigma, rho,
                         lambda0, alpha, beta, mu_J, sigma_J, r, q=0.0):
        """Price with Bates using the stationary mean Hawkes intensity."""
        lambda_eff = BatesHawkes.effective_intensity(lambda0, alpha, beta)
        return Bates.bates_price_fast(
            S0, K, T, v0, kappa, theta, sigma, rho, lambda_eff, mu_J, sigma_J, r, q
        )

    @staticmethod
    def bates_hawkes_proxy_objective(params, df_market, S0, q=0.0):
        """
        Vega-weighted objective for the Bates-Hawkes stationary-intensity proxy.

        params array:
        [v0, kappa, theta, sigma, rho, lambda0, alpha, beta, mu_J, sigma_J]
        """
        v0, kappa, theta, sigma, rho, lambda0, alpha, beta, mu_J, sigma_J = params

        if v0 <= 0 or kappa <= 0 or theta <= 0 or sigma <= 0:
            return 1e8
        if not (-0.999 <= rho <= 0.999):
            return 1e8
        if lambda0 <= 0 or alpha < 0 or beta <= 0 or alpha >= beta:
            return 1e8
        if sigma_J <= 0:
            return 1e8

        branching_ratio = alpha / beta
        if branching_ratio >= 0.98:
            return 1e8

        error = 0.0
        for _, row in df_market.iterrows():
            model_price = BatesHawkes.price_proxy_fast(
                S0,
                row["K"],
                row["T"],
                v0,
                kappa,
                theta,
                sigma,
                rho,
                lambda0,
                alpha,
                beta,
                mu_J,
                sigma_J,
                row["rate"],
                q,
            )
            safe_vega = max(row["vega"], 1e-4)
            error += ((model_price - row["price"]) / safe_vega) ** 2

        # Mild regularization keeps the proxy away from nearly critical regimes
        # unless the surface fit strongly justifies it.
        penalty = 0.01 * branching_ratio**2 / max(1.0 - branching_ratio, 1e-4)
        return error / len(df_market) + penalty

    @staticmethod
    def calibrate_bates_hawkes_proxy(df_market, S0, q=0.0, maxiter=80, popsize=8):
        """Calibrate the stationary-intensity Bates-Hawkes proxy."""
        bounds = [
            (1e-4, 1.0),   # v0
            (0.1, 10.0),   # kappa
            (1e-4, 1.0),   # theta
            (0.01, 15.0),  # sigma
            (-0.99, 0.99), # rho
            (0.01, 3.0),   # lambda0
            (0.01, 4.0),   # alpha
            (0.05, 6.0),   # beta
            (-0.5, 0.5),   # mu_J
            (1e-4, 0.5),   # sigma_J
        ]

        constraints = (
            {
                "type": "ineq",
                "fun": lambda x: x[7] - x[6] - 1e-4,  # beta > alpha
            },
        )

        print("[INFO] Starting Global Calibration for Bates-Hawkes proxy...")
        result_global = differential_evolution(
            BatesHawkes.bates_hawkes_proxy_objective,
            bounds=bounds,
            args=(df_market, S0, q),
            maxiter=maxiter,
            popsize=popsize,
            tol=1e-3,
            polish=False,
            disp=True,
        )

        print("\n[INFO] Global candidate found. Starting Local Refinement (SLSQP)...")
        result_local = minimize(
            BatesHawkes.bates_hawkes_proxy_objective,
            x0=result_global.x,
            args=(df_market, S0, q),
            method="SLSQP",
            bounds=bounds,
            constraints=constraints,
            options={"ftol": 1e-6, "maxiter": 120},
        )

        lambda0, alpha, beta = result_local.x[5], result_local.x[6], result_local.x[7]
        lambda_eff = BatesHawkes.effective_intensity(lambda0, alpha, beta)

        print("\n===================================================")
        print("  OPTIMAL BATES-HAWKES PROXY PARAMETERS")
        print("===================================================")
        labels = [
            "v0", "kappa", "theta", "sigma", "rho",
            "lambda0", "alpha", "beta", "mu_J", "sigma_J",
        ]
        for label, value in zip(labels, result_local.x):
            print(f"{label:10s}: {value:.6f}")
        print(f"branching : {alpha / beta:.6f}")
        print(f"lambda_eff: {lambda_eff:.6f}")
        print("===================================================")

        return result_local.x
