"""Regenerate constrained full Bates--Hawkes calibration diagnostics."""

import json
from pathlib import Path
import sys

import matplotlib

matplotlib.use("Agg")
import matplotlib.pyplot as plt
import numpy as np
import pandas as pd

ROOT = Path(__file__).resolve().parents[1]
if str(ROOT) not in sys.path:
    sys.path.insert(0, str(ROOT))

from Bates import Bates  # noqa: E402
from BatesHawkes import BatesHawkes  # noqa: E402
from BatesHawkesExact import BatesHawkesExact  # noqa: E402
from Hawkes import ExactHawkesCalibration  # noqa: E402
from calibration_workflow import (  # noqa: E402
    load_calibration_surface,
    normality_statistics,
    option_diagnostics,
    plot_residuals,
    plot_smiles,
    price_surface,
)


SEED = 20260811


def error_metrics(diagnostics):
    price_errors = diagnostics["model_price"] - diagnostics["price"]
    iv_errors = diagnostics["model_iv"] - diagnostics["implied_vol"]
    valid_iv = np.isfinite(iv_errors)
    return {
        "price_mae": float(np.mean(np.abs(price_errors))),
        "price_rmse": float(np.sqrt(np.mean(price_errors**2))),
        "iv_mae_bp": float(np.mean(np.abs(iv_errors[valid_iv])) * 10_000.0),
        "iv_rmse_bp": float(np.sqrt(np.mean(iv_errors[valid_iv] ** 2)) * 10_000.0),
        "max_abs_price_error": float(np.max(np.abs(price_errors))),
        "max_abs_iv_error_bp": float(np.max(np.abs(iv_errors[valid_iv])) * 10_000.0),
    }


def exact_prices(surface, spot, p):
    return price_surface(
        surface,
        lambda strikes, maturity, rate: BatesHawkesExact.hawkes_price_cos(
            spot,
            strikes,
            maturity,
            p["v0"],
            p["kappa"],
            p["theta"],
            p["xi"],
            p["rho"],
            p["lambda0"],
            p["lambda_bar"],
            p["alpha"],
            p["beta"],
            p["mu_J"],
            p["sigma_J"],
            rate,
            0.0,
            N=256,
        ),
    )


def comparison_figure(full_diagnostics, bates_diagnostics, output_path):
    figure, axes = plt.subplots(1, 2, figsize=(12, 5))
    axes[0].scatter(
        full_diagnostics["price"], full_diagnostics["model_price"],
        s=30, alpha=0.8, label="Full Bates--Hawkes"
    )
    axes[0].scatter(
        bates_diagnostics["price"], bates_diagnostics["model_price"],
        s=24, alpha=0.7, label="Bates"
    )
    bounds = [0.0, float(full_diagnostics["price"].max())]
    axes[0].plot(bounds, bounds, "k--", linewidth=1.0)
    axes[0].set(xlabel="Market call price", ylabel="Model call price")
    axes[0].legend()
    axes[0].grid(True, alpha=0.25)

    axes[1].scatter(
        full_diagnostics["T"], full_diagnostics["iv_residual_pct"],
        s=30, alpha=0.8, label="Full Bates--Hawkes"
    )
    axes[1].scatter(
        bates_diagnostics["T"], bates_diagnostics["iv_residual_pct"],
        s=24, alpha=0.7, label="Bates"
    )
    axes[1].axhline(0.0, color="black", linestyle="--", linewidth=1.0)
    axes[1].set(xlabel="Time to maturity", ylabel="Market - model IV (pp)")
    axes[1].legend()
    axes[1].grid(True, alpha=0.25)
    figure.tight_layout()
    figure.savefig(output_path, dpi=200, bbox_inches="tight")
    plt.close(figure)


def rebuild_constant_volatility_benchmark(surface, spot, data_dir):
    """Recalibrate the exact constant-volatility Hawkes method check on LSE."""
    result = ExactHawkesCalibration.calibrate_constvol(
        surface, spot, maxiter=25, popsize=6, seed=SEED
    )
    sigma, lambda_bar, alpha, beta, mu_j, sigma_j = map(float, result.x)
    branching = alpha / beta
    parameters = {
        "sigma": sigma,
        "lambda_bar": lambda_bar,
        "lambda0": lambda_bar,
        "alpha": alpha,
        "beta": beta,
        "branching_ratio": branching,
        "mu_J": mu_j,
        "sigma_J": sigma_j,
    }
    payload = {
        "model": "Exact constant-volatility Bates-Hawkes",
        "parameters": parameters,
        "final_objective": float(result.fun),
        "optimizer_success": bool(result.success),
        "source": "current LSE GLD option chain",
        "note": "Exact COS benchmark; lambda0 is tied to lambda_bar and alpha/beta < 1.",
    }
    (data_dir / "hawkes_exact_constvol_params.json").write_text(
        json.dumps(payload, indent=2), encoding="utf-8"
    )

    exact = option_diagnostics(
        surface,
        spot,
        price_surface(
            surface,
            lambda strikes, maturity, rate: BatesHawkesExact.hawkes_price_constvol_cos(
                spot, strikes, maturity, sigma, lambda_bar, lambda_bar,
                alpha, beta, mu_j, sigma_j, rate, N=256
            ),
        ),
    )
    proxy = option_diagnostics(
        surface,
        spot,
        price_surface(
            surface,
            lambda strikes, maturity, rate: BatesHawkes.prices_proxy_cos(
                spot, strikes, maturity, sigma**2, 5.0, sigma**2, 0.01, 0.0,
                lambda_bar, alpha, beta, mu_j, sigma_j, rate, N=256
            ),
        ),
    )
    figure, axes = plt.subplots(1, 2, figsize=(12, 4.8))
    axes[0].scatter(exact["price"], exact["model_price"], s=28, label="Exact Hawkes")
    axes[0].scatter(proxy["price"], proxy["model_price"], s=22, label="Stationary proxy")
    limit = float(exact["price"].max())
    axes[0].plot([0.0, limit], [0.0, limit], "k--", linewidth=1.0)
    axes[0].set(xlabel="LSE-IV normalized call price", ylabel="Model price")
    axes[0].legend()
    axes[0].grid(True, alpha=0.25)
    axes[1].scatter(exact["T"], exact["iv_residual_pct"], s=28, label="Exact Hawkes")
    axes[1].scatter(proxy["T"], proxy["iv_residual_pct"], s=22, label="Stationary proxy")
    axes[1].axhline(0.0, color="black", linestyle="--", linewidth=1.0)
    axes[1].set(xlabel="Maturity", ylabel="Market - model IV (pp)")
    axes[1].legend()
    axes[1].grid(True, alpha=0.25)
    figure.tight_layout()
    figure.savefig(data_dir / "hawkes_exact_vs_proxy.png", dpi=220, bbox_inches="tight")
    plt.close(figure)
    return payload


def main():
    data_dir = ROOT / "Data"
    surface, spot = load_calibration_surface(data_dir / "lse_local")
    bates_payload = json.loads(
        (data_dir / "bates_calibrated_params.json").read_text(encoding="utf-8")
    )
    bates_parameters = bates_payload["parameters"]
    bates_seed = tuple(
        float(bates_parameters[name])
        for name in ("v0", "kappa", "theta", "sigma", "rho", "lambd", "mu_J", "sigma_J")
    )
    rebuild_constant_volatility_benchmark(surface, spot, data_dir)
    result = ExactHawkesCalibration.calibrate_heston(
        surface,
        spot,
        bates_seed=bates_seed,
        maxiter=25,
        popsize=6,
        seed=SEED,
        global_cos_N=128,
        local_cos_N=192,
        min_branching=0.02,
    )
    parameters = ExactHawkesCalibration.unpack_heston_params(result.x)
    parameters["feller_gap"] = (
        2.0 * parameters["kappa"] * parameters["theta"] - parameters["xi"] ** 2
    )
    if not result.success or parameters["feller_gap"] < -1e-10:
        raise RuntimeError(
            f"Constrained calibration failed: {result.message}; {parameters}"
        )

    full_diagnostics = option_diagnostics(
        surface, spot, exact_prices(surface, spot, parameters)
    )
    bates_prices = price_surface(
        surface,
        lambda strikes, maturity, rate: Bates.bates_prices_cos(
            spot, strikes, maturity, *bates_seed, rate, N=256
        ),
    )
    bates_diagnostics = option_diagnostics(surface, spot, bates_prices)
    metrics = {
        "full_bates_hawkes": error_metrics(full_diagnostics),
        "bates": error_metrics(bates_diagnostics),
        "calibration_objective": float(result.fun),
        "bates_objective": float(Bates.bates_objective(bates_seed, surface, spot)),
        "branching_ratio": parameters["branching_ratio"],
        "minimum_branching_ratio": 0.02,
        "observations": int(len(surface)),
        "status": "lse_feller_constrained_restructured_workflow",
    }
    (data_dir / "bates_hawkes_calibration_metrics.json").write_text(
        json.dumps(metrics, indent=2), encoding="utf-8"
    )
    pd.DataFrame(
        [
            {"model": "Bates", **metrics["bates"]},
            {"model": "Full Bates-Hawkes", **metrics["full_bates_hawkes"]},
        ]
    ).to_csv(data_dir / "bates_hawkes_calibration_metrics.csv", index=False)

    residual_path = data_dir / "calibration_residual_normality.csv"
    residual_normality = pd.read_csv(residual_path)
    residual_normality = residual_normality.loc[
        residual_normality["model"] != "Full Bates-Hawkes"
    ].copy()
    full_normality = normality_statistics(full_diagnostics["iv_residual_pct"])
    full_normality.update({
        "model": "Full Bates-Hawkes",
        "residual": "market minus model implied volatility (pp)",
        "decision_5pct_shapiro": (
            "reject" if full_normality["shapiro_pvalue"] < 0.05 else "do not reject"
        ),
        "decision_5pct_jarque_bera": (
            "reject" if full_normality["jarque_bera_pvalue"] < 0.05 else "do not reject"
        ),
        "decision_5pct_dagostino": (
            "reject" if full_normality["dagostino_k2_pvalue"] < 0.05 else "do not reject"
        ),
    })
    residual_normality = pd.concat(
        [residual_normality, pd.DataFrame([full_normality])], ignore_index=True
    )
    residual_normality.to_csv(residual_path, index=False)

    payload = {
        "model": "Full affine Bates-Hawkes (Feller constrained)",
        "parameterization": (
            "lambda0 and lambda_bar are separate; "
            "alpha=branching_ratio*beta enforces stationarity"
        ),
        "parameters": parameters,
        "final_objective": float(result.fun),
        "dataset": {
            "instrument": "GLD calls",
            "observations": int(len(surface)),
            "maturities": int(surface["T"].nunique()),
            "spot": float(spot),
        },
        "calibration": {
            "seed": SEED,
            "global_cos_N": 128,
            "local_cos_N": 192,
            "final_cos_N": 256,
            "minimum_branching_ratio": 0.02,
        },
        "optimizer": {
            "success": bool(result.success),
            "message": str(result.message),
            "iterations": int(getattr(result, "nit", 0)),
            "evaluations": int(getattr(result, "nfev", 0)),
        },
        "status": "lse_feller_constrained_restructured_workflow",
    }
    (data_dir / "bates_hawkes_calibrated_params.json").write_text(
        json.dumps(payload, indent=2), encoding="utf-8"
    )

    plot_residuals(
        full_diagnostics,
        "Full Bates--Hawkes",
        data_dir / "bates_hawkes_residual_heatmap.png",
    )
    plot_smiles(
        full_diagnostics,
        "Full Bates--Hawkes",
        data_dir / "bates_hawkes_volatility_smile.png",
    )
    comparison_figure(
        full_diagnostics,
        bates_diagnostics,
        data_dir / "bates_hawkes_vs_bates.png",
    )
    print(json.dumps(payload, indent=2))
    print(json.dumps(metrics, indent=2))


if __name__ == "__main__":
    main()
