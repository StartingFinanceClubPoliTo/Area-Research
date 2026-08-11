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
from BatesHawkesExact import BatesHawkesExact  # noqa: E402
from Hawkes import ExactHawkesCalibration  # noqa: E402
from calibration_workflow import (  # noqa: E402
    load_calibration_surface,
    option_diagnostics,
    plot_residuals,
    plot_smiles,
    price_surface,
)


SEED = 20260811
BATES_SEED = (0.0791, 1.7164, 0.0428, 0.3833, 0.2206, 0.8963, -0.1901, 0.1006)


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


def main():
    data_dir = ROOT / "Data"
    surface, spot = load_calibration_surface(data_dir)
    result = ExactHawkesCalibration.calibrate_heston(
        surface,
        spot,
        bates_seed=BATES_SEED,
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
            spot, strikes, maturity, *BATES_SEED, rate, N=256
        ),
    )
    bates_diagnostics = option_diagnostics(surface, spot, bates_prices)
    full_diagnostics.to_csv(
        data_dir / "bates_hawkes_option_diagnostics.csv", index=False
    )

    metrics = {
        "full_bates_hawkes": error_metrics(full_diagnostics),
        "bates": error_metrics(bates_diagnostics),
        "calibration_objective": float(result.fun),
        "bates_objective": float(Bates.bates_objective(BATES_SEED, surface, spot)),
        "branching_ratio": parameters["branching_ratio"],
        "minimum_branching_ratio": 0.02,
        "observations": int(len(surface)),
        "status": "feller_constrained_restructured_workflow",
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
        "status": "feller_constrained_restructured_workflow",
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
