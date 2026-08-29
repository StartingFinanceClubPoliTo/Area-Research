"""Rebuild publication-safe BS, Heston, and Bates outputs from local LSE data.

The option chain, Treasury curve, history, and sampled rows remain below
``Data/lse_local`` and are ignored by Git. Only aggregate calibration results
and figures are written to ``Data``.
"""

import hashlib
import json
from pathlib import Path
import sys
from time import perf_counter

import matplotlib

matplotlib.use("Agg")
import matplotlib.pyplot as plt
import numpy as np
import pandas as pd
from scipy.optimize import differential_evolution, minimize, minimize_scalar
from scipy.interpolate import griddata
from scipy.stats import norm, probplot

ROOT = Path(__file__).resolve().parents[1]
if str(ROOT) not in sys.path:
    sys.path.insert(0, str(ROOT))

from Bates import Bates  # noqa: E402
from BnS import BnS  # noqa: E402
from Heston import Heston  # noqa: E402
from calibration_core import OptionSurface, feller_feasible_population  # noqa: E402
from calibration_workflow import (  # noqa: E402
    load_calibration_surface,
    normality_statistics,
    option_diagnostics,
    plot_residuals,
    plot_smiles,
    price_surface,
)
from fourier_pricing import adaptive_cos_call_prices  # noqa: E402


SEED = 20260811
HESTON_NAMES = ("v0", "kappa", "theta", "xi", "rho")
HESTON_BOUNDS = (
    (1e-4, 1.0), (0.1, 10.0), (1e-4, 1.0), (0.01, 8.0), (-0.99, 0.99)
)


def error_metrics(diagnostics):
    price_error = diagnostics["model_price"] - diagnostics["price"]
    iv_error = diagnostics["model_iv"] - diagnostics["implied_vol"]
    iv_error = iv_error[np.isfinite(iv_error)]
    return {
        "price_mae": float(np.mean(np.abs(price_error))),
        "price_rmse": float(np.sqrt(np.mean(price_error**2))),
        "iv_mae_bp": float(np.mean(np.abs(iv_error)) * 10_000.0),
        "iv_rmse_bp": float(np.sqrt(np.mean(iv_error**2)) * 10_000.0),
        "max_abs_price_error": float(np.max(np.abs(price_error))),
        "max_abs_iv_error_bp": float(np.max(np.abs(iv_error)) * 10_000.0),
    }


def heston_prices(surface, spot, params, terms=192):
    def batch(strikes, maturity, rate):
        characteristic = lambda u: Heston.heston_charfunc(
            u, spot, *params, maturity, rate, 0.0
        )
        return adaptive_cos_call_prices(
            characteristic, strikes, maturity, rate, terms=terms, width_scale=12.0
        )

    return price_surface(surface, batch)


def heston_objective(params, prepared, spot, terms=128):
    v0, kappa, theta, xi, rho = (float(value) for value in params)
    if (
        v0 <= 0.0 or kappa <= 0.0 or theta <= 0.0 or xi <= 0.0
        or not -0.999 < rho < 0.999
        or 2.0 * kappa * theta - xi**2 < 0.0
    ):
        return 1e8
    total = 0.0
    try:
        for maturity_slice in prepared.slices:
            characteristic = lambda u: Heston.heston_charfunc(
                u, spot, v0, kappa, theta, xi, rho,
                maturity_slice.maturity, maturity_slice.rate, 0.0
            )
            prices = adaptive_cos_call_prices(
                characteristic,
                maturity_slice.strikes,
                maturity_slice.maturity,
                maturity_slice.rate,
                terms=terms,
                width_scale=12.0,
            )
            if not np.all(np.isfinite(prices)):
                return 1e8
            residual = (
                prices - maturity_slice.market_prices
            ) / maturity_slice.safe_vegas
            total += float(np.sum(residual**2))
    except (FloatingPointError, OverflowError, ValueError):
        return 1e8
    return total / prepared.size


def calibrate_heston(surface, spot):
    prepared = OptionSurface.from_frame(surface)
    started = perf_counter()
    population = feller_feasible_population(HESTON_BOUNDS, popsize=6, seed=SEED)
    global_result = differential_evolution(
        heston_objective,
        HESTON_BOUNDS,
        args=(prepared, spot, 128),
        maxiter=25,
        popsize=6,
        tol=1e-3,
        polish=False,
        seed=SEED,
        init=population,
    )
    local_result = minimize(
        heston_objective,
        global_result.x,
        args=(prepared, spot, 192),
        method="SLSQP",
        bounds=HESTON_BOUNDS,
        constraints=({"type": "ineq", "fun": lambda x: 2*x[1]*x[2]-x[3]**2},),
        options={"ftol": 1e-7, "maxiter": 120},
    )
    payload = {
        "model": "Heston",
        "parameters": dict(zip(HESTON_NAMES, map(float, local_result.x))),
        "objective": float(local_result.fun),
        "global_objective": float(global_result.fun),
        "success": bool(local_result.success),
        "message": str(local_result.message),
        "evaluations": int(getattr(local_result, "nfev", 0)),
        "iterations": int(getattr(local_result, "nit", 0)),
        "elapsed_seconds": float(perf_counter() - started),
        "seed": SEED,
        "pricing": "Heston characteristic function with batched COS inversion",
    }
    p = payload["parameters"]
    p["feller_gap"] = 2.0 * p["kappa"] * p["theta"] - p["xi"] ** 2
    return np.asarray(local_result.x, dtype=float), payload


def black_scholes_fit(surface, spot):
    def objective(volatility):
        prices = np.asarray([
            BnS.bs_call_price(spot, row.K, row.T, row.rate, volatility)
            for row in surface.itertuples(index=False)
        ])
        return float(np.mean(((prices - surface["price"]) / surface["vega"]) ** 2))

    result = minimize_scalar(objective, bounds=(0.01, 1.5), method="bounded")
    volatility = float(result.x)
    prices = np.asarray([
        BnS.bs_call_price(spot, row.K, row.T, row.rate, volatility)
        for row in surface.itertuples(index=False)
    ])
    return volatility, option_diagnostics(surface, spot, prices)


def save_surface_figures(surface, spot, output_dir):
    figure, axis = plt.subplots(figsize=(9.5, 5.4))
    for expiry, group in surface.groupby("expiry"):
        group = group.sort_values("K")
        axis.plot(group["K"] / spot, group["implied_vol"] * 100.0, "o-", label=expiry)
    axis.set(xlabel="Moneyness (K / S0)", ylabel="LSE implied volatility (%)",
             title="Current LSE GLD implied-volatility smiles")
    axis.grid(True, alpha=0.25)
    axis.legend(ncol=2, fontsize=8)
    figure.tight_layout()
    figure.savefig(output_dir / "volatility_smiles_2d.png", dpi=220, bbox_inches="tight")
    plt.close(figure)

    figure = plt.figure(figsize=(10.2, 6.4))
    axis = figure.add_subplot(111, projection="3d")
    moneyness = surface["K"].to_numpy(dtype=float) / spot
    maturity = surface["T"].to_numpy(dtype=float)
    volatility = surface["implied_vol"].to_numpy(dtype=float) * 100.0
    money_grid, maturity_grid = np.meshgrid(
        np.linspace(float(moneyness.min()), float(moneyness.max()), 55),
        np.linspace(float(maturity.min()), float(maturity.max()), 55),
    )
    volatility_grid = griddata(
        np.column_stack((moneyness, maturity)), volatility,
        (money_grid, maturity_grid), method="linear",
    )
    surface_plot = axis.plot_surface(
        money_grid, maturity_grid, volatility_grid, cmap="viridis",
        linewidth=0.16, edgecolor=(0.15, 0.15, 0.15, 0.20),
        antialiased=True, alpha=0.94,
    )
    axis.scatter(moneyness, maturity, volatility, color="black", s=10, alpha=0.75)
    axis.set(xlabel="Moneyness (K / S0)", ylabel="Maturity (years)",
             zlabel="LSE IV (%)", title="Current LSE GLD volatility surface")
    axis.view_init(elev=27, azim=-126)
    figure.colorbar(surface_plot, ax=axis, pad=0.1, label="Implied volatility (%)")
    figure.tight_layout()
    figure.savefig(output_dir / "volatility_surface_3d.png", dpi=220, bbox_inches="tight")
    plt.close(figure)


def save_rate_curve(rate_curve, output_path):
    figure, axis = plt.subplots(figsize=(8.8, 4.8))
    from nss_curve import PARAMETER_NAMES, nss_zero_rates

    observed = rate_curve.get("observed_continuous_rate", rate_curve["continuous_rate"])
    axis.scatter(
        rate_curve["maturity_years"], observed * 100.0,
        s=46, color="#315A7D", edgecolor="white", linewidth=0.7,
        zorder=3, label="LSE observed tenors",
    )
    if all(name in rate_curve for name in PARAMETER_NAMES):
        parameters = {name: float(rate_curve[name].iloc[0]) for name in PARAMETER_NAMES}
        grid = np.geomspace(
            max(float(rate_curve["maturity_years"].min()), 1.0 / 365.25),
            float(rate_curve["maturity_years"].max()),
            500,
        )
        axis.plot(
            grid, nss_zero_rates(grid, parameters) * 100.0,
            linewidth=2.2, color="#D97706", label="Updated NSS fit",
        )
    else:
        axis.plot(
            rate_curve["maturity_years"], rate_curve["continuous_rate"] * 100.0,
            linewidth=2.0, color="#D97706", label="Interpolated curve",
        )
    for row in rate_curve.itertuples(index=False):
        axis.annotate(
            row.maturity, (row.maturity_years, row.continuous_rate * 100.0),
            xytext=(0, 7), textcoords="offset points", ha="center", fontsize=8,
        )
    curve_date = str(rate_curve["date"].iloc[0])
    axis.set(
        title=f"Updated LSE USD Treasury NSS curve ({curve_date})",
        xlabel="Maturity (years)", ylabel="Continuously compounded rate proxy (%)",
    )
    axis.grid(True, alpha=0.25)
    axis.legend()
    figure.tight_layout()
    figure.savefig(output_path, dpi=220, bbox_inches="tight")
    plt.close(figure)


def save_return_diagnostics(history, output_dir):
    returns = history["log_return"].dropna().to_numpy(dtype=float) * 100.0
    stats = normality_statistics(returns)
    stats.update({
        "series": "GLD daily log return (%)",
        "source": "London Strategic Edge /candles, 1d",
        "sample_start": str(history["timestamp"].iloc[1])[:10],
        "sample_end": str(history["timestamp"].iloc[-1])[:10],
        "null_hypothesis": "daily log returns are normally distributed",
        "decision_5pct_shapiro": "reject" if stats["shapiro_pvalue"] < 0.05 else "do not reject",
        "decision_5pct_jarque_bera": "reject" if stats["jarque_bera_pvalue"] < 0.05 else "do not reject",
        "decision_5pct_dagostino": "reject" if stats["dagostino_k2_pvalue"] < 0.05 else "do not reject",
    })
    pd.DataFrame([stats]).to_csv(output_dir / "gld_return_normality_tests.csv", index=False)
    (output_dir / "gld_return_normality_tests.json").write_text(
        json.dumps(stats, indent=2), encoding="utf-8"
    )

    figure, axes = plt.subplots(1, 2, figsize=(11.4, 4.6))
    x = np.linspace(float(np.min(returns)), float(np.max(returns)), 400)
    axes[0].hist(returns, bins=42, density=True, alpha=0.68, color="#4C78A8",
                 edgecolor="white", label="LSE GLD returns")
    axes[0].plot(x, norm.pdf(x, np.mean(returns), np.std(returns, ddof=1)),
                 color="#D62728", linewidth=2.0, label="Matched normal density")
    axes[0].set(xlabel="Daily log return (%)", ylabel="Density",
                title="Empirical returns versus a normal benchmark")
    axes[0].legend()
    axes[0].grid(True, alpha=0.2)
    (theoretical, ordered), (slope, intercept, _) = probplot(returns, dist="norm")
    axes[1].scatter(theoretical, ordered, s=13, alpha=0.58, color="#315A7D")
    axes[1].plot(theoretical, slope * theoretical + intercept, color="#D62728", linewidth=1.7)
    axes[1].set(xlabel="Theoretical normal quantiles", ylabel="Observed return quantiles (%)",
                title="Normal Q--Q plot")
    axes[1].grid(True, alpha=0.2)
    figure.tight_layout()
    figure.savefig(output_dir / "gld_return_normality.png", dpi=220, bbox_inches="tight")
    plt.close(figure)
    return stats

    figure, axis = plt.subplots(figsize=(9.5, 5.3))
    axis.scatter(surface["T"], surface["K"] / spot, c=surface["implied_vol"],
                 cmap="viridis", s=60, edgecolors="black", linewidth=0.3)
    axis.set(xlabel="Maturity (years)", ylabel="Moneyness (K / S0)",
             title="Chebyshev calibration sample from the current LSE chain")
    axis.grid(True, alpha=0.25)
    figure.tight_layout()
    figure.savefig(output_dir / "sampling_comparison.png", dpi=220, bbox_inches="tight")
    plt.close(figure)


def residual_comparison(diagnostics, output_path):
    models = list(diagnostics)
    figure, axes = plt.subplots(1, len(models), figsize=(5.2 * len(models), 4.5), sharey=True)
    for axis, model in zip(np.atleast_1d(axes), models):
        frame = diagnostics[model]
        points = axis.scatter(frame["T"], frame["moneyness"],
                              c=frame["iv_residual_pct"], cmap="coolwarm",
                              vmin=-5.0, vmax=5.0, s=42)
        axis.set(title=model, xlabel="Maturity", ylabel="K / S0")
        axis.grid(True, alpha=0.2)
    figure.colorbar(points, ax=np.atleast_1d(axes).tolist(), label="Market - model IV (pp)")
    figure.subplots_adjust(wspace=0.22, right=0.90)
    figure.savefig(output_path, dpi=220, bbox_inches="tight")
    plt.close(figure)


def main():
    local_dir = ROOT / "Data" / "lse_local"
    output_dir = ROOT / "Data"
    surface, spot = load_calibration_surface(local_dir)
    metadata = json.loads((local_dir / "gld_lse_meta.json").read_text(encoding="utf-8"))
    sample_path = local_dir / "gld_lse_calibration_chebyshev.csv"
    rate_curve = pd.read_csv(local_dir / "usd_treasury_curve.csv")
    history = pd.read_csv(local_dir / "gld_daily_history.csv")

    save_surface_figures(surface, spot, output_dir)
    save_rate_curve(rate_curve, output_dir / "usd_treasury_curve.png")
    return_normality = save_return_diagnostics(history, output_dir)

    bs_volatility, bs_diagnostics = black_scholes_fit(surface, spot)
    heston_params, heston_report = calibrate_heston(surface, spot)
    heston_diagnostics = option_diagnostics(
        surface, spot, heston_prices(surface, spot, heston_params, terms=256)
    )

    bates_report = Bates.calibrate_bates(
        surface, spot, maxiter=25, popsize=6, seed=SEED,
        pricing="cos", cos_N=128, disp=False, return_report=True
    )
    bates_params = bates_report.x
    bates_diagnostics = option_diagnostics(
        surface,
        spot,
        price_surface(
            surface,
            lambda strikes, maturity, rate: Bates.bates_prices_cos(
                spot, strikes, maturity, *bates_params, rate, N=256
            ),
        ),
    )

    metrics = {
        "Black-Scholes": error_metrics(bs_diagnostics),
        "Heston": error_metrics(heston_diagnostics),
        "Bates": error_metrics(bates_diagnostics),
    }
    pd.DataFrame([{"model": key, **value} for key, value in metrics.items()]).to_csv(
        output_dir / "baseline_calibration_metrics.csv", index=False
    )
    (output_dir / "baseline_calibration_metrics.json").write_text(
        json.dumps(metrics, indent=2), encoding="utf-8"
    )
    (output_dir / "heston_calibrated_params.json").write_text(
        json.dumps(heston_report, indent=2), encoding="utf-8"
    )
    (output_dir / "black_scholes_calibrated_params.json").write_text(
        json.dumps({
            "model": "Black-Scholes / GBM",
            "parameters": {"sigma": bs_volatility},
            "objective": float(np.mean(
                ((bs_diagnostics["model_price"] - bs_diagnostics["price"])
                 / bs_diagnostics["vega"]) ** 2
            )),
            "rate_input": "maturity-specific LSE USD Treasury curve",
            "success": True,
        }, indent=2), encoding="utf-8"
    )
    bates_payload = bates_report.as_dict()
    bates_payload["parameters"]["feller_gap"] = (
        2.0 * bates_params[1] * bates_params[2] - bates_params[3] ** 2
    )
    bates_payload["seed"] = SEED
    (output_dir / "bates_calibrated_params.json").write_text(
        json.dumps(bates_payload, indent=2), encoding="utf-8"
    )
    summary = (
        surface.groupby("expiry", as_index=False)
        .agg(n_options=("K", "size"), mean_implied_vol=("implied_vol", "mean"),
             median_implied_vol=("implied_vol", "median"), mean_vega=("vega", "mean"))
    )
    summary.to_csv(output_dir / "black_scholes_calibration_summary.csv", index=False)

    plot_residuals(bs_diagnostics, "Black--Scholes", output_dir / "black_scholes_residual_heatmap.png")
    plot_smiles(bs_diagnostics, "Black--Scholes", output_dir / "black_scholes_volatility_smile.png")
    plot_residuals(heston_diagnostics, "Heston", output_dir / "heston_residual_heatmap.png")
    plot_smiles(heston_diagnostics, "Heston", output_dir / "heston_volatility_smile.png")
    plot_residuals(bates_diagnostics, "Bates", output_dir / "bates_residual_heatmap.png")
    plot_smiles(bates_diagnostics, "Bates", output_dir / "bates_volatility_smile.png")
    residual_comparison(
        {"Black--Scholes": bs_diagnostics, "Heston": heston_diagnostics, "Bates": bates_diagnostics},
        output_dir / "error_heatmaps_comparison.png",
    )
    residual_normality = []
    for model, diagnostics in (
        ("Black-Scholes", bs_diagnostics),
        ("Heston", heston_diagnostics),
        ("Bates", bates_diagnostics),
    ):
        row = normality_statistics(diagnostics["iv_residual_pct"])
        row.update({
            "model": model,
            "residual": "market minus model implied volatility (pp)",
            "decision_5pct_shapiro": "reject" if row["shapiro_pvalue"] < 0.05 else "do not reject",
            "decision_5pct_jarque_bera": "reject" if row["jarque_bera_pvalue"] < 0.05 else "do not reject",
            "decision_5pct_dagostino": "reject" if row["dagostino_k2_pvalue"] < 0.05 else "do not reject",
        })
        residual_normality.append(row)
    pd.DataFrame(residual_normality).to_csv(
        output_dir / "calibration_residual_normality.csv", index=False
    )

    manifest = {
        "source": metadata["source"],
        "as_of_utc": metadata["as_of_utc"],
        "spot": spot,
        "risk_free_rate_source": metadata["risk_free_rate_source"],
        "risk_free_rate_curve_date": metadata["risk_free_rate_curve_date"],
        "risk_free_rate_conversion": metadata["risk_free_rate_conversion"],
        "risk_free_rate_interpolation": metadata["risk_free_rate_interpolation"],
        "risk_free_rate_model": metadata["risk_free_rate_model"],
        "risk_free_rate_nss_parameters": metadata["risk_free_rate_nss_parameters"],
        "risk_free_rate_nss_rmse_bp": metadata["risk_free_rate_nss_rmse_bp"],
        "risk_free_rate_missing_requested_tenors": metadata[
            "risk_free_rate_missing_requested_tenors"
        ],
        "risk_free_rate_range": [
            metadata["risk_free_rate_min"], metadata["risk_free_rate_max"]
        ],
        "dividend_yield_assumption": metadata["dividend_yield_assumption"],
        "calibration_price_method": metadata["calibration_price_method"],
        "maximum_source_age_days": metadata["maximum_source_age_days"],
        "chain_rows": metadata["n_rows_total"],
        "eligible_rows": metadata["n_rows_calibration_eligible"],
        "sample_rows": int(len(surface)),
        "maturities": int(surface["T"].nunique()),
        "sample_sha256": hashlib.sha256(sample_path.read_bytes()).hexdigest(),
        "history_start": metadata["history_start"],
        "history_end": metadata["history_end"],
        "history_observations": metadata["history_observations"],
        "raw_data_committed": False,
        "black_scholes_constant_volatility": bs_volatility,
    }
    (output_dir / "lse_publication_manifest.json").write_text(
        json.dumps(manifest, indent=2), encoding="utf-8"
    )
    print(json.dumps({"manifest": manifest, "metrics": metrics,
                      "heston": heston_report, "bates": bates_payload,
                      "return_normality": return_normality}, indent=2))


if __name__ == "__main__":
    main()
