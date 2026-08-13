"""Thin-notebook workflow helpers for Bates-family calibration diagnostics."""

import json
from pathlib import Path

import matplotlib.pyplot as plt
import numpy as np
import pandas as pd

from BnS import BnS


def load_calibration_surface(data_dir="Data/lse_local", dividend_yield=0.0):
    """Load the local-only LSE sample and add protected Black--Scholes vega."""
    data_dir = Path(data_dir)
    metadata = json.loads(
        (data_dir / "gld_lse_meta.json").read_text(encoding="utf-8")
    )
    spot = next(
        float(metadata[name])
        for name in ("S0", "spot", "spot_price", "underlying_price", "underlying_last")
        if name in metadata
    )
    surface = pd.read_csv(data_dir / "gld_lse_calibration_chebyshev.csv")
    surface["vega"] = [
        BnS.calculate_bs_vega(
            spot,
            row.K,
            row.T,
            row.rate,
            dividend_yield,
            row.implied_vol,
        )
        for row in surface.itertuples(index=False)
    ]
    return surface, spot


def price_surface(surface, batch_pricer):
    """Price a surface once per ``(T, rate)`` group while preserving row order."""
    prices = np.empty(len(surface), dtype=float)
    for (_, _), group in surface.groupby(["T", "rate"], sort=False):
        positions = surface.index.get_indexer(group.index)
        prices[positions] = batch_pricer(
            group["K"].to_numpy(dtype=float),
            float(group["T"].iloc[0]),
            float(group["rate"].iloc[0]),
        )
    return prices


def option_diagnostics(surface, spot, model_prices, dividend_yield=0.0):
    """Convert model prices to IV and return option-level residual diagnostics."""
    diagnostics = surface.copy().reset_index(drop=True)
    diagnostics["model_price"] = np.asarray(model_prices, dtype=float)
    diagnostics["model_iv"] = [
        BnS.implied_vol_call(
            row.model_price,
            spot,
            row.K,
            row.T,
            row.rate,
            dividend_yield,
        )
        for row in diagnostics.itertuples(index=False)
    ]
    diagnostics["moneyness"] = diagnostics["K"] / spot
    diagnostics["iv_residual_pct"] = (
        diagnostics["implied_vol"] - diagnostics["model_iv"]
    ) * 100.0
    return diagnostics


def plot_smiles(diagnostics, model_label, output_path):
    """Plot market/model smile points for every maturity in a compact grid."""
    maturities = sorted(diagnostics["T"].unique())
    columns = min(3, len(maturities))
    rows = int(np.ceil(len(maturities) / columns))
    figure, axes = plt.subplots(rows, columns, figsize=(6 * columns, 4.5 * rows))
    axes = np.atleast_1d(axes).ravel()
    for axis, maturity in zip(axes, maturities):
        group = diagnostics.loc[diagnostics["T"] == maturity].sort_values("K")
        axis.plot(group["K"], group["implied_vol"] * 100.0, "ko", label="Market")
        axis.plot(group["K"], group["model_iv"] * 100.0, "r-", label=model_label)
        expiry = group["expiry"].iloc[0] if "expiry" in group else f"T={maturity:.3f}"
        axis.set(title=f"Expiry: {expiry}", xlabel="Strike", ylabel="IV (%)")
        axis.grid(True, linestyle="--", alpha=0.5)
        axis.legend()
    for axis in axes[len(maturities):]:
        axis.remove()
    figure.tight_layout()
    figure.savefig(output_path, dpi=200, bbox_inches="tight")
    return figure


def plot_residuals(diagnostics, model_label, output_path):
    """Plot IV residuals across maturity and moneyness."""
    figure, axis = plt.subplots(figsize=(11, 6.5))
    points = axis.scatter(
        diagnostics["T"],
        diagnostics["moneyness"],
        c=diagnostics["iv_residual_pct"],
        cmap="coolwarm",
        s=60,
        edgecolors="black",
        linewidth=0.4,
    )
    figure.colorbar(points, ax=axis, label="Market - model IV (pp)")
    axis.axhline(1.0, color="black", linestyle="--", linewidth=1.2)
    axis.set(
        title=f"{model_label} implied-volatility residuals",
        xlabel="Time to maturity (years)",
        ylabel="Moneyness (K / S0)",
    )
    axis.grid(True, linestyle=":", alpha=0.5)
    figure.tight_layout()
    figure.savefig(output_path, dpi=200, bbox_inches="tight")
    return figure
