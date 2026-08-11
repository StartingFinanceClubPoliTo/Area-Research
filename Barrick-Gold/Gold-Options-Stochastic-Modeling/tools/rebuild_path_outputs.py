"""Regenerate publication path figures from constrained model parameters."""

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

from calibration_workflow import load_calibration_surface  # noqa: E402
from path_simulation import (  # noqa: E402
    simulate_bates_paths,
    simulate_full_hawkes_paths,
    simulate_gbm_paths,
    simulate_heston_paths,
    terminal_statistics,
)


SEED = 20260811
N_PATHS = 4096
N_STEPS = 260
HORIZON = 5.0
Heston_KAPPA, HESTON_THETA = 4.7261, 0.0847
HESTON_XI = float(np.sqrt(2.0 * Heston_KAPPA * HESTON_THETA - 1.48e-4))
HESTON_PARAMS = (0.1293, Heston_KAPPA, HESTON_THETA, HESTON_XI, -0.1872)
BATES_PARAMS = (0.0791, 1.7164, 0.0428, 0.3833, 0.2206, 0.8963, -0.1901, 0.1006)


def quantile_indices(paths):
    order = np.argsort(paths[:, -1])
    return [order[int(level * (len(order) - 1))] for level in (0.05, 0.25, 0.5, 0.75, 0.95)]


def save_five_paths(times, paths, title, subtitle, output_path):
    figure, axis = plt.subplots(figsize=(6.2, 3.6))
    for number, index in enumerate(quantile_indices(paths), start=1):
        axis.plot(times, paths[index], linewidth=1.25, label=f"path {number}")
    axis.set(title=title, xlabel="Years", ylabel="GLD price")
    axis.text(0.01, 0.02, subtitle, transform=axis.transAxes, fontsize=8)
    axis.legend(ncol=3, loc="upper left")
    axis.grid(True, alpha=0.22)
    figure.tight_layout()
    figure.savefig(output_path, dpi=220, bbox_inches="tight")
    plt.close(figure)


def save_fan_chart(times, simulations, output_path):
    figure, axis = plt.subplots(figsize=(10.8, 4.8))
    colors = ("#1f77b4", "#2ca02c", "#d62728", "#6f42c1")
    for (label, paths), color in zip(simulations.items(), colors):
        mean = np.mean(paths, axis=0)
        lower, upper = np.percentile(paths, [5, 95], axis=0)
        axis.plot(times, mean, color=color, linewidth=1.8, label=f"{label} mean")
        axis.fill_between(times, lower, upper, color=color, alpha=0.10)
    axis.set(
        title="Simulated gold-price paths: mean and 5--95 percent band",
        xlabel="Years",
        ylabel="GLD price",
    )
    axis.legend(loc="upper left")
    axis.grid(True, alpha=0.22)
    figure.tight_layout()
    figure.savefig(output_path, dpi=220, bbox_inches="tight")
    plt.close(figure)


def save_state_figures(times, heston_var, bates_var, full_var, bates_counts,
                       full_counts, full_intensity, full_params, output_dir):
    selected = range(4)
    figure, axes = plt.subplots(1, 3, figsize=(13.2, 3.8), sharey=True)
    for index in selected:
        axes[0].plot(times, np.sqrt(heston_var[index]), alpha=0.7)
        axes[1].plot(times, np.sqrt(bates_var[index]), alpha=0.7)
        axes[2].plot(times, np.sqrt(full_var[index]), alpha=0.7)
    for axis, title in zip(axes, ("Heston", "Bates", "Full Bates--Hawkes")):
        axis.set(title=title, xlabel="Years")
        axis.grid(True, alpha=0.22)
    axes[0].set_ylabel("Annualized volatility")
    figure.tight_layout()
    figure.savefig(output_dir / "volatility_state_paths.png", dpi=220, bbox_inches="tight")
    plt.close(figure)

    figure, (intensity_axis, count_axis) = plt.subplots(2, 1, figsize=(9.4, 6.5), sharex=True)
    for index in selected:
        intensity_axis.plot(times, full_intensity[index], alpha=0.72)
        count_axis.step(times, full_counts[index], where="post", alpha=0.72)
    intensity_axis.axhline(
        full_params["lambda_bar"], color="black", linestyle="--", label="baseline"
    )
    intensity_axis.set(ylabel="Jump intensity", title="Full Bates--Hawkes jump states")
    intensity_axis.legend()
    count_axis.set(xlabel="Years", ylabel="Cumulative jumps")
    for axis in (intensity_axis, count_axis):
        axis.grid(True, alpha=0.22)
    figure.tight_layout()
    figure.savefig(output_dir / "hawkes_jump_paths.png", dpi=220, bbox_inches="tight")
    plt.close(figure)

    figure, (intensity_axis, count_axis) = plt.subplots(2, 1, figsize=(9.4, 6.5), sharex=True)
    intensity_axis.plot(times, np.repeat(BATES_PARAMS[5], len(times)), linewidth=2.0)
    for index in selected:
        count_axis.step(times, bates_counts[index], where="post", alpha=0.72)
    intensity_axis.set(ylabel="Jump intensity", title="Bates Poisson jump states")
    count_axis.set(xlabel="Years", ylabel="Cumulative jumps")
    for axis in (intensity_axis, count_axis):
        axis.grid(True, alpha=0.22)
    figure.tight_layout()
    figure.savefig(output_dir / "bates_poisson_jump_paths.png", dpi=220, bbox_inches="tight")
    plt.close(figure)


def main():
    data_dir = ROOT / "Data"
    surface, spot = load_calibration_surface(data_dir)
    full_params = json.loads(
        (data_dir / "bates_hawkes_calibrated_params.json").read_text(encoding="utf-8")
    )["parameters"]
    dt = HORIZON / N_STEPS
    times = np.linspace(0.0, HORIZON, N_STEPS + 1)
    rates = np.repeat(float(surface["rate"].median()), N_STEPS)
    bs_volatility = float(surface["implied_vol"].median())

    gbm = simulate_gbm_paths(spot, rates, bs_volatility, dt, N_PATHS, SEED)
    heston, heston_var = simulate_heston_paths(
        spot, rates, HESTON_PARAMS, dt, N_PATHS, SEED + 10
    )
    bates, bates_var, bates_counts = simulate_bates_paths(
        spot, rates, BATES_PARAMS, dt, N_PATHS, SEED + 20
    )
    full, full_var, full_counts, full_intensity = simulate_full_hawkes_paths(
        spot, rates, full_params, dt, N_PATHS, SEED + 30
    )

    simulations = {
        "GBM / BS": gbm,
        "Heston": heston,
        "Bates": bates,
        "Full Bates--Hawkes": full,
    }
    save_five_paths(times, gbm, "Black--Scholes / GBM", "constant volatility", data_dir / "paths_black_scholes_5.png")
    save_five_paths(times, heston, "Heston", "stochastic variance", data_dir / "paths_heston_5.png")
    save_five_paths(times, bates, "Bates", "Heston variance and Poisson jumps", data_dir / "paths_bates_5.png")
    save_five_paths(times, full, "Full Bates--Hawkes", "Heston variance and self-exciting jumps", data_dir / "paths_bates_hawkes_5.png")
    save_fan_chart(times, simulations, data_dir / "gold_path_stats_by_model.png")
    save_state_figures(
        times, heston_var, bates_var, full_var, bates_counts,
        full_counts, full_intensity, full_params, data_dir
    )

    rows = [
        terminal_statistics("GBM / BS", spot, gbm),
        terminal_statistics("Heston", spot, heston),
        terminal_statistics("Bates", spot, bates, bates_counts),
        terminal_statistics(
            "Full Bates-Hawkes", spot, full, full_counts,
            "Feller-constrained affine calibration"
        ),
    ]
    result = pd.DataFrame(rows)
    result.to_csv(data_dir / "path_simulation_summary.csv", index=False)
    print(result.to_string(index=False))


if __name__ == "__main__":
    main()
