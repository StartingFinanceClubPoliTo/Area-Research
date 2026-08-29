"""Generate synthetic figures for the Beyond Black-Scholes theory article."""

from __future__ import annotations

import csv
from pathlib import Path

import matplotlib.pyplot as plt
import numpy as np


ROOT = Path(__file__).resolve().parent
IMG_DIR = ROOT / "img"
OUTPUT_DIR = ROOT / "output"
SEED = 18062026


def write_csv(path: Path, columns: list[str], rows: np.ndarray) -> None:
    with path.open("w", newline="", encoding="utf-8") as handle:
        writer = csv.writer(handle)
        writer.writerow(columns)
        writer.writerows(rows.tolist())


def simulate_jump_paths(rng: np.random.Generator) -> tuple[np.ndarray, np.ndarray, np.ndarray]:
    steps = 252
    horizon = 1.0
    dt = horizon / steps
    time = np.linspace(0.0, horizon, steps + 1)

    spot = 100.0
    drift = 0.03
    volatility = 0.18
    jump_intensity = 2.2
    jump_mean = -0.09
    jump_std = 0.05

    gaussian_shocks = rng.normal(size=steps)
    diffusion_log_returns = (
        (drift - 0.5 * volatility**2) * dt
        + volatility * np.sqrt(dt) * gaussian_shocks
    )

    jump_counts = rng.poisson(jump_intensity * dt, size=steps)
    jump_sizes = rng.normal(jump_mean, jump_std, size=steps) * jump_counts
    jump_log_returns = diffusion_log_returns + jump_sizes

    diffusion_path = spot * np.exp(np.r_[0.0, np.cumsum(diffusion_log_returns)])
    jump_path = spot * np.exp(np.r_[0.0, np.cumsum(jump_log_returns)])
    return time, diffusion_path, jump_path


def simulate_variance_paths(rng: np.random.Generator) -> tuple[np.ndarray, np.ndarray]:
    steps = 252
    horizon = 1.0
    dt = horizon / steps
    time = np.linspace(0.0, horizon, steps + 1)

    kappa = 3.0
    theta = 0.045
    vol_of_vol = 0.32
    paths = np.zeros((steps + 1, 3))
    paths[0, :] = np.array([0.03, 0.045, 0.07])

    shocks = rng.normal(size=(steps, paths.shape[1]))
    for step in range(steps):
        previous = np.maximum(paths[step, :], 0.0)
        next_value = (
            previous
            + kappa * (theta - previous) * dt
            + vol_of_vol * np.sqrt(previous) * np.sqrt(dt) * shocks[step, :]
        )
        paths[step + 1, :] = np.maximum(next_value, 0.0)

    return time, paths


def plot_jump_paths(time: np.ndarray, diffusion: np.ndarray, jump: np.ndarray) -> None:
    fig, axis = plt.subplots(figsize=(9.0, 4.8))
    axis.plot(time, diffusion, color="#255f85", linewidth=2.2, label="Diffusion benchmark")
    axis.plot(time, jump, color="#b73f32", linewidth=2.2, label="Jump-diffusion example")
    axis.set_title("Synthetic price paths: diffusion versus jump-diffusion")
    axis.set_xlabel("Years")
    axis.set_ylabel("Price index")
    axis.grid(alpha=0.25)
    axis.legend(frameon=False)
    fig.tight_layout()
    fig.savefig(IMG_DIR / "jump_diffusion_paths.png", dpi=220)
    plt.close(fig)


def plot_variance_paths(time: np.ndarray, paths: np.ndarray) -> None:
    fig, axis = plt.subplots(figsize=(9.0, 4.8))
    palette = ["#255f85", "#6c8f3d", "#b73f32"]
    for index, color in enumerate(palette, start=1):
        axis.plot(time, paths[:, index - 1], linewidth=2.0, color=color, label=f"Path {index}")
    axis.axhline(0.045, color="#222222", linestyle="--", linewidth=1.2, label="Long-run mean")
    axis.set_title("Synthetic stochastic-variance examples")
    axis.set_xlabel("Years")
    axis.set_ylabel("Variance")
    axis.grid(alpha=0.25)
    axis.legend(frameon=False, ncol=2)
    fig.tight_layout()
    fig.savefig(IMG_DIR / "stochastic_variance_examples.png", dpi=220)
    plt.close(fig)


def main() -> None:
    IMG_DIR.mkdir(parents=True, exist_ok=True)
    OUTPUT_DIR.mkdir(parents=True, exist_ok=True)
    rng = np.random.default_rng(SEED)

    jump_time, diffusion, jump = simulate_jump_paths(rng)
    variance_time, variance_paths = simulate_variance_paths(rng)

    plot_jump_paths(jump_time, diffusion, jump)
    plot_variance_paths(variance_time, variance_paths)

    write_csv(
        OUTPUT_DIR / "jump_diffusion_paths.csv",
        ["time_years", "diffusion_benchmark", "jump_diffusion_example"],
        np.column_stack([jump_time, diffusion, jump]),
    )
    write_csv(
        OUTPUT_DIR / "stochastic_variance_examples.csv",
        ["time_years", "path_1", "path_2", "path_3"],
        np.column_stack([variance_time, variance_paths]),
    )


if __name__ == "__main__":
    main()
