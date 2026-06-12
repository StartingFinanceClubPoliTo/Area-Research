from pathlib import Path
import sys

import matplotlib.pyplot as plt
import numpy as np
import pandas as pd

ROOT = Path(__file__).resolve().parents[1]
DATA_DIR = ROOT / "Data"

if str(ROOT) not in sys.path:
    sys.path.insert(0, str(ROOT))

from Hawkes import Hawkes  # noqa: E402


def build_diagnostics():
    horizon = 10.0
    lambda0 = 0.55
    alpha = 0.95
    beta = 1.55
    poisson_lambda = Hawkes.stationary_mean_intensity(lambda0, alpha, beta)

    grid = np.linspace(0.0, horizon, 600)
    hawkes_events = Hawkes.simulate_exponential(lambda0, alpha, beta, horizon, seed=20260612)
    poisson_events = Hawkes.simulate_poisson(poisson_lambda, horizon, seed=20260612)
    hawkes_intensity = Hawkes.intensity_on_grid(grid, hawkes_events, lambda0, alpha, beta)
    poisson_intensity = np.full_like(grid, poisson_lambda)

    rows = pd.DataFrame(
        {
            "time": grid,
            "poisson_intensity": poisson_intensity,
            "hawkes_intensity": hawkes_intensity,
        }
    )

    DATA_DIR.mkdir(parents=True, exist_ok=True)
    rows.to_csv(DATA_DIR / "hawkes_intensity_comparison.csv", index=False)

    fig, axes = plt.subplots(2, 1, figsize=(10.5, 7.0), sharex=True)

    axes[0].plot(grid, poisson_intensity, color="#365f91", linewidth=2.2, label="Constant Bates intensity")
    axes[0].plot(grid, hawkes_intensity, color="#b73f32", linewidth=2.2, label="Self-exciting Hawkes intensity")
    axes[0].axhline(lambda0, color="#2f2f2f", linestyle="--", linewidth=1.4, label="Hawkes baseline")
    for event_time in hawkes_events:
        axes[0].axvline(event_time, color="#b73f32", alpha=0.18, linewidth=0.8)
    axes[0].set_ylabel("Jump intensity")
    axes[0].set_title("Constant-intensity Bates versus self-exciting Bates-Hawkes jumps")
    axes[0].legend(loc="upper right")
    axes[0].grid(alpha=0.25)

    axes[1].step(
        poisson_events,
        np.arange(1, len(poisson_events) + 1),
        where="post",
        color="#365f91",
        linewidth=2.0,
        label="Poisson jump count",
    )
    axes[1].step(
        hawkes_events,
        np.arange(1, len(hawkes_events) + 1),
        where="post",
        color="#b73f32",
        linewidth=2.0,
        label="Hawkes jump count",
    )
    axes[1].set_xlabel("Time")
    axes[1].set_ylabel("Cumulative jumps")
    axes[1].legend(loc="upper left")
    axes[1].grid(alpha=0.25)

    fig.tight_layout()
    output_path = DATA_DIR / "hawkes_intensity_comparison.png"
    fig.savefig(output_path, dpi=220)
    plt.close(fig)
    print(f"[INFO] Wrote {output_path.relative_to(Path.cwd())}")
    print(f"[INFO] Hawkes events: {len(hawkes_events)}, Poisson events: {len(poisson_events)}")


if __name__ == "__main__":
    build_diagnostics()
