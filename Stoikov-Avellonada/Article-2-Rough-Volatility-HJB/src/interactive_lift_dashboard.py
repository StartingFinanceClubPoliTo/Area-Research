"""Interactive Markovian-lifting dashboard with sliders for H and OU factors."""

from __future__ import annotations

import matplotlib.pyplot as plt
from matplotlib.widgets import Slider
import numpy as np

from rough_processes import covariance_error, fbm_covariance, simulate_markov_lift


def main() -> None:
    fig, axes = plt.subplots(2, 2, figsize=(12, 8))
    plt.subplots_adjust(left=0.08, bottom=0.18, right=0.98, top=0.92, hspace=0.35, wspace=0.25)

    ax_steps = fig.add_axes([0.08, 0.10, 0.20, 0.03])
    ax_paths = fig.add_axes([0.33, 0.10, 0.20, 0.03])
    ax_hurst = fig.add_axes([0.58, 0.10, 0.16, 0.03])
    ax_factors = fig.add_axes([0.80, 0.10, 0.16, 0.03])
    steps_slider = Slider(ax_steps, "Time steps", 80, 420, valinit=300, valstep=10)
    paths_slider = Slider(ax_paths, "Sample paths", 5, 80, valinit=25, valstep=5)
    hurst_slider = Slider(ax_hurst, "Hurst H", 0.10, 0.90, valinit=0.50, valstep=0.05)
    factors_slider = Slider(ax_factors, "OU factors", 3, 20, valinit=10, valstep=1)

    def draw(_=None) -> None:
        steps = int(steps_slider.val)
        paths = int(paths_slider.val)
        hurst = float(hurst_slider.val)
        factors = int(factors_slider.val)
        Y, t, lam, w = simulate_markov_lift(steps, hurst, paths, factors, seed=789)
        target = fbm_covariance(t, hurst)
        diff, rmse = covariance_error(Y, target)
        emp = np.cov(Y, rowvar=False)

        for ax in axes.ravel():
            ax.clear()

        for row in Y[: min(paths, 20)]:
            axes[0, 0].plot(t, row, lw=1.0)
        axes[0, 0].set_title(f"Lifted Paths, H={hurst:.2f}, M={factors}")
        axes[0, 0].set_xlabel("t")
        axes[0, 0].set_ylabel("Y(t)")

        axes[0, 1].plot(np.sqrt(np.mean(diff ** 2, axis=0)), color="#e74c3c", lw=2.0)
        axes[0, 1].set_title("Covariance Error by Lag")
        axes[0, 1].set_xlabel("Lag")
        axes[0, 1].set_ylabel("RMSE")
        axes[0, 1].grid(alpha=0.25)

        im1 = axes[1, 0].imshow(target, origin="lower", aspect="auto", cmap="viridis")
        axes[1, 0].set_title("Target fBM Covariance")
        fig.colorbar(im1, ax=axes[1, 0], fraction=0.046, pad=0.04)

        im2 = axes[1, 1].imshow(emp, origin="lower", aspect="auto", cmap="viridis")
        axes[1, 1].set_title(f"Lift Covariance, RMSE={rmse:.2e}")
        fig.colorbar(im2, ax=axes[1, 1], fraction=0.046, pad=0.04)
        fig.canvas.draw_idle()

    for slider in (steps_slider, paths_slider, hurst_slider, factors_slider):
        slider.on_changed(draw)
    draw()
    plt.show()


if __name__ == "__main__":
    main()
