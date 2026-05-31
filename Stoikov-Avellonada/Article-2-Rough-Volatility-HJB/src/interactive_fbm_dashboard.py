"""Interactive fBM dashboard with sliders for steps, paths, and Hurst parameter."""

from __future__ import annotations

import matplotlib.pyplot as plt
from matplotlib.widgets import Slider
import numpy as np

from rough_processes import covariance_error, fbm_covariance, simulate_fbm_cholesky


def main() -> None:
    fig, axes = plt.subplots(2, 2, figsize=(12, 8))
    plt.subplots_adjust(left=0.08, bottom=0.20, right=0.98, top=0.92, hspace=0.35, wspace=0.25)

    ax_steps = fig.add_axes([0.10, 0.11, 0.25, 0.03])
    ax_paths = fig.add_axes([0.42, 0.11, 0.25, 0.03])
    ax_hurst = fig.add_axes([0.74, 0.11, 0.20, 0.03])
    steps_slider = Slider(ax_steps, "Time steps", 60, 450, valinit=300, valstep=10)
    paths_slider = Slider(ax_paths, "Sample paths", 3, 40, valinit=10, valstep=1)
    hurst_slider = Slider(ax_hurst, "Hurst H", 0.05, 0.95, valinit=0.50, valstep=0.05)

    def draw(_=None) -> None:
        steps = int(steps_slider.val)
        paths = int(paths_slider.val)
        hurst = float(hurst_slider.val)
        X, t, cov = simulate_fbm_cholesky(steps, hurst, paths, seed=123)
        emp = np.cov(X, rowvar=False)
        diff, rmse = covariance_error(X, cov)
        lag_error = np.sqrt(np.mean(diff ** 2, axis=0))

        for ax in axes.ravel():
            ax.clear()

        for row in X[: min(paths, 20)]:
            axes[0, 0].plot(t, row, lw=1.0)
        axes[0, 0].set_title(f"Sample Paths, H={hurst:.2f}")
        axes[0, 0].set_xlabel("t")
        axes[0, 0].set_ylabel("B_H(t)")

        im1 = axes[0, 1].imshow(emp, origin="lower", aspect="auto", cmap="viridis")
        axes[0, 1].set_title(f"Empirical Covariance, RMSE={rmse:.2e}")
        fig.colorbar(im1, ax=axes[0, 1], fraction=0.046, pad=0.04)

        axes[1, 0].plot(lag_error, color="#e74c3c", lw=2.0)
        axes[1, 0].set_title("Covariance Error by Lag")
        axes[1, 0].set_xlabel("Lag")
        axes[1, 0].set_ylabel("RMSE")
        axes[1, 0].grid(alpha=0.25)

        im2 = axes[1, 1].imshow(cov, origin="lower", aspect="auto", cmap="viridis")
        axes[1, 1].set_title("Theoretical Covariance")
        fig.colorbar(im2, ax=axes[1, 1], fraction=0.046, pad=0.04)
        fig.canvas.draw_idle()

    for slider in (steps_slider, paths_slider, hurst_slider):
        slider.on_changed(draw)
    draw()
    plt.show()


if __name__ == "__main__":
    main()
