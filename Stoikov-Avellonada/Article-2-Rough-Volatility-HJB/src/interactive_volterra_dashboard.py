"""Interactive Volterra dashboard with sliders for causal-memory diagnostics."""

from __future__ import annotations

import matplotlib.pyplot as plt
from matplotlib.widgets import Slider
import numpy as np

from rough_processes import fbm_covariance, simulate_volterra


def main() -> None:
    fig, axes = plt.subplots(3, 2, figsize=(12, 12))
    plt.subplots_adjust(left=0.08, bottom=0.18, right=0.98, top=0.93, hspace=0.55, wspace=0.28)

    ax_steps = fig.add_axes([0.08, 0.10, 0.20, 0.03])
    ax_paths = fig.add_axes([0.33, 0.10, 0.20, 0.03])
    ax_hurst = fig.add_axes([0.58, 0.10, 0.16, 0.03])
    ax_window = fig.add_axes([0.80, 0.10, 0.16, 0.03])
    steps_slider = Slider(ax_steps, "Time steps", 80, 420, valinit=300, valstep=10)
    paths_slider = Slider(ax_paths, "Sample paths", 3, 30, valinit=10, valstep=1)
    hurst_slider = Slider(ax_hurst, "Hurst H", 0.10, 0.90, valinit=0.50, valstep=0.05)
    window_slider = Slider(ax_window, "Filter s", 0.10, 1.00, valinit=0.50, valstep=0.05)

    def draw(_=None) -> None:
        steps = int(steps_slider.val)
        paths = int(paths_slider.val)
        hurst = float(hurst_slider.val)
        window = float(window_slider.val)
        X, W, t, K = simulate_volterra(steps, hurst, paths, seed=456)
        stop = max(2, int(window * (steps - 1)))
        cov_target = fbm_covariance(t, hurst)
        cov_emp = np.cov(X, rowvar=False)

        for ax in axes.ravel():
            ax.clear()

        for row in X[: min(paths, 15)]:
            axes[0, 0].plot(t, row, lw=1.0)
        axes[0, 0].axvline(t[stop], color="red", ls="--", lw=1.0)
        axes[0, 0].set_title("Sample Paths with Filtration Window")
        axes[0, 0].set_xlabel("t")
        axes[0, 0].set_ylabel("X(t)")

        im = axes[0, 1].imshow(K, origin="lower", aspect="auto", cmap="viridis")
        axes[0, 1].set_title("Volterra Kernel K(t,s)")
        fig.colorbar(im, ax=axes[0, 1], fraction=0.046, pad=0.04)

        for row in W[: min(paths, 3)]:
            axes[1, 0].plot(t, row, lw=1.0)
        axes[1, 0].set_title("Driving Brownian Paths")
        axes[1, 0].set_xlabel("t")
        axes[1, 0].set_ylabel("W(t)")

        dW = np.diff(W[: min(paths, 3)], axis=1)
        for row in dW:
            axes[1, 1].plot(t[1:], row, lw=0.8, marker=".", ms=2)
        axes[1, 1].set_title("Brownian Increments")
        axes[1, 1].set_xlabel("t")
        axes[1, 1].set_ylabel("dW(t)")

        for frac in (0.25, 0.50, 0.75):
            idx = max(2, int(frac * (steps - 1)))
            axes[2, 0].plot(t[:idx], K[idx, :idx], lw=2.0, label=f"t={t[idx]:.2f}")
        axes[2, 0].set_title("Causal Kernel Slices")
        axes[2, 0].set_xlabel("s")
        axes[2, 0].set_ylabel("K(t,s)")
        axes[2, 0].legend(fontsize=8)

        rmse = float(np.sqrt(np.mean((cov_emp - cov_target) ** 2)))
        im2 = axes[2, 1].imshow(cov_emp, origin="lower", aspect="auto", cmap="viridis")
        axes[2, 1].set_title(f"Empirical Covariance, RMSE={rmse:.2e}")
        fig.colorbar(im2, ax=axes[2, 1], fraction=0.046, pad=0.04)
        fig.canvas.draw_idle()

    for slider in (steps_slider, paths_slider, hurst_slider, window_slider):
        slider.on_changed(draw)
    draw()
    plt.show()


if __name__ == "__main__":
    main()
