"""Interactive Volterra dashboard with sliders for causal-memory diagnostics."""

from __future__ import annotations

import numpy as np

from dashboard_base import InteractiveDashboard, SliderSpec
from rough_processes import SimulationGrid, VolterraProcess, fbm_covariance


class VolterraDashboard(InteractiveDashboard):
    def __init__(self) -> None:
        super().__init__(
            3,
            2,
            (12, 12),
            dict(left=0.08, bottom=0.18, right=0.98, top=0.93, hspace=0.55, wspace=0.28),
            (
                SliderSpec("steps", "Time steps", (0.08, 0.10, 0.20, 0.03), 80, 420, 300, 10),
                SliderSpec("paths", "Sample paths", (0.33, 0.10, 0.20, 0.03), 3, 30, 10, 1),
                SliderSpec("hurst", "Hurst H", (0.58, 0.10, 0.16, 0.03), 0.10, 0.90, 0.50, 0.05),
                SliderSpec("window", "Filter s", (0.80, 0.10, 0.16, 0.03), 0.10, 1.00, 0.50, 0.05),
            ),
        )

    def render(self) -> None:
        steps = int(self.value("steps"))
        paths = int(self.value("paths"))
        hurst = self.value("hurst")
        model = VolterraProcess(SimulationGrid(steps), hurst)
        samples, brownian = model.simulate(paths, seed=456)
        stop = max(2, int(self.value("window") * (steps - 1)))
        target = fbm_covariance(model.grid.time, hurst)
        induced = model.induced_covariance

        for row in samples[: min(paths, 15)]:
            self.axes[0, 0].plot(model.grid.time, row, lw=1.0)
        self.axes[0, 0].axvline(model.grid.time[stop], color="red", ls="--", lw=1.0)
        self.axes[0, 0].set(
            title="Sample Paths with Filtration Window", xlabel="t", ylabel="X(t)"
        )

        image = self.axes[0, 1].imshow(
            model.kernel, origin="lower", aspect="auto", cmap="viridis"
        )
        self.axes[0, 1].set_title("Volterra Kernel K(t,s)")
        self.add_colorbar(image, self.axes[0, 1])

        for row in brownian[: min(paths, 3)]:
            self.axes[1, 0].plot(model.grid.time, row, lw=1.0)
        self.axes[1, 0].set(title="Driving Brownian Paths", xlabel="t", ylabel="W(t)")

        for row in np.diff(brownian[: min(paths, 3)], axis=1):
            self.axes[1, 1].plot(model.grid.time[1:], row, lw=0.8, marker=".", ms=2)
        self.axes[1, 1].set(title="Brownian Increments", xlabel="t", ylabel="dW(t)")

        for fraction in (0.25, 0.50, 0.75):
            index = max(2, int(fraction * (steps - 1)))
            (line,) = self.axes[2, 0].plot(
                model.grid.time,
                model.kernel[index],
                lw=2.0,
                drawstyle="steps-post",
                label=f"t={model.grid.time[index]:.2f}",
            )
            self.axes[2, 0].axvline(
                model.grid.time[index], color=line.get_color(), ls="--", lw=0.9, alpha=0.5
            )
        self.axes[2, 0].set(
            title="Causal Kernel Support",
            xlabel="s",
            ylabel="K(t,s)",
            xlim=(model.grid.time[0], model.grid.time[-1]),
        )
        self.axes[2, 0].legend(fontsize=8)

        rmse = float(np.sqrt(np.mean((induced - target) ** 2)))
        covariance_limit = max(float(induced.max()), float(target.max()), 1e-12)
        image = self.axes[2, 1].imshow(
            induced,
            origin="lower",
            aspect="auto",
            cmap="viridis",
            vmin=0.0,
            vmax=covariance_limit,
        )
        self.axes[2, 1].set_title(f"Kernel-Induced Covariance, fBM RMSE={rmse:.2e}")
        self.add_colorbar(image, self.axes[2, 1])


def main() -> None:
    VolterraDashboard().run()


if __name__ == "__main__":
    main()
