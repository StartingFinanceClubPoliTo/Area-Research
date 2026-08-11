"""Interactive fBM dashboard with sliders for grid, paths, and Hurst value."""

from __future__ import annotations

import numpy as np

from dashboard_base import InteractiveDashboard, SliderSpec
from rough_processes import (
    FractionalBrownianMotion,
    SimulationGrid,
    covariance_error,
)


class FBMDashboard(InteractiveDashboard):
    def __init__(self) -> None:
        super().__init__(
            2,
            2,
            (12, 8),
            dict(left=0.08, bottom=0.20, right=0.98, top=0.92, hspace=0.35, wspace=0.25),
            (
                SliderSpec("steps", "Time steps", (0.10, 0.11, 0.25, 0.03), 60, 450, 300, 10),
                SliderSpec("paths", "Sample paths", (0.42, 0.11, 0.25, 0.03), 3, 40, 10, 1),
                SliderSpec("hurst", "Hurst H", (0.74, 0.11, 0.20, 0.03), 0.05, 0.95, 0.50, 0.05),
            ),
        )

    def render(self) -> None:
        steps = int(self.value("steps"))
        paths = int(self.value("paths"))
        hurst = self.value("hurst")
        model = FractionalBrownianMotion(SimulationGrid(steps), hurst)
        samples = model.simulate(paths, seed=123)
        difference, rmse = covariance_error(samples, model.covariance)
        empirical = np.cov(samples, rowvar=False)

        for row in samples[: min(paths, 20)]:
            self.axes[0, 0].plot(model.grid.time, row, lw=1.0)
        self.axes[0, 0].set(
            title=f"Sample Paths, H={hurst:.2f}", xlabel="t", ylabel="B_H(t)"
        )

        image = self.axes[0, 1].imshow(
            empirical, origin="lower", aspect="auto", cmap="viridis"
        )
        self.axes[0, 1].set_title(f"Empirical Covariance, RMSE={rmse:.2e}")
        self.add_colorbar(image, self.axes[0, 1])

        self.axes[1, 0].plot(
            np.sqrt(np.mean(difference**2, axis=0)), color="#e74c3c", lw=2.0
        )
        self.axes[1, 0].set(title="Covariance Error by Lag", xlabel="Lag", ylabel="RMSE")
        self.axes[1, 0].grid(alpha=0.25)

        image = self.axes[1, 1].imshow(
            model.covariance, origin="lower", aspect="auto", cmap="viridis"
        )
        self.axes[1, 1].set_title("Theoretical Covariance")
        self.add_colorbar(image, self.axes[1, 1])


def main() -> None:
    FBMDashboard().run()


if __name__ == "__main__":
    main()
