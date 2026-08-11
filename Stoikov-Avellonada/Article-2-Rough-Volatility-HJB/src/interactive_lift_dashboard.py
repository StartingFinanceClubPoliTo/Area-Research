"""Interactive Markovian-lifting dashboard with bounded redraw state."""

from __future__ import annotations

import numpy as np

from dashboard_base import InteractiveDashboard, SliderSpec
from rough_processes import MarkovianLift, SimulationGrid, covariance_error, fbm_covariance


class LiftDashboard(InteractiveDashboard):
    def __init__(self) -> None:
        super().__init__(
            2,
            2,
            (12, 8),
            dict(left=0.08, bottom=0.18, right=0.98, top=0.92, hspace=0.35, wspace=0.25),
            (
                SliderSpec("steps", "Time steps", (0.08, 0.10, 0.20, 0.03), 80, 420, 300, 10),
                SliderSpec("paths", "Sample paths", (0.33, 0.10, 0.20, 0.03), 5, 80, 25, 5),
                SliderSpec("hurst", "Hurst H", (0.58, 0.10, 0.16, 0.03), 0.10, 0.90, 0.50, 0.05),
                SliderSpec("factors", "OU factors", (0.80, 0.10, 0.16, 0.03), 3, 20, 10, 1),
            ),
        )

    def render(self) -> None:
        steps = int(self.value("steps"))
        paths = int(self.value("paths"))
        hurst = self.value("hurst")
        factors = int(self.value("factors"))
        model = MarkovianLift(SimulationGrid(steps), hurst, factors)
        samples = model.simulate(paths, seed=789)
        target = fbm_covariance(model.grid.time, hurst)
        difference, rmse = covariance_error(samples, target)
        empirical = np.cov(samples, rowvar=False)

        for row in samples[: min(paths, 20)]:
            self.axes[0, 0].plot(model.grid.time, row, lw=1.0)
        self.axes[0, 0].set(
            title=f"Lifted Paths, H={hurst:.2f}, M={factors}", xlabel="t", ylabel="Y(t)"
        )

        self.axes[0, 1].plot(
            np.sqrt(np.mean(difference**2, axis=0)), color="#e74c3c", lw=2.0
        )
        self.axes[0, 1].set(title="Covariance Error by Lag", xlabel="Lag", ylabel="RMSE")
        self.axes[0, 1].grid(alpha=0.25)

        image = self.axes[1, 0].imshow(
            target, origin="lower", aspect="auto", cmap="viridis"
        )
        self.axes[1, 0].set_title("Target fBM Covariance")
        self.add_colorbar(image, self.axes[1, 0])

        image = self.axes[1, 1].imshow(
            empirical, origin="lower", aspect="auto", cmap="viridis"
        )
        self.axes[1, 1].set_title(f"Lift Covariance, RMSE={rmse:.2e}")
        self.add_colorbar(image, self.axes[1, 1])


def main() -> None:
    LiftDashboard().run()


if __name__ == "__main__":
    main()
