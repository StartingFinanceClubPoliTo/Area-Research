"""Generate deterministic article-style diagnostics for HFT Article 2."""

from __future__ import annotations

import argparse
from dataclasses import dataclass
from pathlib import Path

import matplotlib

matplotlib.use("Agg")
import matplotlib.pyplot as plt
import numpy as np

from rough_processes import (
    DaviesHarteFractionalBrownianMotion,
    MarkovianLift,
    SimulationGrid,
    VolterraProcess,
    covariance_error,
    ensure_dir,
    fbm_covariance,
    lagwise_covariance_error,
)


@dataclass(frozen=True)
class FigureConfig:
    steps: int = 300
    sample_paths: int = 10
    lift_paths: int = 10
    lift_factors: int = 10
    dpi: int = 160


class ArticleFigureGenerator:
    """Own output policy and shared numerical settings for all figures."""

    def __init__(self, output: str | Path, config: FigureConfig | None = None) -> None:
        self.output = ensure_dir(output)
        self.config = config or FigureConfig()
        self.grid = SimulationGrid(self.config.steps)

    @staticmethod
    def _seed(base: int, hurst: float) -> int:
        return base + int(hurst * 100)

    def _save(self, figure, name: str) -> None:
        if figure._suptitle is None:
            figure.tight_layout()
        else:
            figure.tight_layout(rect=(0.0, 0.0, 1.0, 0.96))
        figure.savefig(self.output / name, dpi=self.config.dpi)
        plt.close(figure)

    def save_fbm(self, hurst: float) -> None:
        model = DaviesHarteFractionalBrownianMotion(self.grid, hurst)
        paths = model.simulate(self.config.sample_paths, self._seed(100, hurst))
        difference, rmse = covariance_error(paths, model.covariance)
        figure, axes = plt.subplots(2, 2, figsize=(10, 8))
        for row in paths:
            axes[0, 0].plot(self.grid.time, row, lw=1)
        axes[0, 0].set_title("Sample Paths")
        image = axes[0, 1].imshow(np.cov(paths, rowvar=False), origin="lower", aspect="auto")
        figure.colorbar(image, ax=axes[0, 1])
        axes[0, 1].set_title(f"Covariance\nRMSE: {rmse:.2e}")
        axes[1, 0].plot(
            lagwise_covariance_error(difference), color="#e74c3c", lw=2
        )
        axes[1, 0].set(
            title="Covariance Error by Lag", xlabel="Lag", ylabel="Mean absolute error"
        )
        axes[1, 0].grid(alpha=0.25)
        image = axes[1, 1].imshow(model.covariance, origin="lower", aspect="auto")
        figure.colorbar(image, ax=axes[1, 1])
        axes[1, 1].set_title("Theoretical Covariance")
        figure.suptitle(
            f"Time Steps: {self.config.steps}    Sample Paths: {self.config.sample_paths}    "
            f"Hurst H: {hurst:.1f}"
        )
        self._save(figure, f"fgm{hurst:.1f}.png")

    def save_volterra(self, hurst: float) -> None:
        model = VolterraProcess(self.grid, hurst)
        paths, brownian = model.simulate(
            self.config.sample_paths, self._seed(200, hurst)
        )
        target = fbm_covariance(self.grid.time, hurst)
        induced = model.induced_covariance
        covariance_rmse = float(np.sqrt(np.mean((induced - target) ** 2)))
        kernel_limit = max(float(model.kernel.max()), 1e-12)
        covariance_limit = max(float(induced.max()), float(target.max()), 1e-12)
        figure, axes = plt.subplots(3, 2, figsize=(10, 12))
        for row in paths:
            axes[0, 0].plot(self.grid.time, row, lw=1)
        axes[0, 0].set_title("Sample Paths")
        image = axes[0, 1].imshow(
            model.kernel,
            origin="lower",
            aspect="auto",
            cmap="viridis",
            vmin=0.0,
            vmax=kernel_limit,
        )
        colorbar = figure.colorbar(image, ax=axes[0, 1])
        colorbar.set_label("K(t,s)")
        axes[0, 1].set_title("Volterra Kernel K(t,s)")
        for row in brownian[:3]:
            axes[1, 0].plot(self.grid.time, row, lw=1)
        axes[1, 0].set_title("Driving Brownian Paths")
        for row in np.diff(brownian[:3], axis=1):
            axes[1, 1].plot(self.grid.time[1:], row, lw=0.8)
        axes[1, 1].set_title("Brownian Increments")
        for fraction in (0.25, 0.50, 0.75):
            index = int(fraction * (self.config.steps - 1))
            (line,) = axes[2, 0].plot(
                self.grid.time,
                model.kernel[index],
                lw=2,
                drawstyle="steps-post",
                label=f"t={self.grid.time[index]:.2f}",
            )
            axes[2, 0].axvline(
                self.grid.time[index], color=line.get_color(), ls="--", lw=0.9, alpha=0.5
            )
        axes[2, 0].legend()
        axes[2, 0].set(
            title="Causal Kernel Support",
            xlabel="s",
            ylabel="K(t,s)",
            xlim=(self.grid.time[0], self.grid.time[-1]),
        )
        if abs(hurst - 0.5) < 1e-12:
            axes[2, 0].text(
                0.02,
                0.05,
                "Brownian benchmark: K=1 on s<t, K=0 on s>=t",
                transform=axes[2, 0].transAxes,
                fontsize=8,
                bbox={"facecolor": "white", "alpha": 0.85, "edgecolor": "0.75"},
            )
        image = axes[2, 1].imshow(
            induced,
            origin="lower",
            aspect="auto",
            cmap="viridis",
            vmin=0.0,
            vmax=covariance_limit,
        )
        colorbar = figure.colorbar(image, ax=axes[2, 1])
        colorbar.set_label("Cov(X_t,X_s)")
        axes[2, 1].set_title(
            f"Kernel-Induced Covariance\nfBM target RMSE: {covariance_rmse:.2e}"
        )
        figure.suptitle(
            f"Time Steps: {self.config.steps}    Sample Paths: {self.config.sample_paths}    "
            f"Hurst H: {hurst:.1f}"
        )
        self._save(figure, f"volterra{hurst:.1f}.png")

    def save_iterative(self, hurst: float) -> None:
        model = VolterraProcess(self.grid, hurst)
        paths, _ = model.simulate(3, self._seed(222, hurst))
        figure, axes = plt.subplots(2, 2, figsize=(10, 8))
        for axis, window in zip(axes.ravel(), (0.25, 0.50, 0.75, 1.00)):
            stop = int(window * (self.config.steps - 1))
            for index, row in enumerate(paths):
                axis.plot(
                    self.grid.time[:stop], row[:stop], lw=1.5, label=f"Path {index + 1}"
                )
            axis.axvline(self.grid.time[stop], color="red", ls="--", lw=1)
            axis.set(title=f"Filtration Window s: {window:g}", xlabel="t", ylabel="X(t)")
        figure.suptitle(f"Causal Volterra Construction    Hurst H: {hurst:.1f}")
        self._save(figure, f"iterative{hurst:.1f}.png")

    def save_lift(self, hurst: float) -> None:
        model = MarkovianLift(self.grid, hurst, self.config.lift_factors)
        paths = model.simulate(self.config.lift_paths, self._seed(300, hurst))
        target = fbm_covariance(self.grid.time, hurst)
        difference, rmse = covariance_error(paths, target)
        empirical = np.cov(paths, rowvar=False)
        figure, axes = plt.subplots(2, 2, figsize=(10, 8))
        for row in paths[:10]:
            axes[0, 0].plot(self.grid.time, row, lw=1)
        axes[0, 0].set_title("Sample Paths")
        axes[0, 1].plot(
            lagwise_covariance_error(difference), color="#e74c3c", lw=2
        )
        axes[0, 1].set(
            title="Covariance Error by Lag", xlabel="Lag", ylabel="Mean absolute error"
        )
        axes[0, 1].grid(alpha=0.25)
        image = axes[1, 0].imshow(target, origin="lower", aspect="auto")
        figure.colorbar(image, ax=axes[1, 0])
        axes[1, 0].set_title("Theoretical Covariance")
        image = axes[1, 1].imshow(empirical, origin="lower", aspect="auto")
        figure.colorbar(image, ax=axes[1, 1])
        axes[1, 1].set_title(f"Covariance\nRMSE: {rmse:.2e}")
        figure.suptitle(
            f"Time Steps: {self.config.steps}    Sample Paths: {self.config.lift_paths}    "
            f"Hurst H: {hurst:.1f}    OU factors M: {self.config.lift_factors}"
        )
        self._save(figure, f"ML{hurst:.1f}.png")

    def generate(self, hurst_values: tuple[float, ...] = (0.2, 0.5, 0.8)) -> None:
        for hurst in hurst_values:
            self.save_fbm(hurst)
            self.save_volterra(hurst)
            self.save_lift(hurst)
            self.save_iterative(hurst)


def save_fbm(output: Path, hurst: float) -> None:
    ArticleFigureGenerator(output).save_fbm(hurst)


def save_volterra(output: Path, hurst: float) -> None:
    ArticleFigureGenerator(output).save_volterra(hurst)


def save_iterative(output: Path, hurst: float) -> None:
    ArticleFigureGenerator(output).save_iterative(hurst)


def save_lift(output: Path, hurst: float) -> None:
    ArticleFigureGenerator(output).save_lift(hurst)


def main() -> None:
    parser = argparse.ArgumentParser()
    parser.add_argument("--output", default="output/generated_figures")
    arguments = parser.parse_args()
    ArticleFigureGenerator(arguments.output).generate()


if __name__ == "__main__":
    main()
