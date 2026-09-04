"""Shared infrastructure for the interactive Matplotlib dashboards."""

from __future__ import annotations

from abc import ABC, abstractmethod
from dataclasses import dataclass

import matplotlib.pyplot as plt
from matplotlib.colorbar import Colorbar
from matplotlib.widgets import Slider


@dataclass(frozen=True)
class SliderSpec:
    name: str
    label: str
    position: tuple[float, float, float, float]
    minimum: float
    maximum: float
    initial: float
    step: float


class InteractiveDashboard(ABC):
    """Own a figure, sliders, and colorbars for a redrawable dashboard."""

    def __init__(
        self,
        rows: int,
        columns: int,
        figsize: tuple[float, float],
        adjustments: dict[str, float],
        slider_specs: tuple[SliderSpec, ...],
    ) -> None:
        self.figure, self.axes = plt.subplots(rows, columns, figsize=figsize)
        self.figure.subplots_adjust(**adjustments)
        self.sliders = {
            spec.name: Slider(
                self.figure.add_axes(spec.position),
                spec.label,
                spec.minimum,
                spec.maximum,
                valinit=spec.initial,
                valstep=spec.step,
            )
            for spec in slider_specs
        }
        self._colorbars: list[Colorbar] = []
        for slider in self.sliders.values():
            slider.on_changed(self._redraw)

    def value(self, name: str) -> float:
        return float(self.sliders[name].val)

    def clear(self) -> None:
        """Remove old colorbar axes before clearing plot axes.

        The original dashboards appended new colorbar axes on every slider
        change. Removing them here keeps redraw cost and figure size bounded.
        """

        for colorbar in self._colorbars:
            colorbar.remove()
        self._colorbars.clear()
        for axis in self.axes.ravel():
            axis.clear()

    def add_colorbar(self, image, axis) -> None:
        self._colorbars.append(
            self.figure.colorbar(image, ax=axis, fraction=0.046, pad=0.04)
        )

    def _redraw(self, _=None) -> None:
        self.clear()
        self.render()
        self.figure.canvas.draw_idle()

    @abstractmethod
    def render(self) -> None:
        """Draw the current slider state on the existing axes."""

    def run(self) -> None:
        self._redraw()
        plt.show()

