"""Registry-based gold model dispatch."""

from __future__ import annotations

from collections.abc import Iterable

from .base import GoldPriceModel
from .bates_hawkes import FullBatesHawkesGoldModel
from .bates_poisson import BatesPoissonGoldModel
from .black_scholes import BlackScholesGoldModel
from .heston import HestonGoldModel


class GoldModelRegistry:
    def __init__(self, models: Iterable[GoldPriceModel]) -> None:
        self._models: dict[str, GoldPriceModel] = {}
        for model in models:
            if model.model_id in self._models:
                raise ValueError(f"duplicate gold model id: {model.model_id}")
            self._models[model.model_id] = model
        if not self._models:
            raise ValueError("at least one gold model is required")

    def get(self, model_id: str) -> GoldPriceModel:
        try:
            return self._models[model_id]
        except KeyError as exc:
            raise KeyError(f"unknown gold model: {model_id}") from exc

    @property
    def model_ids(self) -> tuple[str, ...]:
        return tuple(self._models)


def build_frozen_team8_registry() -> GoldModelRegistry:
    return GoldModelRegistry(
        (
            BlackScholesGoldModel(),
            HestonGoldModel(),
            BatesPoissonGoldModel(),
            FullBatesHawkesGoldModel(),
        )
    )
