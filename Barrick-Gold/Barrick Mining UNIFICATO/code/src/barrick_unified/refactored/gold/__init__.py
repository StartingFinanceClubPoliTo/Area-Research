"""Gold-price models. These classes never own production, costs or DCF."""

from .base import GoldPathResult, GoldPriceModel, GoldSimulationContext
from .registry import GoldModelRegistry, build_frozen_team8_registry

__all__ = [
    "GoldModelRegistry",
    "GoldPathResult",
    "GoldPriceModel",
    "GoldSimulationContext",
    "build_frozen_team8_registry",
]
