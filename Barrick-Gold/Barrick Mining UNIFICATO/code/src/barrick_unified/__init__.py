"""Provenance-safe market analytics for the Barrick unified research project."""

from .market_data import (
    BarSchemaError,
    canonicalise_lse_candles,
    compute_market_summary,
    rolling_correlation,
    rolling_statistics,
)
from .project import ProjectLayout, UnifiedWorkflow

__all__ = [
    "BarSchemaError",
    "canonicalise_lse_candles",
    "compute_market_summary",
    "rolling_correlation",
    "rolling_statistics",
    "ProjectLayout",
    "UnifiedWorkflow",
]
