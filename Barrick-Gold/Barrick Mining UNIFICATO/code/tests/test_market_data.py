from __future__ import annotations

import numpy as np
import pandas as pd
import pytest

from barrick_unified.market_data import (
    BarSchemaError,
    canonicalise_lse_candles,
    compute_market_summary,
    rolling_correlation,
    rolling_statistics,
)


def rows(symbol: str, closes: list[float]) -> list[dict[str, object]]:
    dates = pd.date_range("2026-01-01", periods=len(closes), freq="B", tz="UTC")
    return [
        {
            "timestamp": date.isoformat(),
            "symbol": symbol,
            "open": close * 0.99,
            "high": close * 1.01,
            "low": close * 0.98,
            "close": close,
            "volume": 1000 + index,
        }
        for index, (date, close) in enumerate(zip(dates, closes, strict=True))
    ]


def test_canonical_boundary_sorts_and_uses_float64() -> None:
    source = list(reversed(rows("B", [10.0, 10.2, 10.1])))
    bars = canonicalise_lse_candles(source, "B")
    assert bars["timestamp"].is_monotonic_increasing
    assert bars["close"].dtype == np.dtype("float64")


def test_duplicate_timestamp_is_rejected() -> None:
    source = rows("B", [10.0, 10.2])
    source.append(dict(source[-1]))
    with pytest.raises(BarSchemaError, match="Duplicate"):
        canonicalise_lse_candles(source, "B")


def test_summary_and_rolling_are_finite_after_warmup() -> None:
    bars = canonicalise_lse_candles(rows("GLD", np.linspace(100, 130, 90).tolist()), "GLD")
    summary = compute_market_summary(bars)
    rolling = rolling_statistics(bars, window=21)
    assert summary["bars"] == 90
    assert summary["total_simple_return"] == pytest.approx(0.3)
    assert rolling["annualized_volatility"].notna().sum() == 69


def test_no_lookahead_prefix_invariance() -> None:
    closes = (100 * np.exp(np.sin(np.arange(120) / 11) / 20)).tolist()
    full = canonicalise_lse_candles(rows("GLD", closes), "GLD")
    prefix = full.iloc[:80].copy()
    full_result = rolling_statistics(full, window=21).iloc[:80].reset_index(drop=True)
    prefix_result = rolling_statistics(prefix, window=21).reset_index(drop=True)
    pd.testing.assert_frame_equal(full_result, prefix_result)


def test_rolling_correlation_is_inner_aligned_and_prefix_invariant() -> None:
    left = canonicalise_lse_candles(rows("B", np.linspace(20, 35, 100).tolist()), "B")
    right = canonicalise_lse_candles(rows("GLD", np.linspace(100, 140, 100).tolist()), "GLD")
    full = rolling_correlation(left, right, window=20).iloc[:70].reset_index(drop=True)
    prefix = rolling_correlation(left.iloc[:70], right.iloc[:70], window=20).reset_index(drop=True)
    pd.testing.assert_frame_equal(full, prefix)
