"""Canonical LSE candle boundary and trailing market statistics.

All time-series outputs are chronological, UTC-based and backward-looking.
No price or volume is forward-filled. A value reported at timestamp ``t`` uses
only observations at or before ``t``.
"""

from __future__ import annotations

from collections.abc import Iterable, Mapping
from typing import Any

import numpy as np
import pandas as pd


BAR_COLUMNS = ("timestamp", "symbol", "open", "high", "low", "close", "volume")
TRADING_DAYS = 252.0


class BarSchemaError(ValueError):
    """Raised when provider rows cannot satisfy the canonical bar contract."""


def canonicalise_lse_candles(
    rows: Iterable[Mapping[str, Any]], symbol: str
) -> pd.DataFrame:
    """Normalize LSE daily candles without silently repairing bad rows."""

    frame = pd.DataFrame(list(rows))
    missing = sorted(set(BAR_COLUMNS).difference(frame.columns))
    if missing:
        raise BarSchemaError(f"LSE candle response is missing columns: {missing}")
    if frame.empty:
        raise BarSchemaError(f"LSE returned no daily candles for {symbol}.")

    normalized = frame.loc[:, BAR_COLUMNS].copy()
    normalized["timestamp"] = pd.to_datetime(
        normalized["timestamp"], errors="coerce", utc=True
    )
    for column in ("open", "high", "low", "close", "volume"):
        normalized[column] = pd.to_numeric(normalized[column], errors="coerce")

    expected = symbol.strip().upper()
    provider_symbols = normalized["symbol"].astype(str).str.strip().str.upper()
    if not provider_symbols.eq(expected).all():
        observed = sorted(provider_symbols.dropna().unique().tolist())
        raise BarSchemaError(
            f"Provider symbol mismatch for {expected}: observed {observed}."
        )
    normalized["symbol"] = expected

    if normalized.isna().any().any():
        bad_columns = normalized.columns[normalized.isna().any()].tolist()
        raise BarSchemaError(f"Null or non-numeric bar fields: {bad_columns}")

    normalized = normalized.sort_values("timestamp", kind="mergesort").reset_index(drop=True)
    if normalized["timestamp"].duplicated().any():
        raise BarSchemaError("Duplicate timestamps are not accepted at the adapter boundary.")
    if not normalized["timestamp"].is_monotonic_increasing:
        raise BarSchemaError("Timestamps must be chronological.")

    price_columns = ["open", "high", "low", "close"]
    if normalized[price_columns].le(0.0).any().any():
        raise BarSchemaError("OHLC prices must be strictly positive.")
    if normalized["volume"].lt(0.0).any():
        raise BarSchemaError("Volume cannot be negative.")
    if normalized["high"].lt(normalized[price_columns].max(axis=1)).any():
        raise BarSchemaError("High is below another OHLC field.")
    if normalized["low"].gt(normalized[price_columns].min(axis=1)).any():
        raise BarSchemaError("Low is above another OHLC field.")

    return normalized.astype(
        {"open": "float64", "high": "float64", "low": "float64", "close": "float64", "volume": "float64"}
    )


def log_returns(bars: pd.DataFrame) -> pd.Series:
    """Close-to-close log returns aligned to the observation timestamp."""

    values = np.log(bars["close"].astype("float64")).diff()
    values.name = str(bars["symbol"].iloc[0])
    return values


def compute_market_summary(bars: pd.DataFrame) -> dict[str, Any]:
    """Compute reproducible full-sample descriptive statistics.

    Volatility and mean log return use 252 trading days. Skewness and excess
    kurtosis use pandas' bias-corrected sample estimators. Historical 5% VaR is
    a return quantile, not a forward forecast or investment recommendation.
    """

    returns = log_returns(bars).dropna()
    if len(returns) < 2:
        raise ValueError("At least three prices are required for market statistics.")
    close = bars["close"].astype("float64")
    drawdown = close.div(close.cummax()).sub(1.0)
    negative = returns.loc[returns.lt(0.0)]
    downside = (
        float(negative.std(ddof=1) * np.sqrt(TRADING_DAYS))
        if len(negative) >= 2
        else np.nan
    )
    return {
        "symbol": str(bars["symbol"].iloc[0]),
        "bars": int(len(bars)),
        "return_observations": int(len(returns)),
        "start_utc": bars["timestamp"].iloc[0].isoformat(),
        "end_utc": bars["timestamp"].iloc[-1].isoformat(),
        "total_simple_return": float(close.iloc[-1] / close.iloc[0] - 1.0),
        "annualized_mean_log_return": float(returns.mean() * TRADING_DAYS),
        "annualized_volatility": float(returns.std(ddof=1) * np.sqrt(TRADING_DAYS)),
        "annualized_downside_volatility": downside,
        "daily_skewness": float(returns.skew()),
        "daily_excess_kurtosis": float(returns.kurt()),
        "historical_var_05_log_return": float(returns.quantile(0.05)),
        "maximum_drawdown": float(drawdown.min()),
    }


def rolling_statistics(
    bars: pd.DataFrame, window: int = 63, min_periods: int | None = None
) -> pd.DataFrame:
    """Trailing annualized volatility and cumulative log return in O(n)."""

    if window < 1:
        raise ValueError("window must be positive")
    minimum = window if min_periods is None else int(min_periods)
    if not 1 <= minimum <= window:
        raise ValueError("min_periods must be between 1 and window")
    returns = log_returns(bars)
    result = bars.loc[:, ["timestamp", "symbol"]].copy()
    result["log_return"] = returns
    result["annualized_volatility"] = returns.rolling(
        window=window, min_periods=minimum
    ).std(ddof=1) * np.sqrt(TRADING_DAYS)
    result["trailing_log_return"] = returns.rolling(
        window=window, min_periods=minimum
    ).sum()
    return result


def rolling_correlation(
    left: pd.DataFrame,
    right: pd.DataFrame,
    window: int = 63,
    min_periods: int | None = None,
) -> pd.DataFrame:
    """Inner-date trailing correlation of two close-to-close return series."""

    if window < 2:
        raise ValueError("window must be at least 2")
    minimum = window if min_periods is None else int(min_periods)
    if not 2 <= minimum <= window:
        raise ValueError("min_periods must be between 2 and window")
    left_returns = pd.DataFrame(
        {"timestamp": left["timestamp"], "left_return": log_returns(left)}
    )
    right_returns = pd.DataFrame(
        {"timestamp": right["timestamp"], "right_return": log_returns(right)}
    )
    aligned = left_returns.merge(right_returns, on="timestamp", how="inner")
    aligned["rolling_correlation"] = aligned["left_return"].rolling(
        window=window, min_periods=minimum
    ).corr(aligned["right_return"])
    return aligned
