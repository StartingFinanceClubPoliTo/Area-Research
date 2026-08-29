"""Official London Strategic Edge SDK adapter.

The API key is read locally and is never returned, logged or serialized.
Downstream analytics consume plain dictionaries and canonical DataFrames; they
do not import the vendor SDK.
"""

from __future__ import annotations

import json
import os
from datetime import datetime, timedelta, timezone
from pathlib import Path
from typing import Any, Iterable


TREASURY_TENORS = (
    "US1M", "US2M", "US3M", "US6M", "US1Y", "US2Y", "US3Y", "US5Y",
    "US7Y", "US10Y", "US20Y", "US30Y",
)


def find_local_lse_key() -> str:
    """Find the process/user-scoped key without exposing it."""

    key = os.environ.get("LSE_API_KEY", "").strip()
    if key:
        return key
    if os.name == "nt":
        try:
            import winreg

            with winreg.OpenKey(winreg.HKEY_CURRENT_USER, "Environment") as handle:
                value, _ = winreg.QueryValueEx(handle, "LSE_API_KEY")
            return str(value).strip()
        except (FileNotFoundError, OSError):
            return ""
    return ""


class LSEMarketDataProvider:
    """Narrow provider owned by the project, backed by the official SDK."""

    def __init__(self, client: Any | None = None) -> None:
        if client is None:
            key = find_local_lse_key()
            if not key:
                raise RuntimeError("LSE_API_KEY is not configured on this computer.")
            os.environ["LSE_API_KEY"] = key
            from lse import LSE

            client = LSE()
        self._client = client

    def validate_stock_symbols(self, symbols: Iterable[str]) -> dict[str, dict[str, Any]]:
        requested = tuple(dict.fromkeys(str(item).strip().upper() for item in symbols))
        catalog = self._client.catalog("stocks")
        by_symbol = {
            str(row.get("symbol", "")).strip().upper(): row
            for row in catalog
            if row.get("symbol")
        }
        missing = [symbol for symbol in requested if symbol not in by_symbol]
        if missing:
            raise RuntimeError(f"LSE stock catalog does not contain: {missing}")
        return {symbol: by_symbol[symbol] for symbol in requested}

    def fetch_daily_stock_candles(
        self, symbols: Iterable[str], start: str, end: str | None
    ) -> dict[str, list[dict[str, Any]]]:
        requested = tuple(dict.fromkeys(str(item).strip().upper() for item in symbols))
        self.validate_stock_symbols(requested)
        return {
            symbol: self._client.candles(
                symbol,
                "1d",
                start=start,
                end=end,
                limit=5000,
                order="asc",
                dataset="stocks",
            )
            for symbol in requested
        }

    def fetch_daily_market_candles(
        self, symbols: Iterable[str], start: str, end: str | None
    ) -> tuple[dict[str, list[dict[str, Any]]], list[str], dict[str, dict[str, Any]]]:
        """Fetch catalog-validated Barrick equities and the GLD option underlying.

        Historical aliases absent from the current catalog are reported rather
        than guessed or silently remapped. GLD is validated through the options
        underlyings catalog and fetched without a stock dataset override, which
        matches the frozen Team 8 ingestion contract.
        """

        requested = tuple(dict.fromkeys(str(item).strip().upper() for item in symbols))
        stock_catalog = self._client.catalog("stocks")
        stock_entries = {
            str(row.get("symbol", "")).strip().upper(): row
            for row in stock_catalog
            if row.get("symbol")
        }
        option_catalog = self._client.options_underlyings()
        option_underlyings: set[str] = set()
        for row in option_catalog:
            if isinstance(row, str):
                option_underlyings.add(row.strip().upper())
            elif isinstance(row, dict):
                for key in ("underlying", "symbol", "ticker"):
                    if row.get(key):
                        option_underlyings.add(str(row[key]).strip().upper())
                        break

        result: dict[str, list[dict[str, Any]]] = {}
        unavailable: list[str] = []
        catalog_evidence: dict[str, dict[str, Any]] = {}
        for symbol in requested:
            if symbol in stock_entries:
                result[symbol] = self._client.candles(
                    symbol, "1d", start=start, end=end, limit=5000,
                    order="asc", dataset="stocks"
                )
                catalog_evidence[symbol] = {
                    key: stock_entries[symbol].get(key)
                    for key in ("symbol", "name", "dataset", "country", "first", "last")
                }
            elif symbol == "GLD" and symbol in option_underlyings:
                result[symbol] = self._client.candles(
                    symbol, "1d", start=start, end=end, limit=5000, order="asc"
                )
            else:
                unavailable.append(symbol)
        if "B" in requested and "B" not in result:
            raise RuntimeError("Current Barrick symbol B is absent from the LSE stock catalog.")
        return result, unavailable, catalog_evidence

    def fetch_gld_option_inputs(
        self, max_dte: int = 1000, yield_lookback_days: int = 120
    ) -> tuple[list[dict[str, Any]], list[dict[str, Any]]]:
        option_rows = self._client.options(
            "GLD", type="call", max_dte=int(max_dte), limit=5000
        )
        start = (
            datetime.now(timezone.utc) - timedelta(days=int(yield_lookback_days))
        ).date().isoformat()
        yield_rows: list[dict[str, Any]] = []
        for symbol in TREASURY_TENORS:
            yield_rows.extend(
                self._client.bond_yields(symbol, start=start, order="asc", limit=5000)
            )
        return option_rows, yield_rows


def write_private_snapshot(path: Path, payload: dict[str, Any]) -> None:
    """Atomically write licensed row-level data below a Git-ignored path."""

    path.parent.mkdir(parents=True, exist_ok=True)
    temporary = path.with_suffix(path.suffix + ".partial")
    temporary.write_text(json.dumps(payload, indent=2, default=str), encoding="utf-8")
    temporary.replace(path)


def read_private_snapshot(path: Path) -> dict[str, Any]:
    return json.loads(path.read_text(encoding="utf-8"))
