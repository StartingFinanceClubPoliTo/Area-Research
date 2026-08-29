from __future__ import annotations

import json
from pathlib import Path

import numpy as np
import pandas as pd

from barrick_unified.research_snapshot import build_public_outputs


def candle_rows(symbol: str, scale: float) -> list[dict[str, object]]:
    dates = pd.date_range("2025-01-01", periods=100, freq="B", tz="UTC")
    closes = scale * np.exp(np.sin(np.arange(100) / 13) / 20 + np.arange(100) / 1000)
    return [
        {
            "timestamp": date.isoformat(),
            "symbol": symbol,
            "open": close * 0.995,
            "high": close * 1.01,
            "low": close * 0.99,
            "close": close,
            "volume": 1000 + index,
        }
        for index, (date, close) in enumerate(zip(dates, closes, strict=True))
    ]


def test_public_build_writes_aggregates_not_rows(tmp_path: Path) -> None:
    raw = tmp_path / "data/raw/lse_local/current_research_snapshot.json"
    raw.parent.mkdir(parents=True)
    snapshot = {
        "run_id": "fixture-run",
        "generated_at_utc": "2026-08-25T00:00:00+00:00",
        "request": {"stock_symbols": ["B", "GOLD", "GLD"]},
        "catalog_entries": {
            "B": {"symbol": "B", "first": "2025-03-01T00:00:00Z"}
        },
        "rows": {
            "daily_stock_candles": {
                "B": candle_rows("B", 40),
                "GOLD": candle_rows("GOLD", 30),
                "GLD": candle_rows("GLD", 200),
            },
            "gld_option_calls": [],
            "usd_treasury_yields": [],
        },
    }
    raw.write_text(json.dumps(snapshot, default=str), encoding="utf-8")
    manifest_path = tmp_path / "data/manifests/current_research_snapshot_manifest.json"
    manifest = build_public_outputs(
        {"barrick_equity": raw, "gld_market_inputs": raw},
        snapshot,
        tmp_path / "outputs/current",
        tmp_path / "figures/current",
        manifest_path,
        tmp_path / "parity/sources/team-8",
        tmp_path,
        "2026-08-25",
        rolling_window=21,
    )
    assert (tmp_path / "outputs/current/lse_market_summary.csv").is_file()
    assert (tmp_path / "figures/current/lse_market_context.png").stat().st_size > 1000
    assert manifest["option_surface_gate"]["status"] == "BLOCKED"
    barrick = next(row for row in manifest["market_series"] if row["symbol"] == "B")
    assert barrick["pre_entity_rows_discarded"] > 0
    assert barrick["bars"] < 100
    public_text = manifest_path.read_text(encoding="utf-8")
    assert "LSE_API_KEY" not in public_text
    assert '"close"' not in public_text
    assert "NOT_PROVEN" in public_text


def test_post_cutoff_candles_fail_closed(tmp_path: Path) -> None:
    raw = tmp_path / "data/raw/lse_local/run/barrick_equity_candles.json"
    raw.parent.mkdir(parents=True)
    snapshot = {
        "run_id": "run",
        "generated_at_utc": "2026-08-25T00:00:00+00:00",
        "request": {"start": "2025-01-01", "end": "2025-01-10"},
        "rows": {
            "daily_stock_candles": {"B": candle_rows("B", 40)},
            "gld_option_calls": [],
            "usd_treasury_yields": [],
        },
    }
    raw.write_text(json.dumps(snapshot, default=str), encoding="utf-8")
    import pytest

    with pytest.raises(ValueError, match="exceed the requested cutoff"):
        build_public_outputs(
            {"barrick_equity": raw, "gld_market_inputs": raw},
            snapshot,
            tmp_path / "outputs/current/run",
            tmp_path / "figures/current/run",
            tmp_path / "data/manifests/run/run_manifest.json",
            tmp_path / "parity/sources/team-8",
            tmp_path,
            "2025-01-10",
            rolling_window=21,
        )
