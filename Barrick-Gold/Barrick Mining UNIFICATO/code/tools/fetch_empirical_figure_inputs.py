"""Fetch the current LSE inputs used by the unified empirical figures.

Licensed row-level records are written only below ``data/raw/lse_local``.
The public manifest records provenance, coverage, schemas and hashes without
serialising market values.  The API key is read locally and is never printed.
"""

from __future__ import annotations

import argparse
import hashlib
import json
import os
import sys
from datetime import datetime, timezone
from importlib.metadata import version
from pathlib import Path
from typing import Any


ROOT = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(ROOT / "src"))

from barrick_unified.lse_adapter import find_local_lse_key, write_private_snapshot  # noqa: E402


MARKET_SYMBOLS = ("GLD", "SLV", "SPY")
FX_SYMBOLS = (
    "EUR/USD",
    "USD/JPY",
    "GBP/USD",
    "USD/CAD",
    "USD/SEK",
    "USD/CHF",
)
YIELD_SYMBOL = "US10Y"
PIPELINE_VERSION = "1.0.0"


def sha256(path: Path) -> str:
    digest = hashlib.sha256()
    with path.open("rb") as handle:
        for chunk in iter(lambda: handle.read(1024 * 1024), b""):
            digest.update(chunk)
    return digest.hexdigest().upper()


def _date_span(rows: list[dict[str, Any]]) -> dict[str, str | None]:
    values: list[str] = []
    for row in rows:
        for key in ("timestamp", "ts", "date", "updated_at"):
            value = row.get(key)
            if value not in (None, ""):
                values.append(str(value))
                break
    return {"first": min(values) if values else None, "last": max(values) if values else None}


def _fields(rows: list[dict[str, Any]]) -> list[str]:
    return sorted({key for row in rows for key in row})


def _manifest_entry(rows: list[dict[str, Any]]) -> dict[str, Any]:
    return {
        "rows": len(rows),
        "date_span": _date_span(rows),
        "fields": _fields(rows),
    }


def fetch_payload(start: str, end: str) -> dict[str, Any]:
    key = find_local_lse_key()
    if not key:
        raise RuntimeError("LSE_API_KEY is not configured on this computer.")
    os.environ["LSE_API_KEY"] = key

    from lse import LSE

    client = LSE()
    underlyings = client.options_underlyings()
    available = {
        str(row).strip().upper()
        if isinstance(row, str)
        else str(row.get("symbol") or row.get("underlying") or row.get("ticker") or "").strip().upper()
        for row in underlyings
    }
    missing = sorted(set(MARKET_SYMBOLS) - available)
    if missing:
        raise RuntimeError(f"LSE option-underlying catalog does not contain: {missing}")

    market = {
        symbol: client.candles(
            symbol, "1d", start=start, end=end, limit=5000, order="asc"
        )
        for symbol in MARKET_SYMBOLS
    }
    fx = {
        symbol: client.candles(
            symbol, "1d", start=start, end=end, limit=5000, order="asc", dataset="fx"
        )
        for symbol in FX_SYMBOLS
    }
    yields = client.bond_yields(
        YIELD_SYMBOL, start=start, end=end, order="asc", limit=5000
    )
    return {
        "market_candles": market,
        "fx_candles": fx,
        "us10y_yields": yields,
    }


def build_manifest(payload: dict[str, Any], raw_path: Path, run_id: str, start: str, end: str) -> dict[str, Any]:
    return {
        "schema_version": "1.0",
        "status": "CURRENT_LSE_EMPIRICAL_FIGURE_INPUTS",
        "run_id": run_id,
        "generated_at_utc": datetime.now(timezone.utc).isoformat(),
        "provider": "London Strategic Edge",
        "lse_data_version": version("lse-data"),
        "pipeline_version": PIPELINE_VERSION,
        "pipeline_script": "tools/fetch_empirical_figure_inputs.py",
        "pipeline_script_sha256": sha256(Path(__file__)),
        "requested_window": {"start": start, "end_inclusive": end},
        "market_series": {
            symbol: _manifest_entry(rows)
            for symbol, rows in payload["market_candles"].items()
        },
        "fx_series": {
            symbol: _manifest_entry(rows)
            for symbol, rows in payload["fx_candles"].items()
        },
        "yield_series": {YIELD_SYMBOL: _manifest_entry(payload["us10y_yields"])},
        "dxy_proxy": {
            "formula": "100*C*(EURUSD^-0.576)*(USDJPY^0.136)*(GBPUSD^-0.119)*(USDCAD^0.091)*(USDSEK^0.042)*(USDCHF^0.036)",
            "normalisation": "C chosen so the first aligned observation equals 100",
            "status": "TRANSPARENT_LSE_FX_PROXY_NOT_OFFICIAL_ICE_DXY",
        },
        "raw_storage": raw_path.relative_to(ROOT).as_posix(),
        "raw_sha256": sha256(raw_path),
        "redistribution": "Row-level LSE data remain local and Git-ignored.",
        "not_investment_advice": True,
    }


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--start", default="2021-01-01")
    parser.add_argument(
        "--end", default=datetime.now(timezone.utc).date().isoformat()
    )
    parser.add_argument(
        "--run-id",
        default=datetime.now(timezone.utc).strftime("%Y%m%dT%H%M%SZ-empirical-figures"),
    )
    return parser.parse_args()


def main() -> int:
    args = parse_args()
    raw_dir = ROOT / "data" / "raw" / "lse_local" / args.run_id
    manifest_dir = ROOT / "data" / "manifests" / args.run_id
    if raw_dir.exists() or manifest_dir.exists():
        raise FileExistsError(f"Refusing to overwrite run {args.run_id}")
    raw_path = raw_dir / "empirical_figure_inputs.json"
    payload = fetch_payload(args.start, args.end)
    private = {
        "schema_version": "1.0",
        "run_id": args.run_id,
        "generated_at_utc": datetime.now(timezone.utc).isoformat(),
        "provider": "London Strategic Edge",
        "request": {"start": args.start, "end": args.end},
        "rows": payload,
    }
    write_private_snapshot(raw_path, private)
    manifest = build_manifest(payload, raw_path, args.run_id, args.start, args.end)
    manifest_dir.mkdir(parents=True, exist_ok=False)
    manifest_path = manifest_dir / "empirical_figure_inputs_manifest.json"
    manifest_path.write_text(json.dumps(manifest, indent=2), encoding="utf-8")
    print(json.dumps({
        "run_id": args.run_id,
        "raw_path": raw_path.relative_to(ROOT).as_posix(),
        "manifest_path": manifest_path.relative_to(ROOT).as_posix(),
        "manifest_sha256": sha256(manifest_path),
        "coverage": {
            "market": {k: len(v) for k, v in payload["market_candles"].items()},
            "fx": {k: len(v) for k, v in payload["fx_candles"].items()},
            "us10y": len(payload["us10y_yields"]),
        },
    }, indent=2))
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
