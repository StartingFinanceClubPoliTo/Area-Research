"""Fetch or replay the current LSE Barrick/GLD research snapshot."""

from __future__ import annotations

import argparse
import hashlib
import json
import sys
from datetime import datetime, timezone
from importlib.metadata import version
from pathlib import Path

ROOT = Path(__file__).resolve().parent
sys.path.insert(0, str(ROOT / "src"))

from barrick_unified.lse_adapter import (  # noqa: E402
    LSEMarketDataProvider,
    read_private_snapshot,
    write_private_snapshot,
)
from barrick_unified.research_snapshot import build_public_outputs  # noqa: E402


PIPELINE_VERSION = "0.1.0"


def sha256(path: Path) -> str:
    digest = hashlib.sha256()
    with path.open("rb") as handle:
        for chunk in iter(lambda: handle.read(1024 * 1024), b""):
            digest.update(chunk)
    return digest.hexdigest().upper()


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(description=__doc__)
    source = parser.add_mutually_exclusive_group(required=True)
    source.add_argument("--fetch", action="store_true", help="Fetch a new LSE snapshot.")
    source.add_argument("--from-raw", type=Path, help="Replay an existing versioned private raw run directory.")
    parser.add_argument("--symbols", nargs="+", default=["B", "GOLD", "GLD"])
    parser.add_argument("--start", default="2021-01-01")
    parser.add_argument(
        "--end",
        default=datetime.now(timezone.utc).date().isoformat(),
        help="Inclusive UTC data cutoff (YYYY-MM-DD).",
    )
    parser.add_argument("--run-id", default=None, help="Optional immutable run identifier.")
    parser.add_argument("--rolling-window", type=int, default=63)
    return parser.parse_args()


def main() -> int:
    args = parse_args()
    if args.fetch:
        run_id = args.run_id or datetime.now(timezone.utc).strftime("%Y%m%dT%H%M%SZ")
        raw_dir = ROOT / "data/raw/lse_local" / run_id
        if raw_dir.exists():
            raise FileExistsError(f"Refusing to overwrite existing raw run: {raw_dir}")
    else:
        raw_dir = args.from_raw.resolve()
        source_run_id = raw_dir.name
        run_id = args.run_id or (
            source_run_id + "-replay-" + datetime.now(timezone.utc).strftime("%Y%m%dT%H%M%SZ")
        )
    equity_raw = raw_dir / "barrick_equity_candles.json"
    gld_raw = raw_dir / "gld_market_inputs.json"
    if args.fetch:
        provider = LSEMarketDataProvider()
        candles, unavailable_symbols, catalog_entries = provider.fetch_daily_market_candles(
            args.symbols, args.start, args.end
        )
        option_rows, yield_rows = provider.fetch_gld_option_inputs()
        generated_at = datetime.now(timezone.utc).isoformat()
        common = {
            "schema_version": "1.0",
            "run_id": run_id,
            "generated_at_utc": generated_at,
            "provider": "London Strategic Edge",
            "lse_data_version": version("lse-data"),
            "pipeline_version": PIPELINE_VERSION,
            "pipeline_script_sha256": sha256(Path(__file__)),
            "catalog_entries": catalog_entries,
        }
        request = {
            "stock_symbols": [str(item).upper() for item in args.symbols],
            "stock_dataset": "stocks",
            "timeframe": "1d",
            "start": args.start,
            "end": args.end,
            "gld_option_type": "call",
            "gld_max_dte": 1000,
            "yield_lookback_days": 120,
            "catalog_unavailable_symbols": unavailable_symbols,
        }
        equity_payload = {
            **common,
            "dataset": "Barrick_equity_series",
            "request": request,
            "rows": {
                "daily_stock_candles": {
                    symbol: rows for symbol, rows in candles.items() if symbol != "GLD"
                }
            },
        }
        gld_payload = {
            **common,
            "dataset": "GLD_market_and_current_option_inputs",
            "request": request,
            "rows": {
                "daily_stock_candles": {
                    symbol: rows for symbol, rows in candles.items() if symbol == "GLD"
                },
                "gld_option_calls": option_rows,
                "usd_treasury_yields": yield_rows,
            },
        }
        write_private_snapshot(equity_raw, equity_payload)
        write_private_snapshot(gld_raw, gld_payload)
        snapshot = {
            **common,
            "request": request,
            "catalog_entries": catalog_entries,
            "rows": {
                "daily_stock_candles": candles,
                "gld_option_calls": option_rows,
                "usd_treasury_yields": yield_rows,
            },
        }
    else:
        equity_payload = read_private_snapshot(equity_raw)
        gld_payload = read_private_snapshot(gld_raw)
        if equity_payload.get("run_id") != source_run_id or gld_payload.get("run_id") != source_run_id:
            raise ValueError("Raw dataset run IDs do not match the directory run ID.")
        snapshot = {
            **{key: equity_payload.get(key) for key in (
                "schema_version", "run_id", "generated_at_utc", "provider",
                "lse_data_version", "pipeline_version", "pipeline_script_sha256",
            )},
            "catalog_entries": equity_payload.get("catalog_entries", {}),
            "source_run_id": source_run_id,
            "run_id": run_id,
            "request": equity_payload.get("request", {}),
            "rows": {
                "daily_stock_candles": {
                    **equity_payload.get("rows", {}).get("daily_stock_candles", {}),
                    **gld_payload.get("rows", {}).get("daily_stock_candles", {}),
                },
                "gld_option_calls": gld_payload.get("rows", {}).get("gld_option_calls", []),
                "usd_treasury_yields": gld_payload.get("rows", {}).get("usd_treasury_yields", []),
            },
        }

    manifest_dir = ROOT / "data/manifests" / run_id
    manifest_path = manifest_dir / "run_manifest.json"

    manifest = build_public_outputs(
        raw_snapshot_paths={
            "barrick_equity": equity_raw,
            "gld_market_inputs": gld_raw,
        },
        snapshot=snapshot,
        output_dir=ROOT / "outputs/current" / run_id,
        figure_dir=ROOT / "figures/current" / run_id,
        manifest_path=manifest_path,
        team8_root=ROOT / "parity/sources/team-8",
        project_root=ROOT,
        cutoff=args.end,
        rolling_window=args.rolling_window,
    )
    report = {
        "run_id": run_id,
        "cutoff": args.end,
        "manifest": str(manifest_path.relative_to(ROOT)),
        "market_series": [
            {key: row[key] for key in ("symbol", "bars", "start_utc", "end_utc")}
            for row in manifest["market_series"]
        ],
        "option_surface_gate": manifest["option_surface_gate"],
        "historical_g1_5": "NOT_PROVEN",
    }
    print(json.dumps(report, indent=2))
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
