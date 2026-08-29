"""Resolve Barrick Mining in LSE and cache licensed inputs locally.

The script never prints or writes the API key. Row-level responses are saved
only below ``data/raw/lse_local`` (ignored by Git). The adjacent audit manifest
contains provenance, coverage and schema information, but no market prices or
financial-statement values.
"""

from __future__ import annotations

import argparse
import json
import os
from datetime import datetime, timezone
from pathlib import Path
from typing import Any


ROOT = Path(__file__).resolve().parents[1]
RAW_DIR = ROOT / "data" / "raw" / "lse_local"
MANIFEST_DIR = ROOT / "data" / "manifests"
SEARCH_TERMS = ("barrick",)
HISTORICAL_ALIASES = ("B", "ABX", "GOLD")


def find_local_lse_key() -> str:
    """Read the process or Windows user environment without exposing the key."""
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


def _json_default(value: Any) -> str:
    if isinstance(value, (datetime, Path)):
        return value.isoformat() if isinstance(value, datetime) else str(value)
    return str(value)


def _date_span(rows: list[dict[str, Any]], candidates: tuple[str, ...]) -> dict[str, str | None]:
    values: list[str] = []
    for row in rows:
        for key in candidates:
            value = row.get(key)
            if value not in (None, ""):
                values.append(str(value))
                break
    return {"first": min(values) if values else None, "last": max(values) if values else None}


def _catalog_matches(client: Any) -> list[dict[str, Any]]:
    rows = client.catalog("stocks")
    matches = []
    for row in rows:
        haystack = " ".join(str(row.get(k, "")) for k in ("symbol", "name")).lower()
        if any(term in haystack for term in SEARCH_TERMS):
            matches.append(row)
    return matches


def build_snapshot(start: str) -> tuple[dict[str, Any], dict[str, Any]]:
    key = find_local_lse_key()
    if not key:
        raise RuntimeError("LSE_API_KEY is not configured on this computer.")
    os.environ["LSE_API_KEY"] = key

    from lse import LSE

    client = LSE()
    matches = _catalog_matches(client)
    symbols = sorted({str(row.get("symbol", "")).strip() for row in matches if row.get("symbol")})
    if not symbols:
        raise RuntimeError("No LSE stock catalog entry matched 'Barrick'.")

    datasets: dict[str, Any] = {}
    for symbol in symbols:
        profiles = client.company_profiles(symbol=symbol, limit=5000)
        fundamentals = client.fundamentals(symbol=symbol, limit=5000)
        reports = client.financial_reports(symbol=symbol, order="desc", limit=5000)
        candles = client.candles(
            symbol,
            "1d",
            start=start,
            order="asc",
            limit=5000,
            dataset="stocks",
        )
        datasets[symbol] = {
            "company_profiles": profiles,
            "fundamentals": fundamentals,
            "financial_reports": reports,
            "daily_candles": candles,
        }

    generated_at = datetime.now(timezone.utc).isoformat()
    raw = {
        "generated_at_utc": generated_at,
        "provider": "London Strategic Edge",
        "catalog_matches": matches,
        "datasets": datasets,
    }
    manifest_datasets = {}
    for symbol, payload in datasets.items():
        manifest_datasets[symbol] = {
            "row_counts": {name: len(rows) for name, rows in payload.items()},
            "company_profile_fields": sorted({key for row in payload["company_profiles"] for key in row}),
            "fundamental_fields": sorted({key for row in payload["fundamentals"] for key in row}),
            "financial_report_fields": sorted({key for row in payload["financial_reports"] for key in row}),
            "daily_candle_fields": sorted({key for row in payload["daily_candles"] for key in row}),
            "financial_report_span": _date_span(
                payload["financial_reports"], ("filing_date", "date", "period_end", "fiscal_date")
            ),
            "daily_candle_span": _date_span(payload["daily_candles"], ("ts", "date", "timestamp")),
        }
    manifest = {
        "generated_at_utc": generated_at,
        "provider": "London Strategic Edge",
        "purpose": "Barrick Mining entity resolution and current valuation-input coverage audit",
        "search_terms": list(SEARCH_TERMS),
        "historical_aliases_to_reconcile": list(HISTORICAL_ALIASES),
        "resolved_catalog_entries": [
            {key: row.get(key) for key in ("symbol", "name", "dataset", "country", "first", "last")}
            for row in matches
        ],
        "requested_candle_start": start,
        "datasets": manifest_datasets,
        "raw_storage": "data/raw/lse_local/barrick_snapshot.json",
        "redistribution": "row-level LSE data are local-only and excluded from Git",
        "authority_boundary": {
            "LSE": "market prices, market snapshots and provider financial datasets",
            "Barrick_SEC": "legal name, filing perimeter and authoritative reported statements",
            "model": "documented transformations from source inputs to valuation outputs",
        },
    }
    return raw, manifest


def main() -> int:
    parser = argparse.ArgumentParser()
    parser.add_argument("--start", default="2025-05-01", help="First daily candle date (YYYY-MM-DD).")
    parser.add_argument("--write", action="store_true", help="Write local raw cache and audit manifest.")
    args = parser.parse_args()

    raw, manifest = build_snapshot(args.start)
    if args.write:
        RAW_DIR.mkdir(parents=True, exist_ok=True)
        MANIFEST_DIR.mkdir(parents=True, exist_ok=True)
        (RAW_DIR / "barrick_snapshot.json").write_text(
            json.dumps(raw, indent=2, default=_json_default), encoding="utf-8"
        )
        (MANIFEST_DIR / "lse_barrick_audit.json").write_text(
            json.dumps(manifest, indent=2, default=_json_default), encoding="utf-8"
        )

    print(json.dumps(manifest, indent=2, default=_json_default))
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
