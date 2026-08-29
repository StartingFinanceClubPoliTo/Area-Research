"""Retry the LSE US10Y endpoint without mixing asynchronous curve dates.

Raw licensed responses remain in ``data/raw/lse_local``.  The public audit
contains only request coverage, dates and the single curve-date observation
needed to decide whether US10Y can join the current NSS cross-section.
"""

from __future__ import annotations

import argparse
from datetime import datetime, timezone
import json
import os
from pathlib import Path
import time


ROOT = Path(__file__).resolve().parents[1]
CURVE_DATE = "2026-07-24"


def main() -> None:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--run-id", required=True)
    parser.add_argument("--delay-seconds", type=float, default=3.0)
    args = parser.parse_args()
    if not os.environ.get("LSE_API_KEY"):
        raise RuntimeError("LSE_API_KEY is not configured")

    from lse import LSE

    client = LSE()
    attempts = [
        {"label": "narrow", "start": "2026-07-20", "end": "2026-07-31"},
        {"label": "curve_date", "start": CURVE_DATE, "end": "2026-07-25"},
        {"label": "medium", "start": "2026-07-01", "end": "2026-08-28"},
        {"label": "wide", "start": "2026-01-01", "end": "2026-08-28"},
    ]
    raw_payload: dict[str, object] = {
        "fetched_at_utc": datetime.now(timezone.utc).isoformat(),
        "symbol": "US10Y",
        "curve_date": CURVE_DATE,
        "delay_seconds": args.delay_seconds,
        "attempts": [],
    }
    summaries = []
    all_rows: list[dict] = []
    for index, request in enumerate(attempts):
        if index:
            time.sleep(args.delay_seconds)
        rows = client.bond_yields(
            "US10Y",
            start=request["start"],
            end=request["end"],
            order="asc",
            limit=5000,
        )
        rows = list(rows)
        all_rows.extend(rows)
        dates = sorted({str(row.get("date", ""))[:10] for row in rows if row.get("date")})
        exact = [row for row in rows if str(row.get("date", ""))[:10] == CURVE_DATE]
        raw_payload["attempts"].append({"request": request, "rows": rows})
        summaries.append(
            {
                "label": request["label"],
                "start": request["start"],
                "end": request["end"],
                "row_count": len(rows),
                "first_date": dates[0] if dates else None,
                "last_date": dates[-1] if dates else None,
                "curve_date_present": bool(exact),
            }
        )

    time.sleep(args.delay_seconds)
    catalog_rows = [
        row for row in client.catalog("bonds") if str(row.get("symbol")) == "US10Y"
    ]
    raw_payload["catalog"] = catalog_rows
    unique = {}
    for row in all_rows:
        date = str(row.get("date", ""))[:10]
        if date:
            unique[date] = row
    exact_row = unique.get(CURVE_DATE)
    public = {
        "schema_version": "1.0",
        "run_id": args.run_id,
        "provider": "London Strategic Edge",
        "endpoint": "/ref/bond_yields",
        "symbol": "US10Y",
        "curve_date_requested": CURVE_DATE,
        "delay_seconds_between_requests": args.delay_seconds,
        "attempts": summaries,
        "catalog": [
            {
                "symbol": row.get("symbol"),
                "name": row.get("name"),
                "first": row.get("first"),
                "last": row.get("last"),
                "ticks": row.get("ticks"),
            }
            for row in catalog_rows
        ],
        "unique_observation_dates_returned": len(unique),
        "latest_observation_date_returned": max(unique) if unique else None,
        "curve_date_present": exact_row is not None,
        "curve_date_par_yield_pct": (
            float(exact_row["close"]) if exact_row is not None else None
        ),
        "decision": (
            "ADMIT_US10Y_ON_CURRENT_CURVE_DATE"
            if exact_row is not None
            else "DO_NOT_MIX_ASYNCHRONOUS_US10Y_WITH_2026_07_24_CURVE"
        ),
        "redistribution": "raw rows local-only; this file contains aggregate coverage and one decision observation only",
    }

    raw_dir = ROOT / "data" / "raw" / "lse_local" / args.run_id
    audit_dir = ROOT / "data" / "manifests" / "rates"
    raw_dir.mkdir(parents=True, exist_ok=False)
    audit_dir.mkdir(parents=True, exist_ok=True)
    (raw_dir / "us10y_retry_raw.json").write_text(
        json.dumps(raw_payload, indent=2, default=str), encoding="utf-8"
    )
    output = audit_dir / f"lse_us10y_retry_{args.run_id}.json"
    output.write_text(json.dumps(public, indent=2, default=str), encoding="utf-8")
    print(json.dumps(public, indent=2, default=str))
    print(f"Public audit: {output}")


if __name__ == "__main__":
    main()
