"""Build a local GLD option dataset from the London Strategic Edge API.

Retrieved LSE data is intentionally written below ``Data/lse_local`` (ignored
by Git). The committed artifact is this reproducible transformation, not a
redistributed market-data snapshot.
"""

import argparse
import json
import math
import os
from datetime import datetime, timezone
from pathlib import Path

import numpy as np
import pandas as pd

from Sampling import Sampling


WIDE_COLUMNS = (
    "expiry_iso", "expiry_ymd", "T", "right", "K", "conId",
    "localSymbol", "symbol", "bid", "ask", "mid", "last", "close",
    "model_price", "price_used", "price_source", "has_any_quote",
    "impliedVol", "delta", "gamma", "vega", "theta", "undPrice_model",
    "volume", "openInterest",
)
CALIBRATION_COLUMNS = (
    "expiry", "T", "K", "price", "rate", "implied_vol", "vega"
)


def _numeric(frame, column):
    if column not in frame:
        return pd.Series(np.nan, index=frame.index, dtype=float)
    return pd.to_numeric(frame[column], errors="coerce")


def normalise_lse_chain(rows):
    """Map LSE rows to the wide schema using one coherent snapshot date.

    LSE's ``dte`` belongs to each contract's own last update. Some less liquid
    rows therefore carry older DTE values. We infer one snapshot timestamp from
    the newest ``updated_at`` value (or from expiry minus DTE as a fallback) and
    recompute every maturity against that common date.
    """
    source = pd.DataFrame(rows)
    required = {"ticker", "underlying", "strike", "expiry", "contract_type"}
    missing = sorted(required.difference(source.columns))
    if missing:
        raise ValueError(f"LSE option response is missing columns: {missing}")
    if source.empty:
        raise ValueError("LSE returned an empty option chain.")

    expiry = pd.to_datetime(source["expiry"], errors="coerce", utc=True)
    dte = _numeric(source, "dte")
    if "updated_at" in source:
        updated_at = pd.to_datetime(source["updated_at"], errors="coerce", utc=True)
        snapshot = updated_at.max()
        latest_mask = updated_at.ge(snapshot - pd.Timedelta(hours=1))
    else:
        inferred = expiry - pd.to_timedelta(dte, unit="D")
        snapshot = inferred.dropna().median()
        latest_mask = pd.Series(True, index=source.index)
    if pd.isna(snapshot):
        raise ValueError("Cannot infer a coherent LSE snapshot timestamp.")
    snapshot_date = snapshot.normalize()
    maturity_days = (expiry - snapshot_date).dt.total_seconds() / 86400.0
    last_price = _numeric(source, "last_price")
    contract_type = source["contract_type"].astype(str).str.lower()
    right = contract_type.map({"call": "C", "c": "C", "put": "P", "p": "P"})

    wide = pd.DataFrame(index=source.index)
    wide["expiry_iso"] = expiry.dt.strftime("%Y-%m-%d")
    wide["expiry_ymd"] = expiry.dt.strftime("%Y%m%d")
    wide["T"] = maturity_days / 365.25
    wide["right"] = right
    wide["K"] = _numeric(source, "strike")
    wide["conId"] = np.nan
    wide["localSymbol"] = source["ticker"].astype(str)
    wide["symbol"] = source["underlying"].astype(str)
    for column in ("bid", "ask", "mid", "close", "model_price", "openInterest"):
        wide[column] = np.nan
    wide["last"] = last_price
    wide["price_used"] = last_price
    wide["price_source"] = "lse_last_price"
    wide["has_any_quote"] = np.isfinite(last_price) & (last_price > 0.0)
    wide["impliedVol"] = _numeric(source, "iv")
    wide["delta"] = _numeric(source, "delta")
    wide["gamma"] = _numeric(source, "gamma")
    wide["vega"] = _numeric(source, "vega")
    wide["theta"] = _numeric(source, "theta")
    wide["undPrice_model"] = _numeric(source, "underlying_price")
    wide["volume"] = _numeric(source, "volume_today")
    wide = wide.loc[wide["T"].gt(0.0), WIDE_COLUMNS]
    wide = wide.sort_values(["T", "K"]).reset_index(drop=True)
    latest_spot = _numeric(source, "underlying_price").loc[latest_mask]
    latest_spot = latest_spot.loc[latest_spot.gt(0.0)].median()
    if not np.isfinite(latest_spot):
        latest_spot = _numeric(source, "underlying_price").dropna().median()
    wide.attrs["as_of_utc"] = snapshot.isoformat()
    wide.attrs["snapshot_spot"] = float(latest_spot)
    return wide


def build_calibration_sample(
    wide,
    annual_rate=0.037,
    moneyness_window=0.20,
    min_dte=60,
    min_price=1.0,
    min_points_per_expiry=5,
    n_maturities=8,
    n_strikes=8,
):
    """Create a Chebyshev sample compatible with Bates/Heston calibration."""
    calls = wide.loc[wide["right"] == "C"].copy()
    spot = float(wide.attrs.get("snapshot_spot", np.nan))
    if not np.isfinite(spot):
        spot_values = calls["undPrice_model"].dropna()
        spot = float(spot_values.median()) if not spot_values.empty else np.nan
    if not np.isfinite(spot) or spot <= 0.0:
        raise ValueError("LSE response does not contain a valid underlying price.")
    lower_strike = spot * (1.0 - moneyness_window)
    upper_strike = spot * (1.0 + moneyness_window)

    calls = calls.loc[
        calls["price_used"].gt(min_price)
        & calls["T"].ge(min_dte / 365.25)
        & calls["K"].between(lower_strike, upper_strike)
        & calls["impliedVol"].gt(0.0)
        & calls["vega"].gt(0.0)
    ].copy()
    calls = calls.dropna(
        subset=["expiry_iso", "T", "K", "price_used", "impliedVol", "vega"]
    )
    calls = calls.drop_duplicates(["expiry_iso", "K"], keep="first")
    counts = calls.groupby("expiry_iso").size()
    calls = calls.loc[
        calls["expiry_iso"].isin(counts[counts >= min_points_per_expiry].index)
    ]
    if calls["T"].nunique() < 2 or calls["K"].nunique() < 2:
        raise ValueError("Too few valid LSE calls remain for two-dimensional sampling.")

    calibration = calls.rename(
        columns={
            "expiry_iso": "expiry",
            "price_used": "price",
            "impliedVol": "implied_vol",
        }
    )
    calibration["rate"] = float(annual_rate)
    calibration = calibration.loc[:, CALIBRATION_COLUMNS]
    calibration = calibration.sort_values(["T", "K"]).reset_index(drop=True)
    sample = Sampling.sample_chebyshev(
        calibration,
        n_T=min(n_maturities, calibration["T"].nunique()),
        n_K=min(n_strikes, calibration["K"].nunique()),
    )
    return calibration, sample, spot


def fetch_lse_calls(max_dte=1000, limit=5000):
    """Fetch the current GLD call chain using ``LSE_API_KEY`` from the environment."""
    if not os.environ.get("LSE_API_KEY"):
        raise RuntimeError("LSE_API_KEY is not configured in the environment.")
    from lse import LSE

    return LSE().options(
        "GLD", type="call", max_dte=int(max_dte), limit=int(limit)
    )


def write_local_dataset(rows, output_dir, annual_rate=0.037):
    """Write local-only wide, filtered, sampled, metadata, and audit outputs."""
    output_dir = Path(output_dir)
    output_dir.mkdir(parents=True, exist_ok=True)
    wide = normalise_lse_chain(rows)
    calibration, sample, spot = build_calibration_sample(
        wide, annual_rate=annual_rate
    )

    wide_path = output_dir / "gld_chain_wide_chain.csv"
    full_path = output_dir / "gld_iv_dataset_full.csv"
    sample_path = output_dir / "gld_iv_dataset_chebyshev.csv"
    metadata_path = output_dir / "gld_chain_wide_meta.json"
    audit_path = output_dir / "lse_build_audit.json"
    wide.to_csv(wide_path, index=False)
    calibration.to_csv(full_path, index=False)
    sample.to_csv(sample_path, index=False)

    as_of = wide.attrs.get("as_of_utc", datetime.now(timezone.utc).isoformat())
    metadata = {
        "underlying_symbol": "GLD",
        "S0": spot,
        "as_of_utc": as_of,
        "source": "London Strategic Edge /options/chain",
        "snapshot_kind": "current_chain_not_historical_reconstruction",
        "risk_free_rate_assumption": float(annual_rate),
        "n_rows_total": int(len(wide)),
        "n_rows_calibration_eligible": int(len(calibration)),
        "n_rows_chebyshev": int(len(sample)),
        "redistribution": "prohibited; local-only outputs ignored by Git",
    }
    metadata_path.write_text(json.dumps(metadata, indent=2), encoding="utf-8")

    audit = {
        "success": True,
        "schema_matches_historical_wide": list(wide.columns) == list(WIDE_COLUMNS),
        "schema_matches_calibration": list(sample.columns) == list(CALIBRATION_COLUMNS),
        "files": [path.name for path in (wide_path, full_path, sample_path, metadata_path)],
        **metadata,
    }
    audit_path.write_text(json.dumps(audit, indent=2), encoding="utf-8")
    return audit_path, audit


def parse_args():
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--output-dir", default="Data/lse_local")
    parser.add_argument("--max-dte", type=int, default=1000)
    parser.add_argument("--limit", type=int, default=5000)
    parser.add_argument("--annual-rate", type=float, default=0.037)
    return parser.parse_args()


def main():
    args = parse_args()
    if not math.isfinite(args.annual_rate):
        raise ValueError("annual rate must be finite.")
    rows = fetch_lse_calls(max_dte=args.max_dte, limit=args.limit)
    audit_path, audit = write_local_dataset(
        rows, args.output_dir, annual_rate=args.annual_rate
    )
    print(f"LSE dataset build completed: {audit['n_rows_total']} chain rows")
    print(f"Calibration sample: {audit['n_rows_chebyshev']} rows")
    print(f"Local audit: {audit_path}")


if __name__ == "__main__":
    main()
