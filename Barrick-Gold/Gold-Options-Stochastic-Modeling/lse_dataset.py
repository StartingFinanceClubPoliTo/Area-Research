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

from BnS import BnS
from Sampling import Sampling


CHAIN_COLUMNS = (
    "expiry", "expiry_ymd", "T", "option_type", "K", "contract_symbol",
    "underlying", "last_price_lse", "implied_vol", "delta", "gamma",
    "vega_lse", "theta", "underlying_price", "volume", "source_updated_at",
    "source_age_hours",
)
CALIBRATION_COLUMNS = (
    "expiry", "T", "K", "price", "rate", "implied_vol", "vega",
    "source_updated_at", "source_age_hours", "price_method",
)


def _numeric(frame, column):
    if column not in frame:
        return pd.Series(np.nan, index=frame.index, dtype=float)
    return pd.to_numeric(frame[column], errors="coerce")


def normalise_lse_chain(rows):
    """Map LSE rows to a source-native schema and one snapshot timestamp.

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
        updated_at = pd.Series(snapshot, index=source.index, dtype="datetime64[ns, UTC]")
    if pd.isna(snapshot):
        raise ValueError("Cannot infer a coherent LSE snapshot timestamp.")
    snapshot_date = snapshot.normalize()
    maturity_days = (expiry - snapshot_date).dt.total_seconds() / 86400.0
    last_price = _numeric(source, "last_price")
    contract_type = source["contract_type"].astype(str).str.lower()
    right = contract_type.map({"call": "C", "c": "C", "put": "P", "p": "P"})

    chain = pd.DataFrame(index=source.index)
    chain["expiry"] = expiry.dt.strftime("%Y-%m-%d")
    chain["expiry_ymd"] = expiry.dt.strftime("%Y%m%d")
    chain["T"] = maturity_days / 365.25
    chain["option_type"] = right
    chain["K"] = _numeric(source, "strike")
    chain["contract_symbol"] = source["ticker"].astype(str)
    chain["underlying"] = source["underlying"].astype(str)
    chain["last_price_lse"] = last_price
    chain["implied_vol"] = _numeric(source, "iv")
    chain["delta"] = _numeric(source, "delta")
    chain["gamma"] = _numeric(source, "gamma")
    chain["vega_lse"] = _numeric(source, "vega")
    chain["theta"] = _numeric(source, "theta")
    chain["underlying_price"] = _numeric(source, "underlying_price")
    chain["volume"] = _numeric(source, "volume_today")
    chain["source_updated_at"] = updated_at.dt.strftime("%Y-%m-%dT%H:%M:%S%z")
    chain["source_age_hours"] = (
        (snapshot - updated_at).dt.total_seconds() / 3600.0
    )
    chain = chain.loc[chain["T"].gt(0.0), CHAIN_COLUMNS]
    chain = chain.sort_values(["T", "K"]).reset_index(drop=True)
    latest_spot = _numeric(source, "underlying_price").loc[latest_mask]
    latest_spot = latest_spot.loc[latest_spot.gt(0.0)].median()
    if not np.isfinite(latest_spot):
        latest_spot = _numeric(source, "underlying_price").dropna().median()
    chain.attrs["as_of_utc"] = snapshot.isoformat()
    chain.attrs["snapshot_spot"] = float(latest_spot)
    return chain


def build_calibration_sample(
    chain,
    annual_rate=0.037,
    moneyness_window=0.20,
    min_dte=60,
    min_price=1.0,
    min_points_per_expiry=5,
    n_maturities=8,
    n_strikes=8,
    max_source_age_days=7,
):
    """Create a coherent IV-derived sample for Bates/Heston calibration.

    LSE ``last_price`` can refer to an older trade than the current underlying
    snapshot.  It is retained in the local chain for audit only.  Calibration
    prices are rebuilt from LSE implied volatility with the protected
    Black--Scholes implementation, one spot, one maturity clock, and the
    explicitly supplied rate.  This preserves the LSE volatility surface while
    enforcing call-price bounds on a coherent snapshot.
    """
    calls = chain.loc[chain["option_type"] == "C"].copy()
    spot = float(chain.attrs.get("snapshot_spot", np.nan))
    if not np.isfinite(spot):
        spot_values = calls["underlying_price"].dropna()
        spot = float(spot_values.median()) if not spot_values.empty else np.nan
    if not np.isfinite(spot) or spot <= 0.0:
        raise ValueError("LSE response does not contain a valid underlying price.")
    lower_strike = spot * (1.0 - moneyness_window)
    upper_strike = spot * (1.0 + moneyness_window)

    calls = calls.loc[
        calls["T"].ge(min_dte / 365.25)
        & calls["K"].between(lower_strike, upper_strike)
        & calls["implied_vol"].gt(0.0)
        & calls["implied_vol"].lt(5.0)
        & calls["source_age_hours"].le(float(max_source_age_days) * 24.0)
    ].copy()
    calls = calls.dropna(
        subset=["expiry", "T", "K", "implied_vol", "source_updated_at"]
    )
    calls = calls.drop_duplicates(["expiry", "K"], keep="first")
    calls["rate"] = float(annual_rate)
    calls["price"] = [
        BnS.bs_call_price(spot, row.K, row.T, annual_rate, row.implied_vol)
        for row in calls.itertuples(index=False)
    ]
    calls["vega"] = [
        BnS.calculate_bs_vega(
            spot, row.K, row.T, annual_rate, 0.0, row.implied_vol
        )
        for row in calls.itertuples(index=False)
    ]
    calls["price_method"] = "LSE_IV_to_BS_price"
    calls = calls.loc[calls["price"].gt(float(min_price))].copy()
    counts = calls.groupby("expiry").size()
    calls = calls.loc[
        calls["expiry"].isin(counts[counts >= min_points_per_expiry].index)
    ]
    if calls["T"].nunique() < 2 or calls["K"].nunique() < 2:
        raise ValueError("Too few valid LSE calls remain for two-dimensional sampling.")

    calibration = calls.loc[:, CALIBRATION_COLUMNS]
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
    chain = normalise_lse_chain(rows)
    calibration, sample, spot = build_calibration_sample(
        chain, annual_rate=annual_rate
    )

    chain_path = output_dir / "gld_lse_chain.csv"
    full_path = output_dir / "gld_lse_calibration_full.csv"
    sample_path = output_dir / "gld_lse_calibration_chebyshev.csv"
    metadata_path = output_dir / "gld_lse_meta.json"
    audit_path = output_dir / "lse_build_audit.json"
    chain.to_csv(chain_path, index=False)
    calibration.to_csv(full_path, index=False)
    sample.to_csv(sample_path, index=False)

    as_of = chain.attrs.get("as_of_utc", datetime.now(timezone.utc).isoformat())
    metadata = {
        "underlying_symbol": "GLD",
        "S0": spot,
        "as_of_utc": as_of,
        "source": "London Strategic Edge /options/chain",
        "snapshot_kind": "current_lse_option_chain",
        "risk_free_rate_assumption": float(annual_rate),
        "dividend_yield_assumption": 0.0,
        "calibration_price_method": "Black-Scholes price implied by LSE IV",
        "maximum_source_age_days": 7,
        "n_rows_total": int(len(chain)),
        "n_rows_calibration_eligible": int(len(calibration)),
        "n_rows_chebyshev": int(len(sample)),
        "redistribution": "prohibited; local-only outputs ignored by Git",
    }
    metadata_path.write_text(json.dumps(metadata, indent=2), encoding="utf-8")

    audit = {
        "success": True,
        "schema_matches_lse_chain": list(chain.columns) == list(CHAIN_COLUMNS),
        "schema_matches_calibration": list(sample.columns) == list(CALIBRATION_COLUMNS),
        "files": [path.name for path in (chain_path, full_path, sample_path, metadata_path)],
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
