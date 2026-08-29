"""Build a local GLD option dataset from the London Strategic Edge API.

Retrieved LSE data is intentionally written below ``Data/lse_local`` (ignored
by Git). The committed artifact is this reproducible transformation, not a
redistributed market-data snapshot.
"""

import argparse
import json
import os
from datetime import datetime, timedelta, timezone
from pathlib import Path

import numpy as np
import pandas as pd

from BnS import BnS
from Sampling import Sampling
from nss_curve import PARAMETER_NAMES, fit_nss_curve, nss_zero_rates


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
YIELD_CURVE_COLUMNS = (
    "symbol", "date", "maturity", "maturity_days", "maturity_years",
    "par_yield_pct", "observed_continuous_rate", "continuous_rate",
    "nss_residual_bp", *PARAMETER_NAMES, "source_fetched_at",
)
YIELD_HISTORY_COLUMNS = (
    "symbol", "date", "maturity", "maturity_days", "maturity_years",
    "par_yield_pct", "continuous_rate", "source_fetched_at",
)
RETURN_COLUMNS = (
    "timestamp", "symbol", "open", "high", "low", "close", "volume",
    "log_return", "simple_return",
)
US_TREASURY_TENORS = (
    "US1M", "US2M", "US3M", "US6M", "US1Y", "US2Y", "US3Y", "US5Y",
)
NSS_TREASURY_TENORS = (
    *US_TREASURY_TENORS, "US7Y", "US10Y", "US20Y", "US30Y",
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


def normalise_lse_yield_curve(
    rows,
    required_symbols=US_TREASURY_TENORS,
    optional_symbols=NSS_TREASURY_TENORS,
):
    """Build one coherent USD Treasury curve from LSE bond-yield rows.

    The latest date shared by all required tenors is selected, preventing a
    visually smooth but internally asynchronous curve. LSE quotes par yields in
    percent; ``continuous_rate`` is the explicitly documented log(1+y) proxy
    used by the option-pricing functions, which expect continuously compounded
    annual rates.
    """
    source = pd.DataFrame(rows)
    required = {"symbol", "date", "maturity", "maturity_days", "close"}
    missing = sorted(required.difference(source.columns))
    if missing:
        raise ValueError(f"LSE bond-yield response is missing columns: {missing}")
    eligible_symbols = tuple(dict.fromkeys((*required_symbols, *optional_symbols)))
    source = source.loc[source["symbol"].isin(eligible_symbols)].copy()
    source["date"] = pd.to_datetime(source["date"], errors="coerce")
    source["maturity_days"] = _numeric(source, "maturity_days")
    source["close"] = _numeric(source, "close")
    source = source.dropna(subset=["date", "maturity_days", "close"])
    dates_by_symbol = [
        set(source.loc[source["symbol"] == symbol, "date"])
        for symbol in required_symbols
    ]
    common_dates = set.intersection(*dates_by_symbol) if dates_by_symbol else set()
    if not common_dates:
        raise ValueError("LSE Treasury tenors have no common observation date.")
    curve_date = max(common_dates)
    curve = source.loc[
        source["date"].eq(curve_date) & source["symbol"].isin(eligible_symbols)
    ].copy()
    curve = curve.drop_duplicates("symbol", keep="last")
    if not set(required_symbols).issubset(set(curve["symbol"])):
        raise ValueError("The selected LSE Treasury date is missing required tenors.")
    curve["maturity_years"] = curve["maturity_days"] / 365.25
    curve = curve.sort_values("maturity_years").reset_index(drop=True)
    curve["par_yield_pct"] = curve["close"]
    curve["observed_continuous_rate"] = np.log1p(curve["par_yield_pct"] / 100.0)
    fit = fit_nss_curve(curve["maturity_years"], curve["observed_continuous_rate"])
    curve["continuous_rate"] = fit.fitted_rates
    curve["nss_residual_bp"] = fit.residuals * 10_000.0
    for name, value in fit.parameters.items():
        curve[name] = float(value)
    curve["date"] = curve["date"].dt.strftime("%Y-%m-%d")
    if "fetched_at" in curve:
        curve["source_fetched_at"] = curve["fetched_at"].astype(str)
    else:
        curve["source_fetched_at"] = ""
    curve = curve.loc[:, YIELD_CURVE_COLUMNS]
    return curve.sort_values("maturity_years").reset_index(drop=True)


def normalise_lse_yield_history(rows, required_symbols=US_TREASURY_TENORS):
    """Return complete historical Treasury curves on common observation dates.

    Keeping only dates shared by every tenor prevents a later observation from
    leaking into one end of an otherwise older curve.  A backtest can therefore
    select the latest complete curve whose date is no later than the option date.
    """
    source = pd.DataFrame(rows)
    required = {"symbol", "date", "maturity", "maturity_days", "close"}
    missing = sorted(required.difference(source.columns))
    if missing:
        raise ValueError(f"LSE bond-yield response is missing columns: {missing}")
    source = source.loc[source["symbol"].isin(required_symbols)].copy()
    source["date"] = pd.to_datetime(source["date"], errors="coerce")
    source["maturity_days"] = _numeric(source, "maturity_days")
    source["close"] = _numeric(source, "close")
    source = source.dropna(subset=["date", "maturity_days", "close"])
    source = source.drop_duplicates(["date", "symbol"], keep="last")
    counts = source.groupby("date")["symbol"].nunique()
    complete_dates = counts[counts == len(required_symbols)].index
    history = source.loc[source["date"].isin(complete_dates)].copy()
    if history.empty:
        raise ValueError("LSE Treasury history has no complete common-date curves.")
    history["maturity_years"] = history["maturity_days"] / 365.25
    history["par_yield_pct"] = history["close"]
    history["continuous_rate"] = np.log1p(history["par_yield_pct"] / 100.0)
    if "fetched_at" in history:
        history["source_fetched_at"] = history["fetched_at"].astype(str)
    else:
        history["source_fetched_at"] = ""
    history["date"] = history["date"].dt.strftime("%Y-%m-%d")
    history = history.loc[:, YIELD_HISTORY_COLUMNS]
    return history.sort_values(["date", "maturity_years"]).reset_index(drop=True)


def interpolate_zero_rates(maturities, rate_curve):
    """Evaluate the fitted NSS curve, with legacy linear interpolation fallback."""
    maturities = np.asarray(maturities, dtype=float)
    if np.any(~np.isfinite(maturities)) or np.any(maturities <= 0.0):
        raise ValueError("Option maturities must be positive and finite.")
    curve = pd.DataFrame(rate_curve).sort_values("maturity_years")
    if len(curve) < 2:
        raise ValueError("At least two LSE Treasury tenors are required.")
    if all(name in curve.columns for name in PARAMETER_NAMES):
        parameters = {name: float(curve[name].iloc[0]) for name in PARAMETER_NAMES}
        return nss_zero_rates(maturities, parameters)
    tenors = curve["maturity_years"].to_numpy(dtype=float)
    rates = curve["continuous_rate"].to_numpy(dtype=float)
    if np.any(np.diff(tenors) <= 0.0) or np.any(~np.isfinite(rates)):
        raise ValueError("Invalid LSE Treasury curve.")
    return np.interp(maturities, tenors, rates, left=rates[0], right=rates[-1])


def normalise_lse_history(rows):
    """Map LSE daily GLD candles to a return series kept in local-only data."""
    history = pd.DataFrame(rows)
    required = {"timestamp", "symbol", "open", "high", "low", "close", "volume"}
    missing = sorted(required.difference(history.columns))
    if missing:
        raise ValueError(f"LSE candle response is missing columns: {missing}")
    history["timestamp"] = pd.to_datetime(
        history["timestamp"], errors="coerce", utc=True
    )
    for column in ("open", "high", "low", "close", "volume"):
        history[column] = _numeric(history, column)
    history = history.dropna(subset=["timestamp", "close"])
    history = history.loc[history["close"].gt(0.0)].copy()
    history = history.sort_values("timestamp").drop_duplicates("timestamp", keep="last")
    history["log_return"] = np.log(history["close"]).diff()
    history["simple_return"] = history["close"].pct_change(fill_method=None)
    history["timestamp"] = history["timestamp"].dt.strftime("%Y-%m-%dT%H:%M:%SZ")
    return history.loc[:, RETURN_COLUMNS].reset_index(drop=True)


def build_calibration_sample(
    chain,
    rate_curve,
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
    LSE Treasury term structure. This preserves the LSE volatility surface while
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
    calls["rate"] = interpolate_zero_rates(calls["T"], rate_curve)
    calls["price"] = [
        BnS.bs_call_price(spot, row.K, row.T, row.rate, row.implied_vol)
        for row in calls.itertuples(index=False)
    ]
    calls["vega"] = [
        BnS.calculate_bs_vega(
            spot, row.K, row.T, row.rate, 0.0, row.implied_vol
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


def _lse_client():
    """Return the official SDK client without ever logging the configured key."""
    if not os.environ.get("LSE_API_KEY"):
        raise RuntimeError("LSE_API_KEY is not configured in the environment.")
    from lse import LSE

    return LSE()


def fetch_lse_calls(max_dte=1000, limit=5000):
    """Fetch the current GLD call chain from LSE."""
    return _lse_client().options(
        "GLD", type="call", max_dte=int(max_dte), limit=int(limit)
    )


def fetch_lse_yields(lookback_days=120):
    """Fetch enough history to locate one date shared by all curve tenors."""
    client = _lse_client()
    start = (datetime.now(timezone.utc) - timedelta(days=lookback_days)).date().isoformat()
    rows = []
    for symbol in NSS_TREASURY_TENORS:
        rows.extend(client.bond_yields(symbol, start=start, order="asc", limit=5000))
    return rows


def fetch_lse_history(start="2021-01-01", end=None, limit=5000):
    """Fetch daily GLD candles used for empirical return diagnostics."""
    return _lse_client().candles(
        "GLD", "1d", start=start, end=end, limit=int(limit), order="asc"
    )


def fetch_lse_historical_yields(start, end=None):
    """Fetch complete tenor histories used by the no-look-ahead backtest."""
    client = _lse_client()
    rows = []
    for symbol in US_TREASURY_TENORS:
        rows.extend(
            client.bond_yields(
                symbol, start=str(start), end=end, order="asc", limit=5000
            )
        )
    return rows


def fetch_lse_option_history(
    start,
    end,
    output_dir="Data/lse_local",
    timeframe="1d",
    reuse_existing=True,
):
    """Export historical GLD option bars to a local-only Parquet file.

    A recent cache is reused only when it covers the requested start and comes
    within five calendar days of the requested end (weekends and vault lag).
    The function deliberately returns a path, so raw licensed rows never enter
    publication artifacts.
    """
    output_dir = Path(output_dir)
    output_dir.mkdir(parents=True, exist_ok=True)
    path = output_dir / f"options_GLD_{timeframe}.parquet"
    if reuse_existing and path.exists():
        try:
            timestamps = pd.read_parquet(path, columns=["ts"])["ts"]
            first = pd.to_datetime(timestamps.min(), utc=True).normalize()
            last = pd.to_datetime(timestamps.max(), utc=True).normalize()
            requested_start = pd.Timestamp(start, tz="UTC").normalize()
            requested_end = pd.Timestamp(end, tz="UTC").normalize()
            if first <= requested_start + pd.Timedelta(days=3) and last >= requested_end - pd.Timedelta(days=5):
                return path
        except (ImportError, OSError, ValueError, KeyError):
            pass
    exported = _lse_client().history(
        "GLD",
        dataset="options",
        timeframe=timeframe,
        start=str(start),
        end=str(end),
        dest=str(output_dir),
        dataframe=False,
    )
    return Path(exported)


def write_local_historical_inputs(yield_rows, output_dir="Data/lse_local"):
    """Store only the local historical rate panel and an audit manifest."""
    output_dir = Path(output_dir)
    output_dir.mkdir(parents=True, exist_ok=True)
    history = normalise_lse_yield_history(yield_rows)
    rate_path = output_dir / "usd_treasury_history.csv"
    history.to_csv(rate_path, index=False)
    audit = {
        "source": "London Strategic Edge options vault and /ref/bond_yields",
        "rate_history_start": str(history["date"].iloc[0]),
        "rate_history_end": str(history["date"].iloc[-1]),
        "complete_curve_dates": int(history["date"].nunique()),
        "tenors": int(history["symbol"].nunique()),
        "look_ahead_policy": "latest complete Treasury curve dated no later than each option observation",
        "redistribution": "prohibited; local-only outputs ignored by Git",
    }
    audit_path = output_dir / "historical_input_audit.json"
    audit_path.write_text(json.dumps(audit, indent=2), encoding="utf-8")
    return rate_path, audit_path, audit


def write_local_dataset(rows, yield_rows, history_rows, output_dir):
    """Write local-only wide, filtered, sampled, metadata, and audit outputs."""
    output_dir = Path(output_dir)
    output_dir.mkdir(parents=True, exist_ok=True)
    chain = normalise_lse_chain(rows)
    rate_curve = normalise_lse_yield_curve(yield_rows)
    history = normalise_lse_history(history_rows)
    calibration, sample, spot = build_calibration_sample(
        chain, rate_curve=rate_curve
    )

    chain_path = output_dir / "gld_lse_chain.csv"
    full_path = output_dir / "gld_lse_calibration_full.csv"
    sample_path = output_dir / "gld_lse_calibration_chebyshev.csv"
    metadata_path = output_dir / "gld_lse_meta.json"
    audit_path = output_dir / "lse_build_audit.json"
    curve_path = output_dir / "usd_treasury_curve.csv"
    nss_path = output_dir / "usd_treasury_nss_fit.json"
    history_path = output_dir / "gld_daily_history.csv"
    chain.to_csv(chain_path, index=False)
    calibration.to_csv(full_path, index=False)
    sample.to_csv(sample_path, index=False)
    rate_curve.to_csv(curve_path, index=False)
    history.to_csv(history_path, index=False)

    as_of = chain.attrs.get("as_of_utc", datetime.now(timezone.utc).isoformat())
    nss_parameters = {
        name: float(rate_curve[name].iloc[0]) for name in PARAMETER_NAMES
    }
    nss_rmse_bp = float(
        np.sqrt(np.mean(np.square(rate_curve["nss_residual_bp"].to_numpy(dtype=float))))
    )
    nss_payload = {
        "method": "Nelson-Siegel-Svensson nonlinear least squares",
        "source": "London Strategic Edge /ref/bond_yields",
        "curve_date": str(rate_curve["date"].iloc[0]),
        "available_symbols": rate_curve["symbol"].astype(str).tolist(),
        "requested_symbols": list(NSS_TREASURY_TENORS),
        "missing_symbols_on_curve_date": sorted(
            set(NSS_TREASURY_TENORS).difference(rate_curve["symbol"].astype(str))
        ),
        "parameters": nss_parameters,
        "observations": int(len(rate_curve)),
        "rmse_bp": nss_rmse_bp,
        "max_abs_error_bp": float(rate_curve["nss_residual_bp"].abs().max()),
        "rate_definition": "continuous proxy log(1 + Treasury par yield)",
        "use": "common risk-free zero curve for Team 8 calibration and unified simulations",
        "legacy_team4_nss_used": False,
    }
    nss_path.write_text(json.dumps(nss_payload, indent=2), encoding="utf-8")

    metadata = {
        "underlying_symbol": "GLD",
        "S0": spot,
        "as_of_utc": as_of,
        "source": "London Strategic Edge /options/chain, /ref/bond_yields, /candles",
        "snapshot_kind": "current_lse_option_chain",
        "risk_free_rate_source": "LSE US Treasury par-yield curve",
        "risk_free_rate_curve_date": str(rate_curve["date"].iloc[0]),
        "risk_free_rate_conversion": "continuous proxy log(1 + par_yield_decimal)",
        "risk_free_rate_interpolation": "current LSE Nelson-Siegel-Svensson fit",
        "risk_free_rate_model": "Nelson-Siegel-Svensson",
        "risk_free_rate_nss_parameters": nss_parameters,
        "risk_free_rate_nss_rmse_bp": nss_rmse_bp,
        "risk_free_rate_missing_requested_tenors": nss_payload["missing_symbols_on_curve_date"],
        "risk_free_rate_min": float(rate_curve["continuous_rate"].min()),
        "risk_free_rate_max": float(rate_curve["continuous_rate"].max()),
        "dividend_yield_assumption": 0.0,
        "calibration_price_method": "Black-Scholes price implied by LSE IV",
        "maximum_source_age_days": 7,
        "n_rows_total": int(len(chain)),
        "n_rows_calibration_eligible": int(len(calibration)),
        "n_rows_chebyshev": int(len(sample)),
        "n_treasury_tenors": int(len(rate_curve)),
        "history_start": str(history["timestamp"].iloc[0]),
        "history_end": str(history["timestamp"].iloc[-1]),
        "history_observations": int(history["log_return"].notna().sum()),
        "redistribution": "prohibited; local-only outputs ignored by Git",
    }
    metadata_path.write_text(json.dumps(metadata, indent=2), encoding="utf-8")

    audit = {
        "success": True,
        "schema_matches_lse_chain": list(chain.columns) == list(CHAIN_COLUMNS),
        "schema_matches_calibration": list(sample.columns) == list(CALIBRATION_COLUMNS),
        "files": [
            path.name for path in (
                chain_path, full_path, sample_path, curve_path, nss_path, history_path,
                metadata_path,
            )
        ],
        **metadata,
    }
    audit_path.write_text(json.dumps(audit, indent=2), encoding="utf-8")
    return audit_path, audit


def parse_args():
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--output-dir", default="Data/lse_local")
    parser.add_argument("--max-dte", type=int, default=1000)
    parser.add_argument("--limit", type=int, default=5000)
    parser.add_argument("--history-start", default="2021-01-01")
    parser.add_argument("--yield-lookback-days", type=int, default=120)
    return parser.parse_args()


def main():
    args = parse_args()
    rows = fetch_lse_calls(max_dte=args.max_dte, limit=args.limit)
    yield_rows = fetch_lse_yields(lookback_days=args.yield_lookback_days)
    history_rows = fetch_lse_history(start=args.history_start)
    audit_path, audit = write_local_dataset(
        rows, yield_rows, history_rows, args.output_dir
    )
    print(f"LSE dataset build completed: {audit['n_rows_total']} chain rows")
    print(f"Calibration sample: {audit['n_rows_chebyshev']} rows")
    print(
        "LSE Treasury curve: "
        f"{audit['n_treasury_tenors']} tenors at {audit['risk_free_rate_curve_date']}"
    )
    print(f"GLD daily return observations: {audit['history_observations']}")
    print(f"Local audit: {audit_path}")


if __name__ == "__main__":
    main()
