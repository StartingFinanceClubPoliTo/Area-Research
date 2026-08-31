"""Pure transformation tests; no API key or network access required."""

import pandas as pd

import lse_dataset

from BnS import BnS
from lse_dataset import (
    CALIBRATION_COLUMNS,
    CHAIN_COLUMNS,
    US_TREASURY_TENORS,
    build_calibration_sample,
    interpolate_zero_rates,
    normalise_lse_chain,
    normalise_lse_history,
    normalise_lse_yield_curve,
)


def sample_rows():
    rows = []
    for dte, expiry in ((90, "2026-11-09"), (180, "2027-02-07")):
        for strike in range(80, 125, 5):
            rows.append(
                {
                    "ticker": f"GLD-{expiry}-{strike}-C",
                    "underlying": "GLD",
                    "strike": strike,
                    "expiry": expiry,
                    "contract_type": "call",
                    "last_price": max(2.0, 102.0 - strike),
                    "underlying_price": 100.0,
                    "dte": dte,
                    "iv": 0.25,
                    "delta": 0.5,
                    "gamma": 0.02,
                    "theta": -0.1,
                    "vega": 15.0,
                    "volume_today": 10,
                    "updated_at": "2026-08-11T15:00:00Z",
                }
            )
    return rows


def sample_yields():
    maturities = {
        "US1M": ("1M", 30, 3.70), "US2M": ("2M", 60, 3.75),
        "US3M": ("3M", 90, 3.80), "US6M": ("6M", 180, 3.90),
        "US1Y": ("1Y", 365, 4.00), "US2Y": ("2Y", 730, 4.10),
        "US3Y": ("3Y", 1095, 4.20), "US5Y": ("5Y", 1825, 4.30),
    }
    rows = []
    for date in ("2026-07-23", "2026-07-24"):
        for symbol, (maturity, days, close) in maturities.items():
            rows.append({
                "symbol": symbol, "date": date, "maturity": maturity,
                "maturity_days": days, "close": close,
                "fetched_at": "2026-07-26T00:49:16Z",
            })
    return rows


def test_lse_mapping_uses_source_native_schema():
    chain = normalise_lse_chain(sample_rows())
    assert tuple(chain.columns) == CHAIN_COLUMNS
    assert set(chain["option_type"]) == {"C"}
    assert chain.attrs["as_of_utc"].startswith("2026-08-11")
    assert chain["T"].min() > 0.0


def test_lse_fetch_excludes_expired_contracts(monkeypatch):
    captured = {}

    class FakeClient:
        def options(self, underlying, **kwargs):
            captured["underlying"] = underlying
            captured.update(kwargs)
            return []

    monkeypatch.setattr(lse_dataset, "_lse_client", lambda: FakeClient())
    assert lse_dataset.fetch_lse_calls(max_dte=1000, limit=5000) == []
    assert captured == {
        "underlying": "GLD",
        "type": "call",
        "min_dte": 1,
        "max_dte": 1000,
        "limit": 5000,
    }


def test_lse_chebyshev_sample_matches_calibration_contract():
    chain = normalise_lse_chain(sample_rows())
    curve = normalise_lse_yield_curve(sample_yields())
    full, sampled, spot = build_calibration_sample(
        chain, curve, n_maturities=2, n_strikes=4
    )
    assert tuple(full.columns) == CALIBRATION_COLUMNS
    assert tuple(sampled.columns) == CALIBRATION_COLUMNS
    assert len(sampled) == 8
    assert spot == 100.0
    assert isinstance(sampled, pd.DataFrame)
    assert set(sampled["price_method"]) == {"LSE_IV_to_BS_price"}
    recovered = [
        BnS.implied_vol_call(row.price, spot, row.K, row.T, row.rate)
        for row in sampled.itertuples(index=False)
    ]
    assert max(abs(a - b) for a, b in zip(recovered, sampled["implied_vol"])) < 1e-6
    lower = spot - sampled["K"] * (-sampled["rate"] * sampled["T"]).map(__import__("math").exp)
    assert (sampled["price"] >= lower.clip(lower=0.0) - 1e-10).all()
    assert sampled.groupby("T")["rate"].nunique().max() == 1
    assert sampled["rate"].nunique() == 2


def test_lse_curve_uses_latest_common_date_and_interpolates():
    rows = sample_yields()
    rows = [r for r in rows if not (r["date"] == "2026-07-24" and r["symbol"] == "US5Y")]
    curve = normalise_lse_yield_curve(rows)
    assert set(curve["symbol"]) == set(US_TREASURY_TENORS)
    assert set(curve["date"]) == {"2026-07-23"}
    rates = interpolate_zero_rates([0.25, 1.5, 4.0], curve)
    assert rates.shape == (3,)
    assert rates[0] < rates[1] < rates[2]


def test_lse_history_produces_log_and_simple_returns():
    rows = [
        {"timestamp": "2026-01-02T00:00:00Z", "symbol": "GLD", "open": 99,
         "high": 101, "low": 98, "close": 100, "volume": 10},
        {"timestamp": "2026-01-05T00:00:00Z", "symbol": "GLD", "open": 100,
         "high": 103, "low": 99, "close": 102, "volume": 12},
    ]
    history = normalise_lse_history(rows)
    assert history["log_return"].isna().sum() == 1
    assert abs(history["simple_return"].iloc[-1] - 0.02) < 1e-12
