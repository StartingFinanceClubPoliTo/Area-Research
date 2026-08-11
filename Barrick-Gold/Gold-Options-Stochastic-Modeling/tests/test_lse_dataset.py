"""Pure transformation tests; no API key or network access required."""

import pandas as pd

from lse_dataset import CALIBRATION_COLUMNS, WIDE_COLUMNS, build_calibration_sample, normalise_lse_chain


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


def test_lse_mapping_matches_historical_wide_schema():
    wide = normalise_lse_chain(sample_rows())
    assert tuple(wide.columns) == WIDE_COLUMNS
    assert set(wide["right"]) == {"C"}
    assert set(wide["price_source"]) == {"lse_last_price"}
    assert wide.attrs["as_of_utc"].startswith("2026-08-11")
    assert wide["T"].min() > 0.0


def test_lse_chebyshev_sample_matches_calibration_contract():
    wide = normalise_lse_chain(sample_rows())
    full, sampled, spot = build_calibration_sample(
        wide, n_maturities=2, n_strikes=4
    )
    assert tuple(full.columns) == CALIBRATION_COLUMNS
    assert tuple(sampled.columns) == CALIBRATION_COLUMNS
    assert len(sampled) == 8
    assert spot == 100.0
    assert isinstance(sampled, pd.DataFrame)
