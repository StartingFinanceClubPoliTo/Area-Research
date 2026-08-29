import json

import numpy as np
import pandas as pd

from barrick_unified.empirical_figures import FIGURE_NAMES, generate_empirical_figures, load_current_panel


def _rows(dates, values):
    return [
        {"timestamp": date.isoformat(), "close": float(value)}
        for date, value in zip(dates, values)
    ]


def test_current_panel_and_all_figure_outputs(tmp_path):
    dates = pd.date_range("2024-01-02", periods=380, freq="B", tz="UTC")
    time = np.arange(len(dates), dtype=float)
    payload = {
        "rows": {
            "market_candles": {
                "GLD": _rows(dates, 180.0 * np.exp(0.0003 * time + 0.02 * np.sin(time / 17.0))),
                "SLV": _rows(dates, 22.0 * np.exp(0.0002 * time + 0.03 * np.cos(time / 19.0))),
                "SPY": _rows(dates, 430.0 * np.exp(0.0004 * time + 0.015 * np.sin(time / 13.0))),
            },
            "fx_candles": {
                "EUR/USD": _rows(dates, 1.08 + 0.01 * np.sin(time / 31.0)),
                "USD/JPY": _rows(dates, 145.0 + 2.0 * np.cos(time / 29.0)),
                "GBP/USD": _rows(dates, 1.26 + 0.01 * np.sin(time / 37.0)),
                "USD/CAD": _rows(dates, 1.34 + 0.01 * np.cos(time / 23.0)),
                "USD/SEK": _rows(dates, 10.4 + 0.1 * np.sin(time / 41.0)),
                "USD/CHF": _rows(dates, 0.88 + 0.01 * np.cos(time / 43.0)),
            },
            "us10y_yields": _rows(dates, 4.0 + 0.15 * np.sin(time / 47.0)),
        }
    }
    raw_path = tmp_path / "empirical_figure_inputs.json"
    raw_path.write_text(json.dumps(payload), encoding="utf-8")
    levels, returns = load_current_panel(raw_path)
    assert len(levels) == len(returns) == len(dates) - 1
    assert set(returns) == {"gold", "silver", "sp500", "dxy", "ust10y_change"}
    assert np.isfinite(returns.to_numpy()).all()

    output = tmp_path / "figures"
    manifest = generate_empirical_figures(raw_path, output, rolling_window=63)
    assert manifest["figure_count"] == len(FIGURE_NAMES) == 19
    assert manifest["sample"]["observations"] == len(dates) - 1
    assert manifest["garch"]["gold"]["alpha_plus_beta"] < 1.0
    assert manifest["garch"]["silver"]["alpha_plus_beta"] < 1.0
    assert {path.name for path in output.glob("*.png")} == set(FIGURE_NAMES)
    assert all((output / name).stat().st_size > 10_000 for name in FIGURE_NAMES)
