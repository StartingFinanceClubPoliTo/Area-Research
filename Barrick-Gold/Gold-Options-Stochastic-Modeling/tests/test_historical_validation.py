import numpy as np
import pandas as pd

from historical_validation import (
    attach_prevailing_mean,
    cumulative_loss_analysis,
    latest_curve_without_lookahead,
    oos_r2_metrics,
    fixed_surface_nodes,
    weekly_last_dates,
)
from online_validation import _forecast_parameters


def test_latest_curve_never_uses_a_future_rate():
    rates = pd.DataFrame({
        "date": ["2026-01-30", "2026-01-30", "2026-02-13", "2026-02-13"],
        "symbol": ["US1M", "US1Y", "US1M", "US1Y"],
        "maturity_years": [1 / 12, 1.0, 1 / 12, 1.0],
        "continuous_rate": [0.03, 0.04, 0.031, 0.041],
    })
    curve, curve_date = latest_curve_without_lookahead(rates, "2026-02-10")
    assert curve_date == pd.Timestamp("2026-01-30", tz="UTC")
    assert set(curve["symbol"]) == {"US1M", "US1Y"}


def test_prevailing_mean_updates_only_after_each_week():
    panel = pd.DataFrame({
        "date": pd.to_datetime([
            "2026-02-20", "2026-02-20", "2026-02-27", "2026-02-27"
        ], utc=True),
        "implied_vol": [0.30, 0.40, 0.50, 0.60],
    })
    result = attach_prevailing_mean(panel, [0.10, 0.20])
    assert np.allclose(result.loc[:1, "mean_forecast"], 0.15)
    assert np.allclose(result.loc[2:, "mean_forecast"], 0.25)


def test_oos_r2_and_welch_goyal_signs_favor_better_model():
    dates = pd.to_datetime(["2026-03-06", "2026-03-13", "2026-03-20"], utc=True)
    frame = pd.DataFrame({
        "date": dates,
        "implied_vol": [0.20, 0.30, 0.40],
        "mean_forecast": [0.25, 0.25, 0.25],
        "good": [0.21, 0.29, 0.39],
        "bad": [0.10, 0.10, 0.10],
    })
    common, metrics, pairwise = oos_r2_metrics(
        frame, {"Good": "good", "Bad": "bad"}
    )
    good_r2 = metrics.loc[metrics["model"].eq("Good"), "r2_oos_vs_prevailing_mean"].iloc[0]
    good_vs_bad = pairwise.loc[
        pairwise["model"].eq("Good") & pairwise["comparator"].eq("Bad"),
        "r2_oos_pairwise",
    ].iloc[0]
    assert good_r2 > 0.0
    assert good_vs_bad > 0.0
    _, cumulative, tests = cumulative_loss_analysis(
        common, {"Good": "good", "Bad": "bad"}
    )
    assert cumulative["Good vs Prevailing mean"].iloc[-1] > 0.0
    assert cumulative["Good vs Bad"].iloc[-1] > 0.0
    comparison = tests.loc[
        tests["model"].eq("Good") & tests["comparator"].eq("Bad")
    ].iloc[0]
    assert comparison["mean_loss_difference"] > 0.0


def test_weekly_last_dates_uses_only_available_observations():
    dates = pd.to_datetime([
        "2026-02-09", "2026-02-12", "2026-02-16", "2026-02-20"
    ], utc=True)
    selected = weekly_last_dates(dates, "2026-02-09", "2026-02-20")
    assert selected == [
        pd.Timestamp("2026-02-12", tz="UTC"),
        pd.Timestamp("2026-02-20", tz="UTC"),
    ]


def test_fixed_surface_nodes_are_ex_ante_and_unique():
    nodes = fixed_surface_nodes(5, 5, (0.95, 1.05), (90, 270))
    assert len(nodes) == 25
    assert nodes["node_id"].is_unique
    assert nodes["moneyness"].between(0.95, 1.05).all()
    assert nodes["T"].between(90 / 365.25, 270 / 365.25).all()


def test_one_step_state_forecast_uses_only_fitted_state_laws():
    heston = np.asarray([0.09, 2.0, 0.04, 0.30, -0.2])
    predicted = _forecast_parameters("Heston", heston, 1)
    expected_v0 = 0.04 + (0.09 - 0.04) * np.exp(-2.0 / 365.25)
    assert np.isclose(predicted[0], expected_v0)
    assert np.allclose(predicted[1:], heston[1:])

    hawkes = np.asarray([
        0.09, 2.0, 0.04, 0.30, -0.2, 0.8, 0.3, 0.25, 4.0, -0.05, 0.10
    ])
    predicted = _forecast_parameters("Full Bates-Hawkes", hawkes, 1)
    stationary = 0.3 / (1.0 - 0.25)
    expected_lambda = stationary + (0.8 - stationary) * np.exp(
        -4.0 * (1.0 - 0.25) / 365.25
    )
    assert np.isclose(predicted[5], expected_lambda)


def test_oos_r2_can_use_random_walk_benchmark():
    frame = pd.DataFrame({
        "date": pd.to_datetime(["2026-02-11", "2026-02-12"], utc=True),
        "implied_vol": [0.20, 0.30],
        "random_walk": [0.19, 0.29],
        "model": [0.22, 0.32],
    })
    _, metrics, _ = oos_r2_metrics(
        frame,
        {"Model": "model"},
        benchmark_column="random_walk",
        benchmark_name="Random walk",
    )
    assert metrics.loc[0, "benchmark"] == "Random walk"
    assert metrics.loc[0, "r2_oos_vs_benchmark"] < 0.0
