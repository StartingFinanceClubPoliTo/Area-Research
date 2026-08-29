"""Regression tests for the Bates and Bates--Hawkes restructuring."""

import numpy as np
import pandas as pd
import pytest

from Bates import Bates
from BatesHawkes import BatesHawkes
from calibration_core import CalibrationReport, OptionSurface, feller_feasible_population


SPOT = 100.0
MATURITY = 0.5
RATE = 0.03
BATES_PARAMS = (0.04, 1.5, 0.04, 0.30, -0.60, 0.50, -0.08, 0.13)
STRIKES = np.array([80.0, 90.0, 100.0, 110.0, 120.0])
BASELINE_PRICES = np.array(
    [
        21.950267328403807,
        13.583514697138359,
        6.914920319261018,
        2.6491543252818506,
        0.7385742898526466,
    ]
)


def market_frame():
    return pd.DataFrame(
        {
            "K": STRIKES,
            "T": MATURITY,
            "rate": RATE,
            "price": BASELINE_PRICES,
            "vega": 20.0,
        }
    )


def test_scalar_bates_prices_preserve_frozen_baseline():
    prices = np.array(
        [
            Bates.bates_price_fast(SPOT, strike, MATURITY, *BATES_PARAMS, RATE)
            for strike in STRIKES
        ]
    )
    np.testing.assert_allclose(prices, BASELINE_PRICES, rtol=0.0, atol=2e-10)


def test_batched_cos_agrees_with_reference_quadrature():
    prices = Bates.bates_prices_cos(
        SPOT, STRIKES, MATURITY, *BATES_PARAMS, RATE, N=256
    )
    np.testing.assert_allclose(prices, BASELINE_PRICES, rtol=0.0, atol=2e-7)


def test_prepared_surface_preserves_objective_value():
    frame = market_frame()
    surface = OptionSurface.from_frame(frame)
    raw = Bates.bates_objective(BATES_PARAMS, frame, SPOT)
    prepared = Bates.bates_objective(BATES_PARAMS, surface, SPOT)
    assert prepared == pytest.approx(raw, abs=1e-18)
    assert surface.size == len(frame)
    assert len(surface.slices) == 1


def test_surface_rejects_missing_or_invalid_inputs():
    with pytest.raises(ValueError, match="missing required columns"):
        OptionSurface.from_frame(market_frame().drop(columns=["vega"]))
    invalid = market_frame()
    invalid.loc[0, "K"] = 0.0
    with pytest.raises(ValueError, match="strikes must be positive"):
        OptionSurface.from_frame(invalid)


def test_proxy_reduces_exactly_to_bates_at_effective_intensity():
    lambda0, alpha, beta = 0.25, 0.5, 1.0
    proxy = BatesHawkes.price_proxy_fast(
        SPOT,
        100.0,
        MATURITY,
        *BATES_PARAMS[:5],
        lambda0,
        alpha,
        beta,
        *BATES_PARAMS[6:],
        RATE,
    )
    bates = Bates.bates_price_fast(SPOT, 100.0, MATURITY, *BATES_PARAMS, RATE)
    assert proxy == pytest.approx(bates, abs=1e-12)


def test_proxy_objective_rejects_feller_violation():
    invalid = (0.04, 1.0, 0.04, 1.0, -0.6, 0.25, 0.5, 1.0, -0.08, 0.13)
    assert (
        BatesHawkes.bates_hawkes_proxy_objective(
            invalid, market_frame(), SPOT
        )
        >= BatesHawkes.INVALID_OBJECTIVE
    )


def test_calibration_report_is_serialisable_and_numpy_compatible():
    report = CalibrationReport(
        model="Bates",
        parameter_names=Bates.PARAMETER_NAMES,
        values=BATES_PARAMS,
        objective=0.0,
        global_objective=1.0,
        success=True,
        message="ok",
        evaluations=10,
        iterations=2,
        elapsed_seconds=0.5,
    )
    np.testing.assert_allclose(report.x, BATES_PARAMS)
    assert report.as_dict()["parameters"]["lambd"] == BATES_PARAMS[5]


def test_global_initial_population_is_feller_feasible_and_reproducible():
    first = feller_feasible_population(Bates.BOUNDS, popsize=3, seed=17)
    second = feller_feasible_population(Bates.BOUNDS, popsize=3, seed=17)
    np.testing.assert_allclose(first, second)
    assert np.all(2.0 * first[:, 1] * first[:, 2] - first[:, 3] ** 2 >= 0.0)
