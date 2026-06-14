"""Tests for the exponential and rough point-process calibration classes."""

import math
import os
import sys

import numpy as np

HERE = os.path.dirname(os.path.abspath(__file__))
ROOT = os.path.dirname(HERE)
if ROOT not in sys.path:
    sys.path.insert(0, ROOT)

from Hawkes import (  # noqa: E402
    ExponentialHawkesCalibration,
    Hawkes,
    RoughHawkesCalibration,
)


def test_exponential_log_likelihood_matches_closed_form():
    events = np.array([1.0, 2.0])
    horizon = 3.0
    lambda0, alpha, beta = 0.4, 0.3, 1.2
    expected_log_intensity = math.log(lambda0) + math.log(
        lambda0 + alpha * math.exp(-beta)
    )
    expected_compensator = lambda0 * horizon + (alpha / beta) * (
        2.0 - math.exp(-2.0 * beta) - math.exp(-beta)
    )
    expected = expected_log_intensity - expected_compensator
    actual = ExponentialHawkesCalibration.log_likelihood(
        events, horizon, lambda0, alpha, beta
    )
    assert abs(actual - expected) < 1e-12


def test_rough_log_likelihood_matches_closed_form():
    events = np.array([1.0, 2.0])
    horizon = 3.0
    lambda0, tail_index, cutoff = 0.4, 0.45, 0.2
    branching = 0.55
    alpha = branching * tail_index * cutoff ** tail_index
    kernel_lag_one = alpha / (1.0 + cutoff) ** (1.0 + tail_index)
    expected_log_intensity = math.log(lambda0) + math.log(lambda0 + kernel_lag_one)
    remaining = horizon - events
    expected_compensator = lambda0 * horizon + branching * np.sum(
        1.0 - (cutoff / (cutoff + remaining)) ** tail_index
    )
    expected = expected_log_intensity - expected_compensator
    actual = RoughHawkesCalibration.log_likelihood(
        events, horizon, lambda0, alpha, tail_index, cutoff
    )
    assert abs(actual - expected) < 1e-12


def test_exponential_fit_is_stationary_and_finite():
    events = ExponentialHawkesCalibration.simulate(
        lambda0=0.5, alpha=0.6, beta=1.2, horizon=80.0, seed=20260614
    )
    assert events.size > 10
    result = ExponentialHawkesCalibration.fit(events, horizon=80.0)
    assert result.success, result.message
    assert math.isfinite(result.log_likelihood)
    assert 0.0 < result.params["branching_ratio"] < 1.0
    assert result.aic < math.inf and result.bic < math.inf


def test_rough_fit_is_stationary_and_finite():
    tail_index, cutoff, branching = 0.45, 0.25, 0.50
    alpha = branching * tail_index * cutoff ** tail_index
    events = RoughHawkesCalibration.simulate(
        lambda0=0.5,
        alpha=alpha,
        tail_index=tail_index,
        cutoff=cutoff,
        horizon=80.0,
        seed=20260614,
    )
    assert events.size > 10
    result = RoughHawkesCalibration.fit(events, horizon=80.0)
    assert result.success, result.message
    assert math.isfinite(result.log_likelihood)
    assert 0.0 < result.params["branching_ratio"] < 1.0
    assert 0.0 < result.params["tail_index"] < 1.0


def test_time_rescaling_residuals_are_positive():
    events = np.array([0.5, 0.9, 1.8, 2.0, 3.4])
    exponential = ExponentialHawkesCalibration.time_rescaling_residuals(
        events, lambda0=0.6, alpha=0.4, beta=1.1
    )
    tail_index, cutoff, branching = 0.4, 0.2, 0.5
    alpha = branching * tail_index * cutoff ** tail_index
    rough = RoughHawkesCalibration.time_rescaling_residuals(
        events, lambda0=0.6, alpha=alpha, tail_index=tail_index, cutoff=cutoff
    )
    assert exponential.shape == events.shape
    assert rough.shape == events.shape
    assert np.all(exponential > 0.0)
    assert np.all(rough > 0.0)


def test_existing_hawkes_simulation_api_is_preserved():
    events = Hawkes.simulate_exponential(0.5, 0.4, 1.0, 10.0, seed=7)
    assert events.ndim == 1
    assert np.all(np.diff(events) > 0.0)
    assert np.all((events >= 0.0) & (events < 10.0))


def _run_all():
    tests = [
        value
        for name, value in sorted(globals().items())
        if name.startswith("test_") and callable(value)
    ]
    failures = 0
    for test in tests:
        try:
            test()
        except Exception as exc:  # noqa: BLE001
            failures += 1
            print(f"[FAIL] {test.__name__}: {exc}")
        else:
            print(f"[PASS] {test.__name__}")
    print(f"{len(tests) - failures}/{len(tests)} tests passed")
    return failures


if __name__ == "__main__":
    sys.exit(1 if _run_all() else 0)

