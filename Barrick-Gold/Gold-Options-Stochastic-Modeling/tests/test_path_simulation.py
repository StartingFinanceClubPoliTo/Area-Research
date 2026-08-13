"""Small deterministic checks for the publication path engines."""

import numpy as np

from path_simulation import (
    forward_rates_from_zero_curve,
    simulate_bates_paths,
    simulate_full_hawkes_paths,
    terminal_statistics,
)


def test_bates_paths_are_reproducible_and_positive():
    params = (0.04, 1.5, 0.04, 0.3, -0.6, 0.5, -0.08, 0.13)
    rates = np.repeat(0.03, 8)
    first = simulate_bates_paths(100.0, rates, params, 0.05, 16, 7)
    second = simulate_bates_paths(100.0, rates, params, 0.05, 16, 7)
    np.testing.assert_allclose(first[0], second[0])
    assert np.all(first[0] > 0.0)
    assert first[2].shape == (16, 9)


def test_full_hawkes_paths_include_variance_intensity_and_counts():
    params = {
        "v0": 0.04, "kappa": 1.5, "theta": 0.04, "xi": 0.3, "rho": -0.6,
        "lambda0": 0.5, "lambda_bar": 0.4, "alpha": 0.2, "beta": 1.0,
        "mu_J": -0.08, "sigma_J": 0.13,
    }
    outputs = simulate_full_hawkes_paths(
        100.0, np.repeat(0.03, 6), params, 0.1, 12, 11
    )
    prices, variances, counts, intensities = outputs
    assert prices.shape == variances.shape == counts.shape == intensities.shape == (12, 7)
    assert np.all(prices > 0.0)
    assert np.all(variances >= 0.0)
    assert np.all(intensities > 0.0)
    assert np.all(np.diff(counts, axis=1) >= 0)


def test_forward_rates_preserve_zero_curve_discount_integral():
    maturities = np.array([0.25, 1.0, 2.0, 5.0])
    zero_rates = np.array([0.03, 0.035, 0.04, 0.045])
    forwards = forward_rates_from_zero_curve(maturities, zero_rates, 5.0, 20)
    dt = 5.0 / 20
    assert abs(np.sum(forwards * dt) - 5.0 * zero_rates[-1]) < 1e-12


def test_terminal_statistics_include_percentage_return_shape():
    paths = np.array([[100.0, 80.0], [100.0, 100.0], [100.0, 130.0]])
    stats = terminal_statistics("test", 100.0, paths, simulation_method="unit")
    assert np.isclose(stats["return_p00_pct"], -20.0)
    assert np.isclose(stats["return_p50_pct"], 0.0)
    assert np.isclose(stats["return_p100_pct"], 30.0)
    assert "return_skewness" in stats and "return_excess_kurtosis" in stats
