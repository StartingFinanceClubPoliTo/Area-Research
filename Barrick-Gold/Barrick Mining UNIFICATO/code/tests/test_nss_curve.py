from __future__ import annotations

import sys
from pathlib import Path

import numpy as np


TEAM8 = Path(__file__).resolve().parents[1] / "parity" / "sources" / "team-8"
if str(TEAM8) not in sys.path:
    sys.path.insert(0, str(TEAM8))

from nss_curve import fit_nss_curve, nss_zero_rates, quarterly_forward_rates  # noqa: E402


def test_nss_fit_recovers_a_smooth_curve_and_forward_integrals() -> None:
    parameters = {
        "beta0": 0.046,
        "beta1": -0.010,
        "beta2": 0.012,
        "beta3": 0.006,
        "tau1": 0.8,
        "tau2": 7.5,
    }
    maturities = np.array([1 / 12, 2 / 12, 0.25, 0.5, 1, 2, 3, 5, 7, 20, 30])
    observed = nss_zero_rates(maturities, parameters)
    fit = fit_nss_curve(maturities, observed)
    assert fit.rmse_bp < 0.01
    np.testing.assert_allclose(fit.fitted_rates, observed, atol=1e-6, rtol=0.0)

    forwards = quarterly_forward_rates(fit.parameters, 20)
    terminal_zero = nss_zero_rates(np.array([5.0]), fit.parameters)[0]
    assert forwards.shape == (20,)
    assert np.isfinite(forwards).all()
    np.testing.assert_allclose(
        np.sum(forwards * 0.25), terminal_zero * 5.0, atol=1e-12, rtol=0.0
    )


def test_nss_requires_enough_distinct_tenors() -> None:
    maturities = np.array([0.25, 0.5, 1.0, 2.0, 3.0, 5.0, 7.0])
    try:
        fit_nss_curve(maturities, np.repeat(0.04, len(maturities)))
    except ValueError as exc:
        assert "At least eight" in str(exc)
    else:  # pragma: no cover
        raise AssertionError("underspecified NSS fit should fail")
