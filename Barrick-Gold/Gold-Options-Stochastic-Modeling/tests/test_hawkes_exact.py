"""
Mathematical sanity and limit tests for the exact (event-dependent) Bates-Hawkes
option-pricing engine implemented in ``BatesHawkesExact``.

These tests are the specification for the exact Hawkes model. They check the
properties that distinguish a correct affine characteristic function from a
plausible-but-wrong one:

* characteristic function normalisation ``phi(0) = 1``;
* the Black-Scholes limit (no jumps, constant volatility);
* the Bates limit (``alpha = 0`` with ``lambda0 = lambda_bar`` -> constant
  intensity), which must reproduce the existing closed-form Bates pricer;
* continuity of the ODE solution towards the closed form as ``alpha -> 0``;
* the risk-neutral martingale condition ``E[S_T] = S0 exp((r - q) T)``;
* absence of NaN/Inf on a reasonable Fourier grid;
* put-call parity, checked across two independent Fourier inversions;
* rejection of the non-stationary regime ``alpha >= beta``.

The file is runnable directly (``python tests/test_hawkes_exact.py``) and is
also discoverable by ``pytest``.
"""

import math
import os
import sys

import numpy as np

# Make the repository root model modules importable.
HERE = os.path.dirname(os.path.abspath(__file__))
ROOT = os.path.dirname(HERE)
for _p in (HERE, ROOT):
    if _p not in sys.path:
        sys.path.insert(0, _p)

from BatesHawkesExact import BatesHawkesExact  # noqa: E402
from Bates import Bates  # noqa: E402
from BnS import BnS  # noqa: E402
from Hawkes import ExactHawkesCalibration  # noqa: E402


# --- Shared test parameters --------------------------------------------------

S0 = 100.0
R = 0.03
Q = 0.0
T = 0.5

# Constant-volatility diffusion.
SIGMA = 0.20

# Heston diffusion block (shared with the existing Bates pricer).
V0, KAPPA, THETA, SIGMA_V, RHO = 0.04, 1.5, 0.04, 0.30, -0.6

# Log-normal jump sizes.
MU_J, SIGMA_J = -0.08, 0.13

# Genuinely self-exciting Hawkes parameters (branching ratio alpha/beta = 0.5).
LAMBDA0, LAMBDA_BAR, ALPHA, BETA = 0.90, 0.60, 0.80, 1.60

# Constant-intensity case used for the Bates limit.
LAMBDA_CONST = 0.50

STRIKES = [80.0, 90.0, 100.0, 110.0, 120.0]


# --- Validation --------------------------------------------------------------

def test_validate_rejects_alpha_ge_beta():
    """Non-stationary self-excitation (alpha >= beta) must be rejected."""
    for bad_alpha in (BETA, BETA + 0.5, 2.0 * BETA):
        try:
            BatesHawkesExact.validate_params(LAMBDA0, LAMBDA_BAR, bad_alpha, BETA)
        except ValueError:
            continue
        raise AssertionError(f"alpha={bad_alpha} >= beta={BETA} was not rejected")


def test_validate_rejects_nonpositive_intensities():
    """lambda0 > 0 and lambda_bar > 0 are required."""
    for lam0, lambar in ((0.0, LAMBDA_BAR), (-0.1, LAMBDA_BAR),
                          (LAMBDA0, 0.0), (LAMBDA0, -0.1)):
        try:
            BatesHawkesExact.validate_params(lam0, lambar, ALPHA, BETA)
        except ValueError:
            continue
        raise AssertionError(f"non-positive intensity (lambda0={lam0}, "
                             f"lambda_bar={lambar}) was not rejected")


def test_validate_accepts_valid_params():
    """A stationary, strictly positive parameter set must be accepted."""
    # Should not raise.
    BatesHawkesExact.validate_params(LAMBDA0, LAMBDA_BAR, ALPHA, BETA,
                                     sigma=SIGMA, sigma_J=SIGMA_J)


# --- phi(0) = 1 --------------------------------------------------------------

def test_cf_at_zero_is_one_constvol():
    """The constant-vol Hawkes characteristic function satisfies phi(0) = 1."""
    phi0 = BatesHawkesExact.hawkes_charfunc_constvol(
        0.0, S0, SIGMA, LAMBDA0, LAMBDA_BAR, ALPHA, BETA, MU_J, SIGMA_J, T, R, Q
    )
    assert abs(phi0 - 1.0) < 1e-8, f"const-vol phi(0) = {phi0}, expected 1"


def test_cf_at_zero_is_one_heston():
    """The Heston-Hawkes characteristic function satisfies phi(0) = 1."""
    phi0 = BatesHawkesExact.hawkes_charfunc(
        0.0, S0, V0, KAPPA, THETA, SIGMA_V, RHO,
        LAMBDA0, LAMBDA_BAR, ALPHA, BETA, MU_J, SIGMA_J, T, R, Q
    )
    assert abs(phi0 - 1.0) < 1e-8, f"Heston-Hawkes phi(0) = {phi0}, expected 1"


# --- Black-Scholes limit -----------------------------------------------------

def test_black_scholes_limit():
    """With no jumps (lambda0 = lambda_bar = 0) the constant-vol Hawkes price
    collapses onto the closed-form Black-Scholes call price."""
    for K in STRIKES:
        price = BatesHawkesExact.hawkes_price_constvol_fast(
            S0, K, T, SIGMA,
            lambda0=0.0, lambda_bar=0.0, alpha=0.0, beta=BETA,
            mu_J=0.0, sigma_J=SIGMA_J, r=R, q=Q,
        )
        bs = BnS.bs_call_price(S0, K, T, R, SIGMA, Q)
        assert abs(price - bs) < 1e-2, f"K={K}: Hawkes {price} vs BS {bs}"


# --- Bates limit -------------------------------------------------------------

def test_bates_limit_alpha_zero():
    """With alpha = 0 and lambda0 = lambda_bar the intensity is constant, so the
    Heston-Hawkes price must match the existing closed-form Bates pricer."""
    for K in STRIKES:
        price = BatesHawkesExact.hawkes_price_fast(
            S0, K, T, V0, KAPPA, THETA, SIGMA_V, RHO,
            lambda0=LAMBDA_CONST, lambda_bar=LAMBDA_CONST, alpha=0.0, beta=2.0,
            mu_J=MU_J, sigma_J=SIGMA_J, r=R, q=Q,
        )
        bates = Bates.bates_price_fast(
            S0, K, T, V0, KAPPA, THETA, SIGMA_V, RHO,
            LAMBDA_CONST, MU_J, SIGMA_J, R, Q
        )
        assert abs(price - bates) < 1e-6, f"K={K}: Hawkes {price} vs Bates {bates}"


def test_ode_matches_closed_form_as_alpha_to_zero():
    """For alpha -> 0 the numerically integrated jump transform must converge to
    the analytic deterministic-intensity (alpha = 0) closed form."""
    us = np.array([0.25, 0.5, 1.0, 2.0, 5.0, 10.0])
    cf_ode = BatesHawkesExact.jump_cf(
        us, T, LAMBDA0, LAMBDA_BAR, 1e-7, BETA, MU_J, SIGMA_J
    )
    cf_closed = BatesHawkesExact.jump_cf(
        us, T, LAMBDA0, LAMBDA_BAR, 0.0, BETA, MU_J, SIGMA_J
    )
    assert np.allclose(cf_ode, cf_closed, atol=1e-5), (
        f"ODE vs closed form mismatch: {np.abs(cf_ode - cf_closed)}"
    )


# --- Martingale condition ----------------------------------------------------

def test_martingale_constvol():
    """phi(-i) = E[S_T] = S0 exp((r - q) T) for the self-exciting model."""
    fwd = S0 * math.exp((R - Q) * T)
    phi = BatesHawkesExact.hawkes_charfunc_constvol(
        -1j, S0, SIGMA, LAMBDA0, LAMBDA_BAR, ALPHA, BETA, MU_J, SIGMA_J, T, R, Q
    )
    assert abs(phi - fwd) / fwd < 1e-6, f"E[S_T]={phi}, forward={fwd}"


def test_martingale_heston():
    """phi(-i) = E[S_T] = S0 exp((r - q) T) for the Heston-Hawkes model."""
    fwd = S0 * math.exp((R - Q) * T)
    phi = BatesHawkesExact.hawkes_charfunc(
        -1j, S0, V0, KAPPA, THETA, SIGMA_V, RHO,
        LAMBDA0, LAMBDA_BAR, ALPHA, BETA, MU_J, SIGMA_J, T, R, Q
    )
    assert abs(phi - fwd) / fwd < 1e-6, f"E[S_T]={phi}, forward={fwd}"


# --- Numerical robustness ----------------------------------------------------

def test_no_nan_or_inf_on_grid():
    """The characteristic functions stay finite on a reasonable Fourier grid."""
    us = np.linspace(1e-3, 100.0, 128)
    cf_cv = BatesHawkesExact.hawkes_charfunc_constvol(
        us, S0, SIGMA, LAMBDA0, LAMBDA_BAR, ALPHA, BETA, MU_J, SIGMA_J, T, R, Q
    )
    cf_h = BatesHawkesExact.hawkes_charfunc(
        us, S0, V0, KAPPA, THETA, SIGMA_V, RHO,
        LAMBDA0, LAMBDA_BAR, ALPHA, BETA, MU_J, SIGMA_J, T, R, Q
    )
    assert np.all(np.isfinite(cf_cv)), "non-finite values in const-vol CF"
    assert np.all(np.isfinite(cf_h)), "non-finite values in Heston-Hawkes CF"


def test_prices_finite_and_positive():
    """Self-exciting call prices are finite, non-negative, and below the spot."""
    for K in STRIKES:
        price = BatesHawkesExact.hawkes_price_constvol_fast(
            S0, K, T, SIGMA, LAMBDA0, LAMBDA_BAR, ALPHA, BETA, MU_J, SIGMA_J, R, Q
        )
        assert math.isfinite(price), f"K={K}: non-finite price"
        assert 0.0 <= price <= S0, f"K={K}: price {price} outside [0, S0]"


def test_call_price_monotone_in_strike():
    """Call prices must be non-increasing in the strike."""
    prices = [
        BatesHawkesExact.hawkes_price_constvol_fast(
            S0, K, T, SIGMA, LAMBDA0, LAMBDA_BAR, ALPHA, BETA, MU_J, SIGMA_J, R, Q
        )
        for K in STRIKES
    ]
    for lo, hi, p_lo, p_hi in zip(STRIKES, STRIKES[1:], prices, prices[1:]):
        assert p_lo >= p_hi - 1e-6, f"call not monotone: K={lo}->{p_lo}, K={hi}->{p_hi}"


# --- Put-call parity ---------------------------------------------------------

def test_put_call_parity_self_exciting():
    """Carr-Madan call and an independent Gil-Pelaez put obey put-call parity."""
    fwd_diff = S0 * math.exp(-Q * T) - 0.0  # placeholder, recomputed per strike
    for K in STRIKES:
        call = BatesHawkesExact.hawkes_price_constvol_fast(
            S0, K, T, SIGMA, LAMBDA0, LAMBDA_BAR, ALPHA, BETA, MU_J, SIGMA_J, R, Q
        )
        put = BatesHawkesExact.hawkes_put_price_constvol_fast(
            S0, K, T, SIGMA, LAMBDA0, LAMBDA_BAR, ALPHA, BETA, MU_J, SIGMA_J, R, Q
        )
        fwd_diff = S0 * math.exp(-Q * T) - K * math.exp(-R * T)
        assert abs((call - put) - fwd_diff) < 1e-2, (
            f"K={K}: parity broken, C-P={call - put}, fwd_diff={fwd_diff}"
        )


def test_carr_madan_and_gil_pelaez_calls_agree():
    """Two independent Fourier inversions of the same CF give the same call."""
    for K in STRIKES:
        call_cm = BatesHawkesExact.hawkes_price_constvol_fast(
            S0, K, T, SIGMA, LAMBDA0, LAMBDA_BAR, ALPHA, BETA, MU_J, SIGMA_J, R, Q
        )
        call_gp = BatesHawkesExact.hawkes_call_price_constvol_gp(
            S0, K, T, SIGMA, LAMBDA0, LAMBDA_BAR, ALPHA, BETA, MU_J, SIGMA_J, R, Q
        )
        assert abs(call_cm - call_gp) < 1e-2, (
            f"K={K}: Carr-Madan {call_cm} vs Gil-Pelaez {call_gp}"
        )


# --- COS pricer (fast, used by calibration) ----------------------------------

def test_cos_matches_carr_madan_constvol():
    """The vectorised COS pricer matches the trusted Carr-Madan pricer."""
    ks = np.array(STRIKES)
    cos_prices = BatesHawkesExact.hawkes_price_constvol_cos(
        S0, ks, T, SIGMA, LAMBDA0, LAMBDA_BAR, ALPHA, BETA, MU_J, SIGMA_J, R, Q
    )
    for K, p_cos in zip(ks, cos_prices):
        p_cm = BatesHawkesExact.hawkes_price_constvol_fast(
            S0, K, T, SIGMA, LAMBDA0, LAMBDA_BAR, ALPHA, BETA, MU_J, SIGMA_J, R, Q
        )
        assert abs(p_cos - p_cm) < 5e-3, f"K={K}: COS {p_cos} vs Carr-Madan {p_cm}"


def test_cos_black_scholes_limit():
    """COS with no jumps reproduces the closed-form Black-Scholes price."""
    ks = np.array(STRIKES)
    cos_prices = BatesHawkesExact.hawkes_price_constvol_cos(
        S0, ks, T, SIGMA, 0.0, 0.0, 0.0, BETA, 0.0, SIGMA_J, R, Q
    )
    for K, p_cos in zip(ks, cos_prices):
        bs = BnS.bs_call_price(S0, K, T, R, SIGMA, Q)
        assert abs(p_cos - bs) < 5e-3, f"K={K}: COS {p_cos} vs BS {bs}"


def test_cos_matches_carr_madan_heston_hawkes():
    """The full Heston--Hawkes COS pricer matches Carr-Madan."""
    ks = np.array(STRIKES)
    cos_prices = BatesHawkesExact.hawkes_price_cos(
        S0, ks, T, V0, KAPPA, THETA, SIGMA_V, RHO,
        LAMBDA0, LAMBDA_BAR, ALPHA, BETA, MU_J, SIGMA_J, R, Q,
    )
    for K, p_cos in zip(ks, cos_prices):
        p_cm = BatesHawkesExact.hawkes_price_fast(
            S0, K, T, V0, KAPPA, THETA, SIGMA_V, RHO,
            LAMBDA0, LAMBDA_BAR, ALPHA, BETA, MU_J, SIGMA_J, R, Q,
        )
        assert abs(p_cos - p_cm) < 8e-3, f"K={K}: COS {p_cos} vs Carr-Madan {p_cm}"


def test_cos_full_bates_limit():
    """At zero branching the full COS pricer reduces to Bates."""
    ks = np.array(STRIKES)
    cos_prices = BatesHawkesExact.hawkes_price_cos(
        S0, ks, T, V0, KAPPA, THETA, SIGMA_V, RHO,
        LAMBDA_CONST, LAMBDA_CONST, 0.0, 2.0, MU_J, SIGMA_J, R, Q,
    )
    for K, p_cos in zip(ks, cos_prices):
        p_bates = Bates.bates_price_fast(
            S0, K, T, V0, KAPPA, THETA, SIGMA_V, RHO,
            LAMBDA_CONST, MU_J, SIGMA_J, R, Q,
        )
        assert abs(p_cos - p_bates) < 8e-3, f"K={K}: COS {p_cos} vs Bates {p_bates}"


def test_put_call_parity_heston_hawkes():
    """The full stochastic-volatility Hawkes model obeys put-call parity."""
    for K in STRIKES:
        call = BatesHawkesExact.hawkes_price_fast(
            S0, K, T, V0, KAPPA, THETA, SIGMA_V, RHO,
            LAMBDA0, LAMBDA_BAR, ALPHA, BETA, MU_J, SIGMA_J, R, Q,
        )
        put = BatesHawkesExact.hawkes_put_price_fast(
            S0, K, T, V0, KAPPA, THETA, SIGMA_V, RHO,
            LAMBDA0, LAMBDA_BAR, ALPHA, BETA, MU_J, SIGMA_J, R, Q,
        )
        parity = S0 * math.exp(-Q * T) - K * math.exp(-R * T)
        assert abs((call - put) - parity) < 1e-2


# --- Calibration guard -------------------------------------------------------

def test_calibration_objective_penalizes_alpha_ge_beta():
    """The calibration objective must reject non-stationary alpha >= beta."""
    import pandas as pd

    df = pd.DataFrame({"K": [100.0], "T": [T], "rate": [R],
                       "price": [6.4], "vega": [10.0]})
    # params = [sigma, lambda_bar, alpha, beta, mu_J, sigma_J]
    bad = [SIGMA, LAMBDA_BAR, 1.5, 1.0, MU_J, SIGMA_J]  # alpha=1.5 >= beta=1.0
    value = ExactHawkesCalibration.objective_constvol(bad, df, S0, Q)
    assert value >= 1e8, f"alpha>=beta not penalized in objective: {value}"


def test_calibration_objective_finite_for_valid_params():
    """The objective returns a finite, non-negative error for valid params."""
    import pandas as pd

    df = pd.DataFrame({"K": [100.0], "T": [T], "rate": [R],
                       "price": [6.4], "vega": [10.0]})
    good = [SIGMA, LAMBDA_BAR, ALPHA, BETA, MU_J, SIGMA_J]
    value = ExactHawkesCalibration.objective_constvol(good, df, S0, Q)
    assert math.isfinite(value) and value >= 0.0, f"objective not finite/>=0: {value}"


def test_full_calibration_objective_finite():
    """The full Heston--Hawkes objective is finite for an admissible vector."""
    import pandas as pd

    df = pd.DataFrame({"K": [100.0], "T": [T], "rate": [R],
                       "price": [6.4], "vega": [10.0]})
    params = [V0, KAPPA, THETA, SIGMA_V, RHO, 0.9, 0.6, 0.5, BETA, MU_J, SIGMA_J]
    value = ExactHawkesCalibration.objective_heston(params, df, S0, Q, cos_N=128)
    assert math.isfinite(value) and value >= 0.0


def test_full_calibration_objective_rejects_nonstationary_branching():
    """The full objective rejects branching ratios outside [0, 1)."""
    import pandas as pd

    df = pd.DataFrame({"K": [100.0], "T": [T], "rate": [R],
                       "price": [6.4], "vega": [10.0]})
    params = [V0, KAPPA, THETA, SIGMA_V, RHO, 0.9, 0.6, 1.0, BETA, MU_J, SIGMA_J]
    value = ExactHawkesCalibration.objective_heston(params, df, S0, Q, cos_N=128)
    assert value >= 1e8


# --- Standalone runner -------------------------------------------------------

def _run_all():
    tests = [v for k, v in sorted(globals().items())
             if k.startswith("test_") and callable(v)]
    failures = 0
    for test in tests:
        name = test.__name__
        try:
            test()
        except Exception as exc:  # noqa: BLE001
            failures += 1
            print(f"[FAIL] {name}: {exc}")
        else:
            print(f"[PASS] {name}")
    print("-" * 60)
    total = len(tests)
    print(f"{total - failures}/{total} tests passed")
    return failures


if __name__ == "__main__":
    sys.exit(1 if _run_all() else 0)
