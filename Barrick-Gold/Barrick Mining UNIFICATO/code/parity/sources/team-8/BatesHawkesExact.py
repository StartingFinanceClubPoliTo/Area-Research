"""
Exact (event-dependent) Bates-Hawkes option-pricing engine.

This module implements a *genuine* self-exciting jump model, in contrast with
the stationary-intensity ``BatesHawkes`` proxy in the repository root. There,
Hawkes jumps are collapsed onto an effective constant Bates intensity
``lambda_eff = lambda0 / (1 - alpha / beta)``; here the jump intensity is a
state variable that jumps up by ``alpha`` at every jump and mean-reverts at
speed ``beta`` towards a baseline ``lambda_bar``:

    dlambda_t = beta (lambda_bar - lambda_t) dt + alpha dN_t.

Under the risk-neutral measure the log-price is

    dX_t = (r - q - 0.5 sigma^2 - kappa_J lambda_t) dt + sigma dW_t + Y dN_t,

with log-normal jump sizes ``Y ~ Normal(mu_J, sigma_J^2)`` and martingale
compensator ``kappa_J = E[e^Y - 1] = exp(mu_J + 0.5 sigma_J^2) - 1``. The same
self-exciting jump block can be attached to a Heston diffusion, giving the
Heston-Hawkes model of Souto Arias, Cirillo & Oosterlee (arXiv:2205.13321).

The intensity is affine, so the characteristic function of the compensated jump
martingale ``M_t = sum_{j<=N_t} Y_j - kappa_J int_0^t lambda_s ds`` is
semi-analytic:

    E[exp(i v M_tau) | lambda_0] = exp(A(v, tau) + B(v, tau) lambda_0),

where ``A`` and ``B`` solve the Riccati-type ODE system (forward in ``tau``)

    dA/dtau = beta lambda_bar B,
    dB/dtau = psi_Y(v) exp(alpha B) - beta B - i v kappa_J - 1,     A(0)=B(0)=0,

with ``psi_Y(v) = exp(i v mu_J - 0.5 sigma_J^2 v^2)``. For ``alpha = 0`` the
intensity is deterministic and the system has the closed form used below; for
``alpha > 0`` it is integrated numerically with a vectorised complex RK4 scheme.

The full characteristic function of ``X_T = log S_T`` is the product of a
jump-free diffusion characteristic function and ``CF_M``:

    phi_X(v) = phi_diffusion(v) * exp(A(v, T) + B(v, T) lambda0).

Two diffusion blocks are provided:

* ``hawkes_charfunc_constvol`` uses a Black-Scholes diffusion. With the jumps
  switched off (``lambda0 = lambda_bar = 0``) it collapses exactly onto
  Black-Scholes.
* ``hawkes_charfunc`` reuses the repository's stabilised Heston characteristic
  function. With ``alpha = 0`` and ``lambda0 = lambda_bar`` the intensity is
  constant and it reproduces the existing closed-form ``Bates`` pricer exactly.

European options are priced with the same single-integral Carr-Madan/Lewis
inversion used by ``Heston``/``Bates``; an independent Gil-Pelaez two-integral
inversion is also provided and used to validate put-call parity.
"""

import math
import os
import sys

import numpy as np

# Make the repository-root model modules importable.
_ROOT = os.path.dirname(os.path.abspath(__file__))
if _ROOT not in sys.path:
    sys.path.insert(0, _ROOT)

from Heston import Heston  # noqa: E402
from Bates import Bates  # noqa: E402
from fourier_pricing import (  # noqa: E402
    adaptive_cos_call_prices,
    carr_madan_call,
    cos_call_prices_from_values,
    gil_pelaez_call_put,
)

# Real part of ``alpha * B`` is capped here before exponentiation. This only
# bites in the far Fourier tail, where the diffusion factor exp(-0.5 sigma^2 v^2 T)
# has already driven the integrand to zero; it prevents inf/NaN without affecting
# the contributing region of the integral.
_EXP_REAL_CAP = 700.0


class BatesHawkesExact(Bates):
    """Exact self-exciting jump-diffusion pricer (constant-vol or Heston).

    All methods are static, matching the style of ``Heston``/``Bates``. Naming
    is kept unambiguous on purpose:

    * ``lambda0``     - jump intensity at the pricing date (initial state);
    * ``lambda_bar``  - long-run/baseline mean-reversion level of the intensity;
    * ``alpha``       - upward jump of the intensity at each event;
    * ``beta``        - mean-reversion / decay speed of the intensity.

    Stationarity of the exponential-kernel Hawkes process requires the branching
    ratio ``alpha / beta`` to be strictly below one, i.e. ``alpha < beta``.
    """

    # --- A. PARAMETER VALIDATION ---------------------------------------------
    @staticmethod
    def validate_params(lambda0, lambda_bar, alpha, beta, sigma=None, sigma_J=None):
        """Validate Hawkes (and optionally diffusion/jump) parameters.

        Raises ``ValueError`` if the parameters fall outside the admissible,
        stationary region. This is the guard used by calibration; the pricing
        routines themselves stay permissive so degenerate limits (no jumps) can
        be evaluated.
        """
        if beta <= 0:
            raise ValueError("beta must be positive.")
        if alpha < 0:
            raise ValueError("alpha must be non-negative.")
        if alpha >= beta:
            raise ValueError(
                "stationarity requires alpha < beta (branching ratio < 1)."
            )
        if lambda0 <= 0:
            raise ValueError("lambda0 must be positive.")
        if lambda_bar <= 0:
            raise ValueError("lambda_bar must be positive.")
        if sigma is not None and sigma <= 0:
            raise ValueError("sigma must be positive.")
        if sigma_J is not None and sigma_J < 0:
            raise ValueError("sigma_J must be non-negative.")

    @staticmethod
    def branching_ratio(alpha, beta):
        """Exponential-kernel branching ratio alpha / beta."""
        if beta <= 0:
            raise ValueError("beta must be positive.")
        return alpha / beta

    # --- B. JUMP BUILDING BLOCKS ---------------------------------------------
    @staticmethod
    def kappa_J(mu_J, sigma_J):
        """Risk-neutral jump compensator kappa_J = E[exp(Y) - 1]."""
        return math.exp(mu_J + 0.5 * sigma_J ** 2) - 1.0

    @staticmethod
    def _psi_Y(v, mu_J, sigma_J):
        """Characteristic function of a single log-normal jump size Y."""
        return np.exp(1j * v * mu_J - 0.5 * sigma_J ** 2 * v ** 2)

    # --- C. EXACT HAWKES JUMP CHARACTERISTIC FUNCTION ------------------------
    @staticmethod
    def _jump_ode_solution(u_arr, T, lambda_bar, alpha, beta, mu_J, sigma_J,
                           n_steps=None):
        """Integrate the (A, B) Riccati system forward in tau over [0, T].

        ``u_arr`` is a complex ndarray of Fourier arguments. Returns the pair of
        complex ndarrays ``(A, B)`` evaluated at ``tau = T``. The system is
        solved with a fixed-step vectorised RK4 scheme; A is driven by B.
        """
        kJ = BatesHawkesExact.kappa_J(mu_J, sigma_J)
        psi = BatesHawkesExact._psi_Y(u_arr, mu_J, sigma_J)
        const = -1j * u_arr * kJ - 1.0  # part of dB/dtau independent of B

        if n_steps is None:
            n_steps = min(600, max(80, int(math.ceil(120.0 * T))))
        h = T / n_steps

        A = np.zeros_like(u_arr)
        B = np.zeros_like(u_arr)

        def dB(Bc):
            z = alpha * Bc
            # Cap the real part of the exponent to avoid overflow in the tail.
            expaB = np.exp(np.minimum(z.real, _EXP_REAL_CAP) + 1j * z.imag)
            return psi * expaB - beta * Bc + const

        coeff = beta * lambda_bar
        for _ in range(n_steps):
            k1B = dB(B)
            k2B = dB(B + 0.5 * h * k1B)
            k3B = dB(B + 0.5 * h * k2B)
            k4B = dB(B + h * k3B)
            # A is the integral of (beta * lambda_bar * B); use the same stages.
            k1A = coeff * B
            k2A = coeff * (B + 0.5 * h * k1B)
            k3A = coeff * (B + 0.5 * h * k2B)
            k4A = coeff * (B + h * k3B)
            B = B + (h / 6.0) * (k1B + 2.0 * k2B + 2.0 * k3B + k4B)
            A = A + (h / 6.0) * (k1A + 2.0 * k2A + 2.0 * k3A + k4A)
        return A, B

    @staticmethod
    def jump_cf(u, T, lambda0, lambda_bar, alpha, beta, mu_J, sigma_J,
                n_steps=None):
        """Characteristic function ``CF_M(u)`` of the compensated jump term.

        Accepts a scalar or ndarray ``u`` (real or complex) and returns the
        matching complex value(s). For ``alpha == 0`` the deterministic-intensity
        closed form is used; otherwise the Riccati ODE is integrated.
        """
        scalar_input = np.ndim(u) == 0
        u_arr = np.atleast_1d(np.asarray(u, dtype=np.complex128))

        if T <= 0:
            out = np.ones_like(u_arr)
            return complex(out[0]) if scalar_input else out

        kJ = BatesHawkesExact.kappa_J(mu_J, sigma_J)

        if alpha == 0.0:
            # Deterministic intensity lambda_t = lambda_bar + (lambda0 - lambda_bar) e^{-beta t}.
            # B(tau) = (c/beta)(1 - e^{-beta tau}),  A(tau) = lambda_bar c (tau - (1-e^{-beta tau})/beta).
            c = BatesHawkesExact._psi_Y(u_arr, mu_J, sigma_J) - 1.0 - 1j * u_arr * kJ
            decay = -np.expm1(-beta * T)  # 1 - e^{-beta T}, stable
            B = c * decay / beta
            A = lambda_bar * c * (T - decay / beta)
        else:
            A, B = BatesHawkesExact._jump_ode_solution(
                u_arr, T, lambda_bar, alpha, beta, mu_J, sigma_J, n_steps
            )

        cf = np.exp(A + B * lambda0)
        return complex(cf[0]) if scalar_input else cf

    # --- D. FULL CHARACTERISTIC FUNCTIONS ------------------------------------
    @staticmethod
    def _bs_charfunc(u, S0, sigma, T, r, q=0.0):
        """Black-Scholes characteristic function of log S_T (no jumps)."""
        drift = math.log(S0) + (r - q - 0.5 * sigma ** 2) * T
        return np.exp(1j * u * drift - 0.5 * sigma ** 2 * u ** 2 * T)

    @staticmethod
    def hawkes_charfunc_constvol(u, S0, sigma, lambda0, lambda_bar, alpha, beta,
                                 mu_J, sigma_J, T, r, q=0.0, n_steps=None):
        """Constant-volatility exact-Hawkes characteristic function of log S_T."""
        cf_diffusion = BatesHawkesExact._bs_charfunc(u, S0, sigma, T, r, q)
        cf_jump = BatesHawkesExact.jump_cf(
            u, T, lambda0, lambda_bar, alpha, beta, mu_J, sigma_J, n_steps
        )
        return cf_diffusion * cf_jump

    @staticmethod
    def hawkes_charfunc(u, S0, v0, kappa, theta, sigma, rho, lambda0, lambda_bar,
                        alpha, beta, mu_J, sigma_J, T, r, q=0.0, n_steps=None):
        """Heston-Hawkes exact characteristic function of log S_T.

        Reuses the repository's stabilised Heston characteristic function for the
        diffusion block and multiplies by the exact self-exciting jump transform.
        """
        cf_diffusion = Heston.heston_charfunc(u, S0, v0, kappa, theta, sigma, rho, T, r, q)
        cf_jump = BatesHawkesExact.jump_cf(
            u, T, lambda0, lambda_bar, alpha, beta, mu_J, sigma_J, n_steps
        )
        return cf_diffusion * cf_jump

    # --- E. CARR-MADAN SINGLE-INTEGRAL PRICING -------------------------------
    @staticmethod
    def _carr_madan_call(charfunc, S0, K, T, r, q=0.0):
        """European call via the single-integral Carr-Madan/Lewis inversion.

        ``charfunc`` is a callable ``u -> E[exp(i u log S_T)]``. The formula is
        identical to the one used by ``Heston``/``Bates``.
        """
        return carr_madan_call(charfunc, S0, K, T, r, q)

    @staticmethod
    def hawkes_price_constvol_fast(S0, K, T, sigma, lambda0, lambda_bar, alpha,
                                   beta, mu_J, sigma_J, r, q=0.0, n_steps=None):
        """Price a European call under the constant-vol exact-Hawkes model."""
        def cf(u):
            return BatesHawkesExact.hawkes_charfunc_constvol(
                u, S0, sigma, lambda0, lambda_bar, alpha, beta, mu_J, sigma_J,
                T, r, q, n_steps
            )
        return BatesHawkesExact._carr_madan_call(cf, S0, K, T, r, q)

    @staticmethod
    def hawkes_price_fast(S0, K, T, v0, kappa, theta, sigma, rho, lambda0,
                          lambda_bar, alpha, beta, mu_J, sigma_J, r, q=0.0,
                          n_steps=None):
        """Price a European call under the Heston-Hawkes exact model."""
        def cf(u):
            return BatesHawkesExact.hawkes_charfunc(
                u, S0, v0, kappa, theta, sigma, rho, lambda0, lambda_bar, alpha,
                beta, mu_J, sigma_J, T, r, q, n_steps
            )
        return BatesHawkesExact._carr_madan_call(cf, S0, K, T, r, q)

    # --- F. GIL-PELAEZ TWO-INTEGRAL PRICING (INDEPENDENT INVERSION) ----------
    @staticmethod
    def _gil_pelaez_call_put(charfunc, S0, K, T, r, q=0.0):
        """Return ``(call, put)`` via the Gil-Pelaez probabilities P1, P2.

        This is a structurally different Fourier inversion from
        ``_carr_madan_call``; agreement between the two validates the
        characteristic function and the inversion, and hence put-call parity.
        """
        return gil_pelaez_call_put(charfunc, S0, K, T, r, q)

    @staticmethod
    def hawkes_call_price_constvol_gp(S0, K, T, sigma, lambda0, lambda_bar, alpha,
                                      beta, mu_J, sigma_J, r, q=0.0, n_steps=None):
        """Constant-vol exact-Hawkes call via Gil-Pelaez (cross-check)."""
        def cf(u):
            return BatesHawkesExact.hawkes_charfunc_constvol(
                u, S0, sigma, lambda0, lambda_bar, alpha, beta, mu_J, sigma_J,
                T, r, q, n_steps
            )
        return BatesHawkesExact._gil_pelaez_call_put(cf, S0, K, T, r, q)[0]

    @staticmethod
    def hawkes_put_price_constvol_fast(S0, K, T, sigma, lambda0, lambda_bar, alpha,
                                       beta, mu_J, sigma_J, r, q=0.0, n_steps=None):
        """Constant-vol exact-Hawkes European put via Gil-Pelaez."""
        def cf(u):
            return BatesHawkesExact.hawkes_charfunc_constvol(
                u, S0, sigma, lambda0, lambda_bar, alpha, beta, mu_J, sigma_J,
                T, r, q, n_steps
            )
        return BatesHawkesExact._gil_pelaez_call_put(cf, S0, K, T, r, q)[1]

    @staticmethod
    def hawkes_put_price_fast(S0, K, T, v0, kappa, theta, sigma, rho, lambda0,
                              lambda_bar, alpha, beta, mu_J, sigma_J, r, q=0.0,
                              n_steps=None):
        """Heston-Hawkes exact European put via Gil-Pelaez."""
        def cf(u):
            return BatesHawkesExact.hawkes_charfunc(
                u, S0, v0, kappa, theta, sigma, rho, lambda0, lambda_bar, alpha,
                beta, mu_J, sigma_J, T, r, q, n_steps
            )
        return BatesHawkesExact._gil_pelaez_call_put(cf, S0, K, T, r, q)[1]

    # --- G. COS METHOD (fast, vectorised across strikes) ---------------------
    # The COS method (Fang & Oosterlee, 2008) evaluates the characteristic
    # function once on a fixed cosine grid and prices every strike of a maturity
    # from it. Because the exact jump transform is solved for the whole grid in a
    # single vectorised ODE pass, this is ~100x faster than the per-strike
    # Carr-Madan inversion and is what makes exact-Hawkes calibration tractable.

    @staticmethod
    def _cos_call_prices(phi_vals, u_k, a, b, K_array, r, T):
        """COS European-call prices for a vector of strikes at one maturity.

        ``phi_vals`` are the characteristic-function values of log S_T on the
        grid ``u_k = k*pi/(b-a)`` over the truncation range ``[a, b]``.
        """
        return cos_call_prices_from_values(
            phi_vals, u_k, a, b, K_array, r, T
        )

    @staticmethod
    def hawkes_price_constvol_cos(S0, K, T, sigma, lambda0, lambda_bar, alpha,
                                  beta, mu_J, sigma_J, r, q=0.0, N=256, L=12.0,
                                  n_steps=None):
        """Constant-vol exact-Hawkes call prices for a vector of strikes (COS).

        Returns a 1-D array aligned with ``K``. The truncation range is set from
        the first two cumulants estimated directly from the characteristic
        function, so it adapts to the current parameters during calibration.
        """
        k_arr = np.atleast_1d(np.asarray(K, dtype=float))
        if T <= 0:
            return np.maximum(S0 - k_arr, 0.0)

        def phi(u):
            return BatesHawkesExact.hawkes_charfunc_constvol(
                u, S0, sigma, lambda0, lambda_bar, alpha, beta, mu_J, sigma_J,
                T, r, q, n_steps
            )

        return adaptive_cos_call_prices(
            phi, k_arr, T, r, terms=N, width_scale=L
        )
    @staticmethod
    def hawkes_price_cos(S0, K, T, v0, kappa, theta, sigma, rho, lambda0,
                         lambda_bar, alpha, beta, mu_J, sigma_J, r, q=0.0,
                         N=256, L=12.0, n_steps=None):
        """Heston--Hawkes call prices for a vector of strikes using COS.

        This is the full affine Bates--Hawkes pricer used by calibration. The
        Heston variance block and the event-dependent Hawkes jump transform are
        evaluated on the same Fourier grid, once per maturity.
        """
        k_arr = np.atleast_1d(np.asarray(K, dtype=float))
        if T <= 0:
            return np.maximum(S0 - k_arr, 0.0)

        def phi(u):
            return BatesHawkesExact.hawkes_charfunc(
                u, S0, v0, kappa, theta, sigma, rho,
                lambda0, lambda_bar, alpha, beta, mu_J, sigma_J,
                T, r, q, n_steps,
            )

        return adaptive_cos_call_prices(
            phi, k_arr, T, r, terms=N, width_scale=L
        )

    # Calibration objectives and optimizers are defined in Hawkes.py.
