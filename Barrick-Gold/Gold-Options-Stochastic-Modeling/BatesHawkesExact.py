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
from scipy.integrate import quad
from scipy.optimize import differential_evolution, minimize

# Make the repository-root modules importable (mirrors calibrations/common.py).
_ROOT = os.path.dirname(os.path.abspath(__file__))
if _ROOT not in sys.path:
    sys.path.insert(0, _ROOT)

from Heston import Heston  # noqa: E402
from Bates import Bates  # noqa: E402

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
        if T <= 0:
            return float(max(S0 - K, 0.0))

        def integrand(u):
            cf_shifted = charfunc(u - 1j)
            cf_standard = charfunc(u)
            num = np.exp(-r * T) * (cf_shifted - K * cf_standard)
            den = 1j * u * K ** (1j * u)
            return np.real(num / den)

        real_val, _ = quad(integrand, 0.0, 100.0, limit=200, epsabs=1e-4, epsrel=1e-4)
        price = np.real((S0 * np.exp(-q * T) - K * np.exp(-r * T)) / 2 + real_val / np.pi)
        return float(max(price, 0.0))

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
        if T <= 0:
            call = float(max(S0 - K, 0.0))
            return call, float(max(K - S0, 0.0))

        ln_k = math.log(K)
        fwd_cf = charfunc(-1j)  # E[S_T]; equals S0 exp((r - q) T) for a valid model

        def integrand_p2(u):
            return np.real(np.exp(-1j * u * ln_k) * charfunc(u) / (1j * u))

        def integrand_p1(u):
            return np.real(np.exp(-1j * u * ln_k) * charfunc(u - 1j) / (1j * u * fwd_cf))

        p2 = 0.5 + quad(integrand_p2, 0.0, 100.0, limit=200,
                        epsabs=1e-4, epsrel=1e-4)[0] / np.pi
        p1 = 0.5 + quad(integrand_p1, 0.0, 100.0, limit=200,
                        epsabs=1e-4, epsrel=1e-4)[0] / np.pi

        disc_spot = S0 * np.exp(-q * T)
        disc_strike = K * np.exp(-r * T)
        call = disc_spot * p1 - disc_strike * p2
        put = disc_strike * (1.0 - p2) - disc_spot * (1.0 - p1)
        return float(call), float(put)

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
        f_re = np.real(phi_vals * np.exp(-1j * u_k * a))
        weights = np.ones_like(u_k)
        weights[0] = 0.5
        fac = weights * f_re  # shape (N,)

        k_arr = np.atleast_1d(np.asarray(K_array, dtype=float))
        c = np.clip(np.log(k_arr), a, b)[:, None]  # (M, 1)
        d = b
        uk = u_k[None, :]  # (1, N)

        arg_c = uk * (c - a)           # (M, N)
        arg_d = uk * (d - a)           # (1, N)
        e_c = np.exp(c)                # (M, 1)
        e_d = math.exp(d)              # scalar
        inv = 1.0 / (1.0 + uk ** 2)    # (1, N)

        chi = inv * (
            np.cos(arg_d) * e_d - np.cos(arg_c) * e_c
            + uk * (np.sin(arg_d) * e_d - np.sin(arg_c) * e_c)
        )

        uk_safe = uk.copy()
        uk_safe[0, 0] = 1.0
        psi = (np.sin(arg_d) - np.sin(arg_c)) / uk_safe
        psi[:, 0] = (d - c[:, 0])  # k = 0 column

        u_payoff = (2.0 / (b - a)) * (chi - k_arr[:, None] * psi)  # (M, N)
        prices = math.exp(-r * T) * (u_payoff * fac[None, :]).sum(axis=1)
        return np.maximum(prices, 0.0)

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

        # Truncation range from numerically estimated cumulants of log S_T.
        h = 0.05
        log_phi = np.log(phi(h))
        c1 = log_phi.imag / h
        c2 = max(-2.0 * log_phi.real / h ** 2, 1e-6)
        width = L * math.sqrt(c2)
        a, b = c1 - width, c1 + width

        u_k = np.arange(N) * np.pi / (b - a)
        phi_vals = phi(u_k)
        return BatesHawkesExact._cos_call_prices(phi_vals, u_k, a, b, k_arr, r, T)

    # --- H. CONSTANT-VOL CALIBRATION -----------------------------------------
    # Calibrating the exact model is intentionally constrained to the stationary
    # region: the branching ratio alpha / beta is bounded below one and the
    # initial intensity is tied to the baseline (lambda0 = lambda_bar). This is
    # the cheap, identifiable first pass recommended in the paper digest; the
    # initial intensity can be unlocked once a stable fit exists. Exact pricing
    # is semi-analytic (one ODE solve per Fourier node), so calibration is much
    # heavier than the Bates proxy and is meant to be seeded from it.

    # params layout: [sigma, lambda_bar, alpha, beta, mu_J, sigma_J]
    _MAX_BRANCHING = 0.98

    @staticmethod
    def hawkes_objective_constvol(params, df_market, S0, q=0.0, n_steps=None,
                                  cos_N=256):
        """Vega-weighted calibration objective for the constant-vol exact model.

        ``params = [sigma, lambda_bar, alpha, beta, mu_J, sigma_J]`` with the
        initial intensity tied to the baseline (``lambda0 = lambda_bar``).
        Options are priced with the COS method grouped by maturity, so the jump
        ODE is solved once per maturity. Returns a large constant outside the
        admissible stationary region.
        """
        sigma, lambda_bar, alpha, beta, mu_J, sigma_J = params

        if sigma <= 0 or lambda_bar <= 0 or beta <= 0 or sigma_J <= 0:
            return 1e8
        if alpha < 0 or alpha >= beta:
            return 1e8
        if alpha / beta >= BatesHawkesExact._MAX_BRANCHING:
            return 1e8

        error = 0.0
        count = 0
        for maturity, group in df_market.groupby("T"):
            strikes = group["K"].to_numpy(dtype=float)
            rate = float(group["rate"].iloc[0])
            model_prices = BatesHawkesExact.hawkes_price_constvol_cos(
                S0, strikes, float(maturity), sigma,
                lambda_bar, lambda_bar, alpha, beta, mu_J, sigma_J,
                rate, q, N=cos_N, n_steps=n_steps,
            )
            market_prices = group["price"].to_numpy(dtype=float)
            safe_vega = np.maximum(group["vega"].to_numpy(dtype=float), 1e-4)
            error += float(np.sum(((model_prices - market_prices) / safe_vega) ** 2))
            count += len(group)

        branching = alpha / beta
        # Mild regularisation away from the near-critical regime, as in the proxy.
        penalty = 0.01 * branching ** 2 / max(1.0 - branching, 1e-4)
        return error / count + penalty

    @staticmethod
    def calibrate_hawkes_exact_constvol(df_market, S0, q=0.0, maxiter=30,
                                        popsize=8, n_steps=None, seed=None):
        """Calibrate the constant-vol exact-Hawkes model (heavy; seed from Bates).

        Two-stage Differential Evolution + SLSQP, matching the other models. The
        branching ratio is bounded by a linear constraint ``beta > alpha``.
        """
        bounds = [
            (1e-2, 2.0),    # sigma
            (1e-2, 5.0),    # lambda_bar
            (0.0, 5.0),     # alpha
            (1e-2, 8.0),    # beta
            (-0.5, 0.5),    # mu_J
            (1e-3, 0.6),    # sigma_J
        ]
        constraints = (
            {"type": "ineq", "fun": lambda x: x[3] - x[2] - 1e-4},  # beta > alpha
        )

        print("[INFO] Starting Global Calibration for exact constant-vol Hawkes...")
        result_global = differential_evolution(
            BatesHawkesExact.hawkes_objective_constvol,
            bounds=bounds,
            args=(df_market, S0, q, n_steps),
            maxiter=maxiter,
            popsize=popsize,
            tol=1e-3,
            polish=False,
            seed=seed,
            disp=True,
        )

        print("\n[INFO] Global candidate found. Starting Local Refinement (SLSQP)...")
        result_local = minimize(
            BatesHawkesExact.hawkes_objective_constvol,
            x0=result_global.x,
            args=(df_market, S0, q, n_steps),
            method="SLSQP",
            bounds=bounds,
            constraints=constraints,
            options={"ftol": 1e-6, "maxiter": 60},
        )

        sigma, lambda_bar, alpha, beta, mu_J, sigma_J = result_local.x
        print("\n===================================================")
        print("  OPTIMAL EXACT CONSTANT-VOL HAWKES PARAMETERS")
        print("===================================================")
        labels = ["sigma", "lambda_bar", "alpha", "beta", "mu_J", "sigma_J"]
        for label, value in zip(labels, result_local.x):
            print(f"{label:11s}: {value:.6f}")
        print(f"branching  : {alpha / beta:.6f}  (lambda0 fixed to lambda_bar)")
        print("===================================================")
        return result_local.x
