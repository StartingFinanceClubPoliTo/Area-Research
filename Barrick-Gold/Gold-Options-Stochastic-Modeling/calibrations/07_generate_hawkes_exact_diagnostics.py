"""
Demonstration of the exact (event-dependent) Bates-Hawkes pricer.

This script is the publication-facing example for the exact Hawkes engine. It

1. prices a single European call with the exact model, the stationary-intensity
   Bates-Hawkes proxy, and the plain Bates model at the same mean intensity;
2. builds an implied-volatility smile and compares
       Hawkes exact  = self-exciting intensity solved via the affine CF,
       Hawkes proxy  = Bates with an equivalent constant intensity;
3. shows how the smile fans out as the branching ratio alpha / beta grows;
4. cross-checks the exact constant-vol price against a Monte-Carlo simulation
   using Ogata thinning (reusing ``Hawkes.simulate_exponential``).

Run from the repository root:

    python calibrations/07_generate_hawkes_exact_diagnostics.py

Outputs ``Data/hawkes_exact_vs_proxy.png`` and
``Data/hawkes_exact_smile.csv``.
"""

import os
import sys
from pathlib import Path

import matplotlib.pyplot as plt
import numpy as np
import pandas as pd

HERE = os.path.dirname(os.path.abspath(__file__))
ROOT = os.path.dirname(HERE)
DATA_DIR = Path(ROOT) / "Data"
for _p in (HERE, ROOT):
    if _p not in sys.path:
        sys.path.insert(0, _p)

from BatesHawkesExact import BatesHawkesExact  # noqa: E402
from BatesHawkes import BatesHawkes  # noqa: E402
from Bates import Bates  # noqa: E402
from BnS import BnS  # noqa: E402
from Hawkes import Hawkes  # noqa: E402


# --- Market and model parameters --------------------------------------------

S0, R, Q, T = 100.0, 0.03, 0.0, 0.5
V0, KAPPA, THETA, SIGMA_V, RHO = 0.04, 1.5, 0.04, 0.30, -0.6
MU_J, SIGMA_J = -0.08, 0.13
LAMBDA_BAR, BETA = 0.60, 1.60
BRANCHING_MAIN = 0.5
SIGMA_CONST = 0.20  # constant-vol diffusion for the Monte-Carlo cross-check

STRIKES = np.linspace(80.0, 125.0, 13)


def stationary_mean(lambda_bar, alpha, beta):
    """Long-run mean intensity E[lambda_inf] = lambda_bar / (1 - alpha / beta)."""
    return lambda_bar / (1.0 - alpha / beta)


# --- 1. Single-option comparison --------------------------------------------

def single_option_comparison():
    alpha = BRANCHING_MAIN * BETA
    mean_int = stationary_mean(LAMBDA_BAR, alpha, BETA)
    K = 100.0

    exact = BatesHawkesExact.hawkes_price_fast(
        S0, K, T, V0, KAPPA, THETA, SIGMA_V, RHO,
        mean_int, LAMBDA_BAR, alpha, BETA, MU_J, SIGMA_J, R, Q,
    )
    proxy = BatesHawkes.price_proxy_fast(
        S0, K, T, V0, KAPPA, THETA, SIGMA_V, RHO,
        LAMBDA_BAR, alpha, BETA, MU_J, SIGMA_J, R, Q,
    )
    bates_mean = Bates.bates_price_fast(
        S0, K, T, V0, KAPPA, THETA, SIGMA_V, RHO, mean_int, MU_J, SIGMA_J, R, Q,
    )
    print("--- Single ATM call (K=100, T=0.5), branching alpha/beta = 0.50 ---")
    print(f"  stationary mean intensity      : {mean_int:.4f}")
    print(f"  Hawkes exact (self-exciting)   : {exact:.4f}")
    print(f"  Bates-Hawkes proxy (lambda_eff): {proxy:.4f}")
    print(f"  Bates at mean intensity        : {bates_mean:.4f}")
    print(f"  exact - proxy                  : {exact - proxy:+.4f}")


# --- 2 & 3. Implied-volatility smiles ---------------------------------------

def _iv(price, K):
    return BnS.implied_vol_call(price, S0, K, T, R, Q)


def build_smiles():
    alpha = BRANCHING_MAIN * BETA
    mean_int = stationary_mean(LAMBDA_BAR, alpha, BETA)

    rows = []
    for K in STRIKES:
        exact = BatesHawkesExact.hawkes_price_fast(
            S0, K, T, V0, KAPPA, THETA, SIGMA_V, RHO,
            mean_int, LAMBDA_BAR, alpha, BETA, MU_J, SIGMA_J, R, Q,
        )
        proxy = BatesHawkes.price_proxy_fast(
            S0, K, T, V0, KAPPA, THETA, SIGMA_V, RHO,
            LAMBDA_BAR, alpha, BETA, MU_J, SIGMA_J, R, Q,
        )
        rows.append({
            "K": K,
            "iv_exact": _iv(exact, K),
            "iv_proxy": _iv(proxy, K),
        })
    smile = pd.DataFrame(rows)

    # Sensitivity: exact-model smile for a range of branching ratios.
    branchings = [0.0, 0.4, 0.8]
    sensitivity = {"K": STRIKES}
    for b in branchings:
        a = b * BETA
        m = stationary_mean(LAMBDA_BAR, a, BETA) if b < 1.0 else LAMBDA_BAR
        ivs = []
        for K in STRIKES:
            price = BatesHawkesExact.hawkes_price_fast(
                S0, K, T, V0, KAPPA, THETA, SIGMA_V, RHO,
                m, LAMBDA_BAR, a, BETA, MU_J, SIGMA_J, R, Q,
            )
            ivs.append(_iv(price, K))
        sensitivity[f"iv_b{b:.1f}"] = ivs

    return smile, pd.DataFrame(sensitivity), branchings


# --- 4. Monte-Carlo cross-check (constant-vol, Ogata thinning) --------------

def mc_price_constvol(K, sigma, lambda_bar, alpha, beta, n_paths=15000, seed=20260613):
    """Monte-Carlo European call under the constant-vol exact-Hawkes model.

    The intensity starts at the baseline (lambda0 = lambda_bar), so event times
    come straight from ``Hawkes.simulate_exponential``. Returns (price, stderr).
    """
    rng = np.random.default_rng(seed)
    kappa_J = BatesHawkesExact.kappa_J(MU_J, sigma_J=SIGMA_J)
    payoffs = np.empty(n_paths)
    for i in range(n_paths):
        events = Hawkes.simulate_exponential(
            lambda_bar, alpha, beta, T, seed=int(rng.integers(1, 2 ** 31 - 1))
        )
        n = events.size
        integ = lambda_bar * T
        if n:
            integ += (alpha / beta) * np.sum(1.0 - np.exp(-beta * (T - events)))
        jump_sum = float(np.sum(rng.normal(MU_J, SIGMA_J, size=n))) if n else 0.0
        z = rng.standard_normal()
        log_st = (
            np.log(S0)
            + (R - Q - 0.5 * sigma ** 2) * T
            + sigma * np.sqrt(T) * z
            - kappa_J * integ
            + jump_sum
        )
        payoffs[i] = max(np.exp(log_st) - K, 0.0)
    disc = np.exp(-R * T)
    price = disc * payoffs.mean()
    stderr = disc * payoffs.std(ddof=1) / np.sqrt(n_paths)
    return price, stderr


def monte_carlo_check():
    alpha = 0.8
    print("\n--- Monte-Carlo cross-check (constant-vol, sigma=0.20, lambda0=lambda_bar) ---")
    print(f"  branching alpha/beta = {alpha / BETA:.2f}, paths = 15000")
    for K in (90.0, 100.0, 110.0):
        fourier = BatesHawkesExact.hawkes_price_constvol_fast(
            S0, K, T, SIGMA_CONST, LAMBDA_BAR, LAMBDA_BAR, alpha, BETA, MU_J, SIGMA_J, R, Q,
        )
        mc, se = mc_price_constvol(K, SIGMA_CONST, LAMBDA_BAR, alpha, BETA)
        flag = "OK" if abs(fourier - mc) <= 3.0 * se else "CHECK"
        print(f"  K={K:5.0f}: Fourier={fourier:7.4f}  MC={mc:7.4f} +/- {1.96 * se:.4f}  [{flag}]")


# --- Figure ------------------------------------------------------------------

def make_figure(smile, sensitivity, branchings):
    fig, axes = plt.subplots(1, 2, figsize=(13.0, 5.2))

    axes[0].plot(smile["K"], 100 * smile["iv_proxy"], "o--", color="#365f91",
                 linewidth=2.0, label="Hawkes proxy (constant intensity)")
    axes[0].plot(smile["K"], 100 * smile["iv_exact"], "o-", color="#b73f32",
                 linewidth=2.2, label="Hawkes exact (self-exciting)")
    axes[0].axvline(S0, color="#888888", linestyle=":", linewidth=1.2)
    axes[0].set_xlabel("Strike K")
    axes[0].set_ylabel("Black-Scholes implied volatility (%)")
    axes[0].set_title("Exact vs proxy smile (alpha/beta = 0.5, T = 0.5)")
    axes[0].legend(loc="upper right")
    axes[0].grid(alpha=0.25)

    colors = ["#2f2f2f", "#f08c00", "#b73f32"]
    for b, color in zip(branchings, colors):
        axes[1].plot(sensitivity["K"], 100 * np.array(sensitivity[f"iv_b{b:.1f}"]),
                     "o-", color=color, linewidth=2.0, label=f"alpha/beta = {b:.1f}")
    axes[1].axvline(S0, color="#888888", linestyle=":", linewidth=1.2)
    axes[1].set_xlabel("Strike K")
    axes[1].set_ylabel("Black-Scholes implied volatility (%)")
    axes[1].set_title("Self-excitation sensitivity (exact model)")
    axes[1].legend(loc="upper right")
    axes[1].grid(alpha=0.25)

    fig.tight_layout()
    DATA_DIR.mkdir(parents=True, exist_ok=True)
    out_png = DATA_DIR / "hawkes_exact_vs_proxy.png"
    fig.savefig(out_png, dpi=200)
    plt.close(fig)
    return out_png


def main():
    single_option_comparison()
    smile, sensitivity, branchings = build_smiles()
    monte_carlo_check()

    DATA_DIR.mkdir(parents=True, exist_ok=True)
    out_csv = DATA_DIR / "hawkes_exact_smile.csv"
    smile.to_csv(out_csv, index=False)
    out_png = make_figure(smile, sensitivity, branchings)
    print(f"\n[INFO] Wrote {out_png}")
    print(f"[INFO] Wrote {out_csv}")


if __name__ == "__main__":
    main()
