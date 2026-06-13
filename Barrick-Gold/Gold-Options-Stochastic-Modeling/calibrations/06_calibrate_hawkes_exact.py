"""
Calibrate the exact constant-volatility Bates-Hawkes model to the GLD
implied-volatility surface.

This reuses the curated Chebyshev dataset and vega weighting shared by the
other calibration scripts. Pricing uses the COS method (one jump-ODE solve per
maturity), which makes the exact-Hawkes global search tractable. The branching
ratio alpha / beta is bounded below one, and the initial intensity is tied to
the baseline (lambda0 = lambda_bar) for identifiability.

Run from the repository root:

    python calibrations/06_calibrate_hawkes_exact.py --maxiter 35 --popsize 8 --seed 20260613
"""

import argparse
import os
import sys

HERE = os.path.dirname(os.path.abspath(__file__))
ROOT = os.path.dirname(HERE)
for _p in (HERE, ROOT, os.path.join(ROOT, "calibrations")):
    if _p not in sys.path:
        sys.path.insert(0, _p)

from common import DATA_DIR, load_chebyshev_dataset, write_json  # noqa: E402
from BatesHawkesExact import BatesHawkesExact  # noqa: E402


def main():
    parser = argparse.ArgumentParser(
        description="Calibrate the exact constant-vol Bates-Hawkes model."
    )
    parser.add_argument("--maxiter", type=int, default=35)
    parser.add_argument("--popsize", type=int, default=8)
    parser.add_argument("--seed", type=int, default=20260613)
    args = parser.parse_args()

    df, s0, q = load_chebyshev_dataset()
    print(f"[INFO] Dataset: {len(df)} options across {df['T'].nunique()} maturities, S0={s0}")

    params = BatesHawkesExact.calibrate_hawkes_exact_constvol(
        df, s0, q=q, maxiter=args.maxiter, popsize=args.popsize, seed=args.seed
    )

    sigma, lambda_bar, alpha, beta, mu_J, sigma_J = (float(x) for x in params)
    final_obj = BatesHawkesExact.hawkes_objective_constvol(params, df, s0, q)

    payload = {
        "model": "Exact constant-volatility Bates-Hawkes",
        "parameters": {
            "sigma": sigma,
            "lambda_bar": lambda_bar,
            "lambda0": lambda_bar,  # tied to baseline
            "alpha": alpha,
            "beta": beta,
            "branching_ratio": alpha / beta,
            "mu_J": mu_J,
            "sigma_J": sigma_J,
        },
        "final_objective": float(final_obj),
        "note": (
            "Exact event-dependent Hawkes calibration (COS pricing). "
            "lambda0 tied to lambda_bar; branching ratio alpha/beta < 1 enforced."
        ),
    }
    out_path = DATA_DIR / "hawkes_exact_constvol_params.json"
    write_json(out_path, payload)
    print(f"\n[INFO] Wrote {out_path}")
    print(f"[INFO] Final vega-weighted objective: {final_obj:.6f}")


if __name__ == "__main__":
    main()
