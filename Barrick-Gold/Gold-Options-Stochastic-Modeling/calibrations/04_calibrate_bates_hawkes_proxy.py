import argparse

from common import DATA_DIR, load_chebyshev_dataset, write_json
from BatesHawkes import BatesHawkes


def main():
    parser = argparse.ArgumentParser(description="Calibrate the Bates-Hawkes stationary-intensity proxy.")
    parser.add_argument("--maxiter", type=int, default=80)
    parser.add_argument("--popsize", type=int, default=8)
    args = parser.parse_args()

    df, s0, q = load_chebyshev_dataset()
    params = BatesHawkes.calibrate_bates_hawkes_proxy(
        df,
        s0,
        q=q,
        maxiter=args.maxiter,
        popsize=args.popsize,
    )

    lambda0, alpha, beta = params[5], params[6], params[7]
    write_json(
        DATA_DIR / "bates_hawkes_proxy_params.json",
        {
            "model": "Bates-Hawkes stationary-intensity proxy",
            "parameters": {
                "v0": float(params[0]),
                "kappa": float(params[1]),
                "theta": float(params[2]),
                "sigma": float(params[3]),
                "rho": float(params[4]),
                "lambda0": float(lambda0),
                "alpha": float(alpha),
                "beta": float(beta),
                "branching_ratio": float(alpha / beta),
                "lambda_eff": float(BatesHawkes.effective_intensity(lambda0, alpha, beta)),
                "mu_J": float(params[8]),
                "sigma_J": float(params[9]),
            },
            "note": (
                "This is a stationary-intensity proxy, not a full event-dependent "
                "Bates-Hawkes pricing engine."
            ),
        },
    )


if __name__ == "__main__":
    main()
