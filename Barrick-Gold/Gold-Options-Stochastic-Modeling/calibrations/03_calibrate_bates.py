from common import DATA_DIR, load_chebyshev_dataset, write_json
from Bates import Bates


def main():
    df, s0, q = load_chebyshev_dataset()
    params = Bates.calibrate_bates(df, s0, q=q)

    write_json(
        DATA_DIR / "bates_calibrated_params.json",
        {
            "model": "Bates",
            "parameters": {
                "v0": float(params[0]),
                "kappa": float(params[1]),
                "theta": float(params[2]),
                "sigma": float(params[3]),
                "rho": float(params[4]),
                "lambda": float(params[5]),
                "mu_J": float(params[6]),
                "sigma_J": float(params[7]),
            },
        },
    )


if __name__ == "__main__":
    main()
