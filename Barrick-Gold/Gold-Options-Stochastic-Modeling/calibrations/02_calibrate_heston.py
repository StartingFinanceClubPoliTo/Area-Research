from common import DATA_DIR, load_chebyshev_dataset, write_json
from Heston import Heston


def main():
    df, s0, q = load_chebyshev_dataset()
    params = Heston.calibrate_heston(df, s0, q=q)

    write_json(
        DATA_DIR / "heston_calibrated_params.json",
        {
            "model": "Heston",
            "parameters": {
                "v0": float(params[0]),
                "kappa": float(params[1]),
                "theta": float(params[2]),
                "sigma": float(params[3]),
                "rho": float(params[4]),
            },
        },
    )


if __name__ == "__main__":
    main()
