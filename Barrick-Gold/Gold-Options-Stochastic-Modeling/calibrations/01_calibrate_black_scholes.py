from pathlib import Path

import pandas as pd

from common import DATA_DIR, load_chebyshev_dataset


def main():
    df, _, _ = load_chebyshev_dataset()

    summary = (
        df.assign(moneyness=df["K"] / df["K"].median())
        .groupby("expiry", as_index=False)
        .agg(
            n_options=("K", "size"),
            mean_implied_vol=("implied_vol", "mean"),
            median_implied_vol=("implied_vol", "median"),
            mean_vega=("vega", "mean"),
        )
    )

    output_path = DATA_DIR / "black_scholes_calibration_summary.csv"
    summary.to_csv(output_path, index=False)
    print(f"[INFO] Wrote {output_path.relative_to(Path.cwd())}")


if __name__ == "__main__":
    main()
