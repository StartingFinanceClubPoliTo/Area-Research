"""Small, deterministic benchmark for the restructured pricing path."""

from pathlib import Path
from time import perf_counter
import sys

import numpy as np

PROJECT_ROOT = Path(__file__).resolve().parents[1]
if str(PROJECT_ROOT) not in sys.path:
    sys.path.insert(0, str(PROJECT_ROOT))

from Bates import Bates  # noqa: E402
from BatesHawkes import BatesHawkes  # noqa: E402


def timed(callable_, repeats):
    started_at = perf_counter()
    value = None
    for _ in range(repeats):
        value = callable_()
    return value, (perf_counter() - started_at) / repeats


def main():
    spot, maturity, rate = 100.0, 0.5, 0.03
    strikes = np.linspace(70.0, 130.0, 61)
    params = (0.04, 1.5, 0.04, 0.30, -0.60, 0.50, -0.08, 0.13)

    reference, scalar_seconds = timed(
        lambda: np.array(
            [
                Bates.bates_price_fast(
                    spot, strike, maturity, *params, rate
                )
                for strike in strikes
            ]
        ),
        repeats=3,
    )
    batch, cos_seconds = timed(
        lambda: Bates.bates_prices_cos(
            spot, strikes, maturity, *params, rate, N=256
        ),
        repeats=20,
    )
    proxy_atm = BatesHawkes.price_proxy_fast(
        spot,
        100.0,
        maturity,
        *params[:5],
        0.25,
        0.50,
        1.0,
        *params[6:],
        rate,
    )

    print(f"strikes={strikes.size}")
    print(f"max_abs_error={np.max(np.abs(reference - batch)):.12g}")
    print(f"scalar_seconds={scalar_seconds:.9f}")
    print(f"cos_seconds={cos_seconds:.9f}")
    print(f"speedup={scalar_seconds / cos_seconds:.2f}x")
    print(f"proxy_atm={proxy_atm:.12f}")
    print(f"bates_atm={reference[strikes.tolist().index(100.0)]:.12f}")


if __name__ == "__main__":
    main()
