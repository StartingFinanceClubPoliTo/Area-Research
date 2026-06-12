import json
from pathlib import Path
import sys

import pandas as pd

ROOT = Path(__file__).resolve().parents[1]
DATA_DIR = ROOT / "Data"

if str(ROOT) not in sys.path:
    sys.path.insert(0, str(ROOT))

from BnS import BnS  # noqa: E402


def load_metadata():
    with (DATA_DIR / "gld_chain_wide_meta.json").open("r", encoding="utf-8") as handle:
        return json.load(handle)


def load_chebyshev_dataset():
    """Load the curated Chebyshev dataset and add Black-Scholes vega."""
    metadata = load_metadata()
    df = pd.read_csv(DATA_DIR / "gld_iv_dataset_chebyshev.csv")
    q = 0.0

    df["vega"] = [
        BnS.calculate_bs_vega(
            metadata["S0"],
            row.K,
            row.T,
            row.rate,
            q,
            row.implied_vol,
        )
        for row in df.itertuples(index=False)
    ]
    return df, float(metadata["S0"]), q


def write_json(path, payload):
    path = Path(path)
    path.parent.mkdir(parents=True, exist_ok=True)
    with path.open("w", encoding="utf-8") as handle:
        json.dump(payload, handle, indent=2)
