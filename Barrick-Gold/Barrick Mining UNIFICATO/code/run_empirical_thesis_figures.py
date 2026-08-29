"""Generate all Chapter 7 empirical figures from a versioned LSE raw run."""

from __future__ import annotations

import argparse
import json
import sys
from pathlib import Path


ROOT = Path(__file__).resolve().parent
sys.path.insert(0, str(ROOT / "src"))

from barrick_unified.empirical_figures import generate_empirical_figures  # noqa: E402


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--data-run-id", required=True)
    parser.add_argument("--run-id", required=True)
    parser.add_argument("--rolling-window", type=int, default=252)
    return parser.parse_args()


def main() -> int:
    args = parse_args()
    raw_path = ROOT / "data" / "raw" / "lse_local" / args.data_run_id / "empirical_figure_inputs.json"
    output_dir = ROOT / "outputs" / "thesis" / args.run_id / "team7_empirical"
    if not raw_path.is_file():
        raise FileNotFoundError(raw_path)
    manifest = generate_empirical_figures(raw_path, output_dir, args.rolling_window)
    print(json.dumps({
        "run_id": args.run_id,
        "output": output_dir.relative_to(ROOT).as_posix(),
        "figure_count": manifest["figure_count"],
        "sample": manifest["sample"],
    }, indent=2))
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
