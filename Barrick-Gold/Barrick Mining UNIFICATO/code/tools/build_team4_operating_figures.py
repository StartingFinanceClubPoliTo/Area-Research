"""Build the current-data Team 4 operating figures inside one thesis run."""

from __future__ import annotations

import argparse
import json
from pathlib import Path
import sys


ROOT = Path(__file__).resolve().parents[1]
SRC = ROOT / "src"
if str(SRC) not in sys.path:
    sys.path.insert(0, str(SRC))

from barrick_unified.team4_operating_figures import build_team4_operating_figures  # noqa: E402


def main() -> None:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--run-id", required=True)
    args = parser.parse_args()
    result = build_team4_operating_figures(
        mine_csv=ROOT / "data/processed/team4/barrick_mine_operating_q1_q2_2026.csv",
        totals_csv=ROOT / "data/processed/team4/barrick_total_gold_q1_q2_2026.csv",
        source_manifest=ROOT / "data/manifests/team4/barrick_operating_q1_q2_2026_manifest.json",
        valuation_config=ROOT / "config/provisional_valuation_20260827_team4_separated.json",
        output_dir=ROOT / "outputs/thesis" / args.run_id / "team4_operations",
        run_id=args.run_id,
    )
    print(json.dumps({"run_id": args.run_id, "manifest": result["manifest_path"]}, indent=2))


if __name__ == "__main__":
    main()
