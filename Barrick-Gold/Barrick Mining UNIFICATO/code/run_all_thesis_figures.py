"""Regenerate and optionally promote every analytical thesis figure.

This is the single branch-local entry point.  It never reads PNG files from a
standalone article as inputs.  Current licensed LSE rows must already exist in
the Git-ignored ``data/raw/lse_local`` tree (or be fetched with the dedicated
acquisition scripts); frozen validation charts are rendered from committed
aggregate tables and labelled as frozen by the provenance audit.
"""

from __future__ import annotations

import argparse
import os
import shutil
import subprocess
import sys
from pathlib import Path


ROOT = Path(__file__).resolve().parent
UNIFIED = ROOT.parent
sys.path.insert(0, str(ROOT / "tools"))
from audit_thesis_figure_provenance import catalog  # noqa: E402


def execute(arguments: list[str]) -> None:
    environment = os.environ.copy()
    environment["PYTHONDONTWRITEBYTECODE"] = "1"
    subprocess.run([sys.executable, *arguments], cwd=ROOT, env=environment, check=True)


def promote(figure_run_id: str, market_run_id: str) -> None:
    mapping = catalog(figure_run_id, market_run_id)
    for image_name, item in mapping.items():
        source = Path(item["output"])
        destination = UNIFIED / "Overleaf" / image_name
        if not source.is_file():
            raise FileNotFoundError(source)
        destination.parent.mkdir(parents=True, exist_ok=True)
        shutil.copy2(source, destination)


def main() -> None:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--run-id", required=True, help="Fresh immutable thesis-figure run ID.")
    parser.add_argument("--market-run-id", required=True, help="Versioned LSE market/option run ID.")
    parser.add_argument("--empirical-data-run-id", required=True, help="Versioned LSE macro-panel run ID.")
    parser.add_argument("--promote-overleaf", action="store_true")
    args = parser.parse_args()

    market_raw = ROOT / "data/raw/lse_local" / args.market_run_id / "gld_market_inputs.json"
    empirical_raw = ROOT / "data/raw/lse_local" / args.empirical_data_run_id / "empirical_figure_inputs.json"
    for path in (market_raw, empirical_raw):
        if not path.is_file():
            raise FileNotFoundError(
                f"Required local-only input is missing: {path}. Run the branch acquisition script first."
            )

    execute(["run_empirical_thesis_figures.py", "--data-run-id", args.empirical_data_run_id, "--run-id", args.run_id])
    execute(["tools/build_team8_thesis_figures.py", "--raw-input", str(market_raw), "--run-id", args.run_id])
    execute(["tools/build_team4_operating_figures.py", "--run-id", args.run_id])
    execute(
        [
            "run_refactored_thesis_figures.py",
            "--run-id", args.run_id,
            "--team8-run-id", args.run_id,
            "--market-run-id", args.market_run_id,
        ]
    )
    execute(["tools/build_team8_oos_r2_figure.py"])

    if args.promote_overleaf:
        promote(args.run_id, args.market_run_id)

    audit_dir = ROOT / "outputs/thesis" / args.run_id / "provenance"
    execute(
        [
            "tools/audit_thesis_figure_provenance.py",
            "--figure-run-id", args.run_id,
            "--market-run-id", args.market_run_id,
            "--output-dir", str(audit_dir),
            "--authoritative", str(ROOT / "data/manifests/figures/AUTHORITATIVE_THESIS_FIGURE_PROVENANCE.json"),
        ]
    )
    print(f"Completed autonomous thesis-figure run: {args.run_id}")
    print(f"Promoted to Overleaf: {args.promote_overleaf}")


if __name__ == "__main__":
    main()
