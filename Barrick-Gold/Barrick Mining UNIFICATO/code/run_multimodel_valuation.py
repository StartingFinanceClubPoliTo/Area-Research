"""CLI for the conditional four-model Barrick valuation experiment."""

from __future__ import annotations

import argparse
import json
from datetime import datetime, timezone
from pathlib import Path
import sys

ROOT = Path(__file__).resolve().parent
SRC = ROOT / "src"
if str(SRC) not in sys.path:
    sys.path.insert(0, str(SRC))

from barrick_unified.multimodel_reporting import write_multimodel_outputs  # noqa: E402
from barrick_unified.multimodel_valuation import (  # noqa: E402
    load_multimodel_inputs,
    run_multimodel_valuation,
)


DEFAULT_CONFIG = ROOT / "config" / "multimodel_valuation_20260826.json"
PROTECTED_V3_RUN_ID = "20260825T143500Z-provisional-v3"


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(
        description=(
            "Compare four Team 8 gold-price layers under common Team 4 "
            "production/cost and Team 5 valuation assumptions."
        )
    )
    parser.add_argument("--config", type=Path, default=DEFAULT_CONFIG)
    parser.add_argument("--run-id", help="New versioned run identifier.")
    parser.add_argument("--output-root", type=Path, default=ROOT)
    return parser.parse_args()


def run(args: argparse.Namespace) -> dict:
    config_path = args.config.resolve()
    payload = json.loads(config_path.read_text(encoding="utf-8"))
    if payload.get("schema_version") != "2.0":
        raise ValueError("multimodel config schema_version must be 2.0")
    run_id = args.run_id or (
        datetime.now(timezone.utc).strftime("%Y%m%dT%H%M%SZ")
        + "-conditional-multimodel-v4"
    )
    if run_id == PROTECTED_V3_RUN_ID:
        raise ValueError("the accepted v3 run id is protected and cannot be overwritten")
    output_root = args.output_root.resolve()
    output_dir = output_root / "outputs" / "valuation" / run_id
    figure_dir = output_root / "figures" / "valuation" / run_id
    manifest_path = (
        output_root
        / "data"
        / "manifests"
        / "valuation"
        / run_id
        / "run_manifest.json"
    )
    occupied = [path for path in (output_dir, figure_dir, manifest_path.parent) if path.exists()]
    if occupied:
        raise FileExistsError(
            "refusing to overwrite an existing run: "
            + ", ".join(str(path) for path in occupied)
        )
    model_run = run_multimodel_valuation(ROOT, payload)
    _, base_config_path = load_multimodel_inputs(ROOT, payload)
    return write_multimodel_outputs(
        run=model_run,
        experiment_config=payload,
        experiment_config_path=config_path,
        base_config_path=base_config_path,
        output_dir=output_dir,
        figure_dir=figure_dir,
        manifest_path=manifest_path,
        project_root=ROOT,
        run_id=run_id,
    )


def main() -> None:
    manifest = run(parse_args())
    print(f"Run: {manifest['run_id']}")
    for row in manifest["model_summaries"]:
        print(
            f"{row['model']}: median ${row['median_value_per_share_usd']:.2f}; "
            f"P(V > $48.39) {row['probability_model_value_exceeds_observed']:.2%}"
        )
    print("Scope: GOLD_PRICE_LAYER_ONLY")
    print("Status: PROVISIONAL_RESEARCH_SENSITIVITY_NOT_TARGET")


if __name__ == "__main__":
    main()
