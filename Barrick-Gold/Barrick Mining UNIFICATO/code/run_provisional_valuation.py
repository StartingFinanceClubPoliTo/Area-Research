"""CLI for the explicitly provisional stochastic Barrick valuation."""

from __future__ import annotations

import argparse
import json
from datetime import datetime, timezone
from pathlib import Path

from barrick_unified.valuation import ValuationInputs, simulate_valuation
from barrick_unified.valuation_reporting import write_valuation_outputs


ROOT = Path(__file__).resolve().parent
DEFAULT_CONFIG = ROOT / "config" / "provisional_valuation_20260825.json"


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(
        description="Run the Barrick provisional stochastic valuation sensitivity."
    )
    parser.add_argument("--config", type=Path, default=DEFAULT_CONFIG)
    parser.add_argument("--run-id", help="Versioned output identifier; defaults to current UTC.")
    parser.add_argument("--observed-price", type=float, help="Override B price in USD/share.")
    parser.add_argument("--observed-price-timestamp", help="Required with --observed-price.")
    parser.add_argument("--observed-price-source", help="Required with --observed-price.")
    parser.add_argument("--output-root", type=Path, default=ROOT)
    return parser.parse_args()


def run(args: argparse.Namespace) -> dict:
    config_path = args.config.resolve()
    payload = json.loads(config_path.read_text(encoding="utf-8"))
    overrides = (
        args.observed_price,
        args.observed_price_timestamp,
        args.observed_price_source,
    )
    if any(value is not None for value in overrides):
        if not all(value is not None for value in overrides):
            raise ValueError(
                "--observed-price, --observed-price-timestamp and --observed-price-source must be supplied together"
            )
        payload["market_price"] = {
            "price_usd": args.observed_price,
            "timestamp_utc": args.observed_price_timestamp,
            "source": args.observed_price_source,
        }

    inputs = ValuationInputs.from_dict(payload)
    result = simulate_valuation(inputs)
    run_id = args.run_id or datetime.now(timezone.utc).strftime("%Y%m%dT%H%M%SZ")
    output_root = args.output_root.resolve()
    manifest = write_valuation_outputs(
        inputs=inputs,
        result=result,
        output_dir=output_root / "outputs" / "valuation" / run_id,
        figure_dir=output_root / "figures" / "valuation" / run_id,
        manifest_path=output_root / "data" / "manifests" / "valuation" / run_id / "run_manifest.json",
        config_path=config_path,
        project_root=ROOT,
        run_id=run_id,
    )
    return manifest


def main() -> None:
    manifest = run(parse_args())
    print(f"Run: {manifest['run_id']}")
    print(
        "Probability provisional model value exceeds observed close: "
        f"{manifest['probability_model_value_exceeds_observed_price']:.2%}"
    )
    print("Status: PROVISIONAL_RESEARCH_SENSITIVITY_NOT_TARGET")


if __name__ == "__main__":
    main()
