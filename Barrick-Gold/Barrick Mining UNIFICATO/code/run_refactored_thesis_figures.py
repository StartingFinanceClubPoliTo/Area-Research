"""Run the refactored Barrick pipeline and build the accepted thesis figures."""

from __future__ import annotations

import argparse
import json
from pathlib import Path
import sys

ROOT = Path(__file__).resolve().parent
SRC = ROOT / "src"
if str(SRC) not in sys.path:
    sys.path.insert(0, str(SRC))

from barrick_unified.refactored import RefactoredBarrickPipeline  # noqa: E402
from barrick_unified.refactored.reporting import ThesisFigureService  # noqa: E402


def main() -> None:
    parser = argparse.ArgumentParser()
    parser.add_argument("--config", type=Path, default=ROOT / "config" / "multimodel_valuation_20260827_team4_separated.json")
    parser.add_argument("--run-id", required=True)
    parser.add_argument(
        "--team8-run-id",
        help="Current Team 8 calibration run. Defaults to --run-id for an autonomous full run.",
    )
    parser.add_argument(
        "--market-run-id",
        required=True,
        help="Versioned LSE market run used by the same autonomous figure run.",
    )
    args = parser.parse_args()
    config_path = args.config.resolve()
    config = json.loads(config_path.read_text(encoding="utf-8"))
    team8_run_id = args.team8_run_id or args.run_id
    team8_dir = ROOT / "outputs" / "thesis" / team8_run_id / "team8_empirical"
    current_files = {
        "black_scholes": "black_scholes_calibrated_params.json",
        "heston": "heston_calibrated_params.json",
        "bates_poisson": "bates_calibrated_params.json",
        "full_bates_hawkes": "bates_hawkes_calibrated_params.json",
    }
    layer = config["gold_price_layer"]
    layer["calibration_manifest"] = str(
        (team8_dir / "lse_publication_manifest.json").relative_to(ROOT)
    ).replace("\\", "/")
    layer["parameter_files"] = {
        model_id: str((team8_dir / filename).relative_to(ROOT)).replace("\\", "/")
        for model_id, filename in current_files.items()
    }
    layer["nss_curve_file"] = str(
        (team8_dir / "usd_treasury_nss_fit.json").relative_to(ROOT)
    ).replace("\\", "/")
    current_snapshot = json.loads(
        (team8_dir / "lse_publication_manifest.json").read_text(encoding="utf-8")
    )
    layer["calibration_snapshot"]["surface_as_of_utc"] = current_snapshot["as_of_utc"]
    layer["calibration_snapshot"]["treasury_curve_date"] = current_snapshot[
        "risk_free_rate_curve_date"
    ]
    config["asynchronous_dates"]["team8_surface"] = current_snapshot["as_of_utc"]
    config["asynchronous_dates"]["team8_treasury_curve"] = current_snapshot[
        "risk_free_rate_curve_date"
    ]
    base_path = ROOT / config["base_valuation_config"]
    run = RefactoredBarrickPipeline().run(ROOT, config)
    output = ROOT / "outputs" / "thesis" / args.run_id
    code_paths = sorted((ROOT / "src" / "barrick_unified" / "refactored").rglob("*.py")) + [Path(__file__).resolve()]
    input_paths = [
        config_path,
        base_path,
        ROOT / "data" / "manifests" / args.market_run_id / "run_manifest.json",
        team8_dir / "lse_publication_manifest.json",
        team8_dir / "usd_treasury_nss_fit.json",
        ROOT
        / "data"
        / "manifests"
        / "rates"
        / "lse_us10y_retry_20260828T143000Z-us10y-retry.json",
        ROOT / "data" / "manifests" / "team4" / "barrick_operating_q1_q2_2026_manifest.json",
    ]
    manifest = ThesisFigureService().build(
        run=run,
        output_dir=output / "figures",
        run_id=args.run_id,
        input_paths=input_paths,
        code_paths=code_paths,
    )
    print(json.dumps({"run_id": args.run_id, "manifest": manifest["manifest_path"], "manifest_sha256": manifest["manifest_sha256"]}, indent=2))


if __name__ == "__main__":
    main()
