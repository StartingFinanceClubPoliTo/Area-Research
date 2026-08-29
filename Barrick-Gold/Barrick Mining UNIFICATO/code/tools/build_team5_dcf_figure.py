"""Regenerate the synthetic Team 5 DCF figure inside the unified branch.

The underlying example is deliberately synthetic.  This wrapper prevents the
thesis from depending on an image copied from the standalone Team 5 article:
it executes the archived generator code and records the exact code/output
hashes used by the unified project.
"""

from __future__ import annotations

import argparse
import hashlib
import importlib.util
import json
import sys
from datetime import datetime, timezone
from pathlib import Path


ROOT = Path(__file__).resolve().parents[1]
SOURCE = ROOT / "parity" / "sources" / "team-5" / "current" / "src" / "generate_three_stage_dcf.py"


def sha256(path: Path) -> str:
    return hashlib.sha256(path.read_bytes()).hexdigest().upper()


def load_generator():
    spec = importlib.util.spec_from_file_location("team5_three_stage_dcf", SOURCE)
    if spec is None or spec.loader is None:
        raise RuntimeError(f"Cannot load Team 5 generator: {SOURCE}")
    module = importlib.util.module_from_spec(spec)
    sys.modules[spec.name] = module
    try:
        spec.loader.exec_module(module)
    except Exception:
        sys.modules.pop(spec.name, None)
        raise
    return module


def main() -> None:
    parser = argparse.ArgumentParser()
    parser.add_argument("--run-id", required=True)
    args = parser.parse_args()

    output_dir = ROOT / "outputs" / "thesis" / args.run_id / "team5_dcf"
    output_dir.mkdir(parents=True, exist_ok=True)
    module = load_generator()
    module.IMG_DIR = output_dir
    module.OUT_DIR = output_dir
    result = module.simulate(module.ModelParams())
    module.plot_base(result)
    generated = output_dir / "mc_dcf_results.png"
    promoted_name = output_dir / "mc_dcf_results_synthetic.png"
    generated.replace(promoted_name)

    summary = module.percentile_summary(result.ev)
    summary["terminal_value_share"] = float((result.pv_terminal / result.ev).mean())
    manifest = {
        "schema_version": "1.0",
        "status": "SYNTHETIC_ILLUSTRATIVE_NOT_BARRICK_VALUATION",
        "run_id": args.run_id,
        "generated_at_utc": datetime.now(timezone.utc).isoformat(),
        "generator": str(SOURCE.relative_to(ROOT)).replace("\\", "/"),
        "generator_sha256": sha256(SOURCE),
        "seed": int(result.params.seed),
        "simulations": int(result.params.n_sim),
        "summary": summary,
        "figure": {
            "name": promoted_name.name,
            "bytes": promoted_name.stat().st_size,
            "sha256": sha256(promoted_name),
        },
        "not_investment_advice": True,
    }
    (output_dir / "figure_manifest.json").write_text(
        json.dumps(manifest, indent=2), encoding="utf-8"
    )
    print(json.dumps(manifest, indent=2))


if __name__ == "__main__":
    main()
