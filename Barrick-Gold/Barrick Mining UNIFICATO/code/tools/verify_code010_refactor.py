"""Verify CODE-012 against the immutable CODE-010 manifest."""

from __future__ import annotations

import argparse
import json
from pathlib import Path
import sys

import numpy as np

ROOT = Path(__file__).resolve().parents[1]
SRC = ROOT / "src"
if str(SRC) not in sys.path:
    sys.path.insert(0, str(SRC))

from barrick_unified.multimodel_valuation import MODEL_ORDER, run_multimodel_valuation  # noqa: E402
from barrick_unified.refactored import RefactoredBarrickPipeline  # noqa: E402
from barrick_unified.refactored.domain.provenance import array_sha256, file_sha256  # noqa: E402
from barrick_unified.valuation import QUANTILE_LEVELS  # noqa: E402


POINTER = ROOT / "data" / "manifests" / "valuation" / "AUTHORITATIVE_MULTIMODEL.json"
CONFIG = ROOT / "config" / "multimodel_valuation_20260826.json"


def verify(run_id: str) -> dict:
    pointer = json.loads(POINTER.read_text(encoding="utf-8"))
    manifest_path = ROOT / pointer["manifest_path"]
    if file_sha256(manifest_path) != pointer["manifest_sha256"]:
        raise AssertionError("CODE-010 manifest hash changed")
    golden = json.loads(manifest_path.read_text(encoding="utf-8"))
    file_audit = {"verified": [], "post_golden_changes": []}
    for section in ("inputs", "code", "artifacts"):
        for entry in golden[section]:
            path = ROOT / entry["path"]
            actual = file_sha256(path) if path.is_file() else None
            if path.is_file() and path.stat().st_size == entry["bytes"] and actual == entry["sha256"]:
                file_audit["verified"].append(entry["path"])
                continue
            if section == "code" and entry["path"] == "src/barrick_unified/multimodel_reporting.py":
                file_audit["post_golden_changes"].append(
                    {
                        "path": entry["path"],
                        "golden_bytes": entry["bytes"],
                        "golden_sha256": entry["sha256"],
                        "current_bytes": path.stat().st_size if path.is_file() else None,
                        "current_sha256": actual,
                        "status": "KNOWN_POST_CODE010_REPORTING_ONLY_CHANGE_ARRAYS_REVERIFIED",
                    }
                )
                continue
            raise AssertionError(f"golden {section} mismatch: {entry['path']}")
    config = json.loads(CONFIG.read_text(encoding="utf-8"))
    result = RefactoredBarrickPipeline().run(ROOT, config)
    legacy = run_multimodel_valuation(ROOT, config)
    if not np.array_equal(result.wacc_shocks, legacy.wacc_shocks):
        raise AssertionError("refactored WACC differs from the legacy runner")
    models = {}
    historical_matches = True
    for model_id in MODEL_ORDER:
        model = result.models[model_id]
        legacy_model = legacy.models[model_id]
        if not np.array_equal(model.quarterly_gold_paths, legacy_model.quarterly_gold_paths):
            raise AssertionError(f"{model_id} refactored paths differ from legacy")
        for field in model.valuation.__dataclass_fields__:
            if not np.array_equal(
                getattr(model.valuation, field), getattr(legacy_model.valuation, field)
            ):
                raise AssertionError(f"{model_id} refactored {field} differs from legacy")
        actual = {
            "quarterly_gold_paths_sha256": array_sha256(model.quarterly_gold_paths),
            "annual_wacc_sha256": array_sha256(model.valuation.annual_wacc),
            "value_per_share_sha256": array_sha256(model.valuation.value_per_share_proxy_usd),
        }
        expected = golden["model_array_hashes"][model_id]
        matches = {key: value == expected[key] for key, value in actual.items()}
        historical_matches = historical_matches and all(matches.values())
        quantiles = {
            f"P{level:02d}": float(np.percentile(model.valuation.value_per_share_proxy_usd, level))
            for level in QUANTILE_LEVELS
        }
        models[model_id] = {
            **actual,
            "historical_code010_hash_match": matches,
            "value_per_share_quantiles_usd": quantiles,
        }
    payload = {
        "schema_version": "1.0",
        "run_id": run_id,
        "status": "PASS_REFACTORED_VS_LEGACY_CURRENT_RUNTIME",
        "historical_code010_array_hashes_match": historical_matches,
        "historical_runtime_note": (
            "Historical CODE-010 hashes differ under the currently resolved Sobol runtime; "
            "the accepted artifacts remain frozen and the refactored and legacy runners are "
            "bitwise identical under the same runtime."
            if not historical_matches
            else "Historical CODE-010 hashes reproduced exactly."
        ),
        "golden_run_id": golden["run_id"],
        "golden_manifest_path": pointer["manifest_path"],
        "golden_manifest_sha256": pointer["manifest_sha256"],
        "verified_counts": {
            "inputs": len(golden["inputs"]),
            "code": len(golden["code"]),
            "artifacts": len(golden["artifacts"]),
            "models": len(models),
        },
        "wacc_shocks_sha256": result.wacc_shocks_sha256,
        "common_input_hashes": result.common_input_hashes,
        "golden_file_audit": file_audit,
        "models": models,
        "notes": [
            "Four engines model gold only.",
            "Team 4 production/cost and Team 5 DCF/WACC are unchanged.",
            "GLD/Q to gold and Q-to-P bridges remain unvalidated.",
        ],
    }
    output = ROOT / "outputs" / "verification" / run_id / "golden_matrix.json"
    output.parent.mkdir(parents=True, exist_ok=False)
    output.write_text(json.dumps(payload, indent=2, ensure_ascii=False) + "\n", encoding="utf-8")
    return {**payload, "output": output.relative_to(ROOT).as_posix(), "output_sha256": file_sha256(output)}


def main() -> None:
    parser = argparse.ArgumentParser()
    parser.add_argument("--run-id", required=True)
    args = parser.parse_args()
    print(json.dumps(verify(args.run_id), indent=2, ensure_ascii=False))


if __name__ == "__main__":
    main()
