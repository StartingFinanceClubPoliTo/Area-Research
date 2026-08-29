"""Independent acceptance checks for the Team 7 G1.3 evidence package."""

from __future__ import annotations

import hashlib
import json
import re
from pathlib import Path


ROOT = Path(__file__).resolve().parents[2]
ORIGINAL = ROOT.parent / "Team 7 - Gold Volatility Dynamics" / "Github-Branch"
RUN = ROOT / "Github-Branch/parity/runs/team-7/20260815T050733Z-py3125"
MANIFESTS = ROOT / "Github-Branch/parity/manifests"
FROZEN_SOURCE = ROOT / "Github-Branch/parity/sources/team-7"
FROZEN_INPUTS = ROOT / "Github-Branch/parity/inputs/team-7"


def sha256(path: Path) -> str:
    return hashlib.sha256(path.read_bytes()).hexdigest().upper()


def main() -> None:
    checks: list[tuple[str, bool]] = []

    def check(name: str, condition: bool) -> None:
        checks.append((name, bool(condition)))

    expected_dirs = {"Utilities-Progetto", "Overleaf", "Github-Branch", "Revisione", "Drive"}
    root_dirs = {path.name for path in ROOT.iterdir() if path.is_dir()}
    root_files = [path for path in ROOT.iterdir() if path.is_file()]
    check("exact five-folder workspace root", root_dirs == expected_dirs and not root_files)
    revisione_files = [path for path in (ROOT / "Revisione").rglob("*") if path.is_file()]
    check("Revisione contains PDFs only", all(path.suffix.lower() == ".pdf" for path in revisione_files))

    source = json.loads((MANIFESTS / "team-7-source.json").read_text(encoding="utf-8"))
    runtime = json.loads((MANIFESTS / "team-7-runtime.json").read_text(encoding="utf-8"))
    inputs = json.loads((MANIFESTS / "team-7-inputs.json").read_text(encoding="utf-8"))
    results = json.loads((MANIFESTS / "team-7-results.json").read_text(encoding="utf-8"))
    comparison = json.loads((RUN / "comparison.json").read_text(encoding="utf-8"))
    check("four manifests/reports parse", all(isinstance(item, dict) for item in (source, runtime, inputs, results, comparison)))

    check("source freeze count 16 plus 2 metadata", len(source["files"]) == 18 and source["verification"]["planned_executable_and_dependency_file_count"] == 16)
    check("all source copies verified", all(item["copy_sha256_verified"] for item in source["files"]))
    check("MIT license and citation frozen", source["license"]["identifier"] == "MIT" and (FROZEN_SOURCE / "LICENSE").is_file() and (FROZEN_SOURCE / "CITATION.cff").is_file())
    check("downloader and main excluded", not source["verification"]["downloader_executed"] and not source["verification"]["main_orchestrator_executed"])

    work_source_matches = []
    for item in source["files"]:
        relative = item["path"]
        if relative in {"LICENSE", "CITATION.cff"}:
            continue
        work_source_matches.append(sha256(FROZEN_SOURCE / relative) == sha256(RUN / "work" / relative))
    check("16 run source/dependency files unchanged", len(work_source_matches) == 16 and all(work_source_matches))

    expected_input_hashes = {
        "data/raw/yahoo_raw_download.csv": "032EF3D6F68626BC6144BD4D73227FF64B3C1E6E1E2C5EEB2333F309D0396251",
        "data/raw/asset_levels_with_nans.csv": "C6F02748B3489867558609DDB590EAD7499700FD47BFE208DE8BACAD339634C7",
        "data/processed/asset_levels.csv": "F17F647560AECA1EC272E4981A3161255936E63644919E8E0F1103019EEEE77B",
        "data/processed/asset_returns.csv": "800C60FE0AFC4A4B7620A3C8C5A52DB9540B4F669F6EEF185FC4A911DE2826D1",
    }
    check("four input snapshots recorded", len(inputs["files"]) == 4)
    check("original/frozen/work inputs match", all(item["original_frozen_work_match"] for item in inputs["files"]))
    check("planned input hashes match", all(item["original_sha256"] == expected_input_hashes[item["path"]] for item in inputs["files"]))
    check("sample cutoff passes", inputs["cutoff_test"]["status"] == "PASS" and inputs["cutoff_test"]["maximum_date"] == "2024-12-30")
    check("Team 8 inputs excluded", not inputs["verification"]["team_8_inputs_consumed"])

    stages = results["execution"]["stages"]
    check("eleven offline stages recorded", len(stages) == 11)
    check("all stage exit codes zero", results["execution"]["all_exit_codes_zero"] and all(item["exit_code"] == 0 for item in stages))
    check("all stage stderr empty", results["execution"]["all_stderr_empty"] and all(item["stderr"]["bytes"] == 0 for item in stages))
    check("no stage network/data download", not results["execution"]["market_data_network_used"] and all(not item["network_used"] for item in stages))

    summary = comparison["summary"]
    check("all 52 generated outputs present", summary["expected_generated_artifact_count"] == 52 and summary["actual_generated_artifact_count"] == 52 and summary["all_generated_artifacts_present"])
    check("README is only absent reference file", summary["original_file_count"] == 53 and summary["actual_file_count"] == 52 and summary["status_counts_all_files"]["MISSING_ACTUAL"] == 1)
    check("21 generated outputs byte-identical", summary["status_counts_generated_artifacts"]["BYTE_IDENTICAL"] == 21)
    check("8 of 24 CSV byte-identical", summary["byte_identical_by_extension"][".csv"] == {"byte_identical": 8, "total": 24})
    check("13 of 13 TeX byte-identical", summary["byte_identical_by_extension"][".tex"] == {"byte_identical": 13, "total": 13})
    check("0 of 15 PNG byte-identical", summary["byte_identical_by_extension"][".png"] == {"byte_identical": 0, "total": 15})

    artifact_by_path = {item["path"]: item for item in comparison["artifacts"]}
    check("GARCH rounded table exact but raw CSV differs", artifact_by_path["tables/garch_params.tex"]["status"] == "BYTE_IDENTICAL" and artifact_by_path["csv/garch_params.csv"]["status"] == "DIFFERENT")
    check("GPH rounded table exact but raw CSV differs", artifact_by_path["tables/arfima_gph_table.tex"]["status"] == "BYTE_IDENTICAL" and artifact_by_path["csv/arfima_gph_long_memory.csv"]["status"] == "DIFFERENT")

    check("limited acceptance flags consistent", results["gate_status"] == "ACCEPTED_LIMITED" and results["source_parity_passed"] and not results["full_output_parity_passed"] and not results["full_historical_environment_parity_proven"])
    check("comparison report hash bound", results["comparison_report"]["sha256"] == sha256(RUN / "comparison.json"))
    check("source manifest hash bound", results["source_manifest"]["sha256"] == sha256(MANIFESTS / "team-7-source.json"))
    check("runtime manifest hash bound", results["runtime_manifest"]["sha256"] == sha256(MANIFESTS / "team-7-runtime.json"))
    check("inputs manifest hash bound", results["inputs_manifest"]["sha256"] == sha256(MANIFESTS / "team-7-inputs.json"))

    initial_lock = runtime["runtime_locks"]["shared_environment_before_arch"]
    arch_lock = runtime["runtime_locks"]["authoritative_stage_07_environment"]
    check("runtime locks complete", initial_lock["package_count"] == 185 and arch_lock["package_count"] == 186 and initial_lock["stderr_bytes"] == 0 and arch_lock["stderr_bytes"] == 0)

    source_text = "\n".join(
        path.read_text(encoding="utf-8")
        for path in (FROZEN_SOURCE / "src").glob("*.py")
    )
    check("no negative shift", re.search(r"shift\s*\(\s*-", source_text, re.IGNORECASE) is None)
    check("no centered rolling window", re.search(r"center\s*=\s*True", source_text, re.IGNORECASE) is None)
    network_files = [
        path.name
        for path in (FROZEN_SOURCE / "src").glob("*.py")
        if re.search(r"yfinance|yf\.download|requests\.|urlopen", path.read_text(encoding="utf-8"), re.IGNORECASE)
    ]
    check("network code isolated to excluded downloader", network_files == ["01_data_download.py"])

    failed = [name for name, passed in checks if not passed]
    for name, passed in checks:
        print(f"{'PASS' if passed else 'FAIL'} | {name}")
    print(f"SUMMARY | {len(checks) - len(failed)}/{len(checks)} PASS")
    if failed:
        raise SystemExit(1)


if __name__ == "__main__":
    main()
