"""Run and record Team 8 G1.4 unit/benchmark parity without market data."""

from __future__ import annotations

import argparse
import hashlib
import importlib.metadata
import json
import os
import platform
import re
import subprocess
import sys
from datetime import datetime, timezone
from pathlib import Path
from typing import Any


EXPECTED_SENTINELS = {
    "main.py": "F69BBDF4D17C70B07F7A0F786C88D40A89A71553E80C80B92FC8A165CBCAA427",
    "lse_dataset.py": "29A256925E9C48A5640EEF57337239437166C0C10C4E18112D8D96BB8D5BD441",
    "calibration_core.py": "280C2603E19E0096200BB50C772E0F3D52C8929256CBE252BE5A28055E10BD8D",
    "calibration_workflow.py": "CDAAA3F5CDDE86C84D1CF7FFCBD051C54D78B9B3419A7ABB23A9017E148C7C81",
    "fourier_pricing.py": "3E212417D9B1A937F21EBB32E04C41C1883B26C4F5978F2CA3A3F16541A87F38",
    "BnS.py": "07465D4C9CCAAC9EA6A4CA28744875747685C88C5A5BB5BEB733C6C361A2EF04",
    "Heston.py": "9D145E9F5A0DDF60FEC99BE4CCA39A2A48CA73CDE7089A83C0B8C53F67163794",
    "Bates.py": "CCF5145EE1497BBB01891D6CB0C95803D7D9FD0F61ECA537BE81AFEA61F463C2",
    "BatesHawkes.py": "606810CA162AF2B682788B88349B07119117BB3A2EC90E6243E4A344004B5790",
    "BatesHawkesExact.py": "59F1207C9230E6522C2A0CFA5A66B638B05AB745B8D1C1BE7643E1DE169F8153",
    "Hawkes.py": "AFB1B48D86F7A0E00AADA1091583F73C21BF3BB1CF6ADD7CC552BD1178658BBB",
    "historical_validation.py": "4A1E73623748BA917871DAFED9D2B0AC5DC2FD9E1E811088E452590111FFE162",
    "online_validation.py": "C3240B12A174535D5F963D387EC799D7EAD12DDE5BF85EC337923155E7DDF86F",
    "path_simulation.py": "DE6AA684EB0937BFB808173B2A2518D9C5C1C2A3BE3533736F6A12FC6F4232BC",
    "Sampling.py": "CC5F0BFFA5CC10DA35574204FC8B2FE5E283BA39CB1C9B5CEBF4E39E617E8FF7",
    "benchmarks/benchmark_pricing.py": "3AD7D76B0CE58C8C240090E8B87F683E8E5D0BD32EE32DB2DFEA38255D661BF3",
    "tests/test_bates_refactor.py": "6CC512DD8CF5434AA67AEC4C04E756B6EA6D922B9F4EB9F8C46D3DE20FEEEA49",
    "tests/test_hawkes_calibration.py": "458CE6D74FEF84B37859B2AACB51649B20849A0BBE99091B16BC6AE8C86C9E97",
    "tests/test_hawkes_exact.py": "4EBB3ACC30BC088893F9702792F9C4DE1F077D050D0EA882229922A357CCE5B5",
    "tests/test_historical_validation.py": "25DF4D77D35B9A1587E90493FFF069DE971E3351A0EBBB67999B03C4A2DBD62B",
    "tests/test_lse_dataset.py": "20D1AE682BA6EFE985E04939E14F063872F8DA40086E2C00DE24F8A9E28C40C8",
    "tests/test_path_simulation.py": "7046AAC113AACF5965CAFDAF55085EB29B3B1DBBDB2E309DAEE557489FCD4A10",
    "requirements.txt": "10F1494EE48D2DABBBC4F85041472F6EC0E8D0F9B7186A99FE5170C7B61A5659",
}


def sha256(path: Path) -> str:
    digest = hashlib.sha256()
    with path.open("rb") as handle:
        for chunk in iter(lambda: handle.read(1024 * 1024), b""):
            digest.update(chunk)
    return digest.hexdigest().upper()


def evidence(path: Path) -> dict[str, Any]:
    return {"bytes": path.stat().st_size, "sha256": sha256(path)}


def write_json(path: Path, payload: dict[str, Any]) -> None:
    path.write_text(json.dumps(payload, indent=2) + "\n", encoding="utf-8")


def run(command: list[str], cwd: Path, env: dict[str, str], stdout: Path, stderr: Path) -> int:
    completed = subprocess.run(command, cwd=cwd, env=env, text=True, capture_output=True, check=False)
    stdout.write_text(completed.stdout, encoding="utf-8")
    stderr.write_text(completed.stderr, encoding="utf-8")
    return int(completed.returncode)


def main() -> int:
    parser = argparse.ArgumentParser()
    parser.add_argument("--original-root", type=Path, required=True)
    parser.add_argument("--workspace-root", type=Path, required=True)
    args = parser.parse_args()
    original = args.original_root.resolve()
    workspace = args.workspace_root.resolve()
    frozen = workspace / "parity/sources/team-8"
    manifests = workspace / "parity/manifests"
    run_id = datetime.now(timezone.utc).strftime("%Y%m%dT%H%M%SZ") + "-py3125"
    run_root = workspace / "parity/runs/team-8" / run_id
    run_root.mkdir(parents=True)
    manifests.mkdir(parents=True, exist_ok=True)

    sentinel_mismatches = {
        relative: {"expected": expected, "actual": sha256(frozen / relative)}
        for relative, expected in EXPECTED_SENTINELS.items()
        if sha256(frozen / relative) != expected
    }
    if sentinel_mismatches:
        raise RuntimeError(f"Team 8 sentinel mismatch: {sentinel_mismatches}")

    source_files = []
    for path in sorted(item for item in frozen.rglob("*") if item.is_file()):
        relative = path.relative_to(frozen).as_posix()
        original_path = original / relative
        frozen_hash = sha256(path)
        source_files.append(
            {
                "path": relative,
                "bytes": path.stat().st_size,
                "sha256": frozen_hash,
                "copy_sha256_verified": original_path.is_file() and sha256(original_path) == frozen_hash,
            }
        )
    source_manifest = {
        "schema_version": "1.0",
        "team": 8,
        "benchmark_class": "market_real_LSE_GLD_models_unit_scope_only",
        "original_root": "Team 8 - Gold Options Stochastic Modeling/Github-Branch",
        "frozen_root": "UNIFICATO/Github-Branch/parity/sources/team-8",
        "files": source_files,
        "verification": {
            "file_count": len(source_files),
            "sentinel_count": len(EXPECTED_SENTINELS),
            "all_copies_verified": all(row["copy_sha256_verified"] for row in source_files),
            "originals_modified": False,
        },
    }
    source_path = manifests / "team-8-source.json"
    write_json(source_path, source_manifest)

    env = os.environ.copy()
    env.pop("LSE_API_KEY", None)
    env["PYTHONDONTWRITEBYTECODE"] = "1"
    env["MPLBACKEND"] = "Agg"
    started = datetime.now(timezone.utc).isoformat()
    test_out = run_root / "pytest.stdout.txt"
    test_err = run_root / "pytest.stderr.txt"
    test_code = run(
        [sys.executable, "-m", "pytest", "tests", "-q", "-p", "no:cacheprovider"],
        frozen,
        env,
        test_out,
        test_err,
    )
    benchmark_out = run_root / "benchmark.stdout.txt"
    benchmark_err = run_root / "benchmark.stderr.txt"
    benchmark_code = run(
        [sys.executable, "benchmarks/benchmark_pricing.py"],
        frozen,
        env,
        benchmark_out,
        benchmark_err,
    )
    lock_out = run_root / "runtime-lock.txt"
    lock_err = run_root / "runtime-lock.stderr.txt"
    lock_code = run([sys.executable, "-m", "pip", "freeze", "--all"], frozen, env, lock_out, lock_err)
    completed = datetime.now(timezone.utc).isoformat()

    direct_versions = {}
    for package in ("numpy", "pandas", "scipy", "matplotlib", "jupyter", "nelson-siegel-svensson", "lse-data", "pyarrow", "pytest"):
        try:
            direct_versions[package] = importlib.metadata.version(package)
        except importlib.metadata.PackageNotFoundError:
            direct_versions[package] = None
    runtime_manifest = {
        "schema_version": "1.0",
        "team": 8,
        "run_id": run_id,
        "runtime_status": "candidate_runtime_original_runtime_unknown",
        "interpreter": {
            "executable": sys.executable.replace("\\", "/"),
            "version": platform.python_version(),
            "implementation": platform.python_implementation(),
            "platform": platform.platform(),
        },
        "direct_dependency_versions": direct_versions,
        "runtime_lock": {"path": f"parity/runs/team-8/{run_id}/runtime-lock.txt", **evidence(lock_out), "exit_code": lock_code},
        "known_gap": "The historical Team 8 Python/dependency lock was not distributed; this is a candidate runtime.",
    }
    runtime_path = manifests / "team-8-runtime.json"
    write_json(runtime_path, runtime_manifest)

    inputs_manifest = {
        "schema_version": "1.0",
        "team": 8,
        "run_id": run_id,
        "scope": "G1.4_unit_and_pricing_benchmark_only",
        "market_data_network_used": False,
        "api_key_present_in_child_process": False,
        "raw_lse_inputs_consumed": False,
        "committed_public_aggregates_rebuilt": False,
        "G1_5_historical_market_parity": "NOT_PROVEN",
    }
    inputs_path = manifests / "team-8-inputs.json"
    write_json(inputs_path, inputs_manifest)

    pytest_text = test_out.read_text(encoding="utf-8")
    match = re.search(r"(\d+) passed", pytest_text)
    benchmark = {}
    for line in benchmark_out.read_text(encoding="utf-8").splitlines():
        if "=" in line:
            key, value = line.split("=", 1)
            benchmark[key] = value
    accepted = test_code == benchmark_code == lock_code == 0 and match is not None
    results = {
        "schema_version": "1.0",
        "team": 8,
        "run_id": run_id,
        "gate": "G1.4_team_8_unit_parity_offline",
        "gate_status": "ACCEPTED" if accepted else "FAILED",
        "accepted_at_utc": completed if accepted else None,
        "source_parity_passed": all(row["copy_sha256_verified"] for row in source_files),
        "historical_market_parity_passed": False,
        "source_manifest": {"path": "parity/manifests/team-8-source.json", "sha256": sha256(source_path)},
        "runtime_manifest": {"path": "parity/manifests/team-8-runtime.json", "sha256": sha256(runtime_path)},
        "inputs_manifest": {"path": "parity/manifests/team-8-inputs.json", "sha256": sha256(inputs_path)},
        "execution": {
            "cwd": "parity/sources/team-8",
            "started_utc": started,
            "completed_utc": completed,
            "network_used": False,
            "lse_api_key_removed_from_child_environment": True,
            "pytest": {"command": "python -m pytest tests -q -p no:cacheprovider", "exit_code": test_code, "passed": int(match.group(1)) if match else None, "stdout": evidence(test_out), "stderr": evidence(test_err)},
            "benchmark": {"command": "python benchmarks/benchmark_pricing.py", "exit_code": benchmark_code, "metrics": benchmark, "stdout": evidence(benchmark_out), "stderr": evidence(benchmark_err)},
        },
        "tolerance": "TOL-T8-TESTS; existing test thresholds unchanged",
        "acceptance_scope": "Pure unit/regression suite and deterministic pricing benchmark only.",
        "excluded": [
            "G1.5 historical LSE market parity",
            "current LSE refresh as a substitute for the 2026-08-12 raw snapshot",
            "GLD-to-gold or realized-price conversion",
            "Barrick valuation, EBITDA, FCFF, enterprise value or value per share",
        ],
    }
    write_json(manifests / "team-8-results.json", results)
    print(json.dumps({"run_id": run_id, "gate_status": results["gate_status"], "pytest_passed": results["execution"]["pytest"]["passed"], "benchmark": benchmark}, indent=2))
    return 0 if accepted else 1


if __name__ == "__main__":
    raise SystemExit(main())
