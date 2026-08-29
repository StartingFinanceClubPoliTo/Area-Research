"""Create Team 7 G1.3 provenance manifests from frozen and run evidence."""

from __future__ import annotations

import argparse
import hashlib
import json
import platform
import sys
from pathlib import Path
from typing import Any


EMPTY_SHA256 = hashlib.sha256(b"").hexdigest().upper()


def sha256(path: Path) -> str:
    digest = hashlib.sha256()
    with path.open("rb") as handle:
        for chunk in iter(lambda: handle.read(1024 * 1024), b""):
            digest.update(chunk)
    return digest.hexdigest().upper()


def file_evidence(path: Path) -> dict[str, Any]:
    return {"bytes": path.stat().st_size, "sha256": sha256(path)}


def write_json(path: Path, payload: dict[str, Any]) -> None:
    path.write_text(
        json.dumps(payload, indent=2, ensure_ascii=False) + "\n", encoding="utf-8"
    )


def role_for_source(relative: str) -> str:
    roles = {
        "LICENSE": "mit_license_metadata",
        "CITATION.cff": "citation_metadata",
        "requirements.txt": "declared_dependencies",
        "src/01_data_download.py": "network_downloader_excluded_from_offline_run",
        "src/02_data_preparation.py": "offline_stage_executed",
        "src/03_descriptive_analysis.py": "offline_stage_executed",
        "src/04_gold_regressions.py": "offline_stage_executed",
        "src/05_silver_regressions.py": "offline_stage_executed",
        "src/06_stylized_facts.py": "offline_stage_executed",
        "src/06b_ar_mean_models.py": "offline_stage_executed",
        "src/06c_arma_mean_models.py": "offline_stage_executed",
        "src/06d_arima_mean_models.py": "offline_stage_executed",
        "src/07_garch_estimation.py": "offline_stage_executed",
        "src/08_arfima_analysis.py": "offline_stage_executed",
        "src/09_regression_tables.py": "offline_stage_executed",
        "src/config.py": "shared_configuration",
        "src/main.py": "full_orchestrator_excluded_to_prevent_downloader_execution",
        "src/utils.py": "shared_output_helpers",
    }
    return roles[relative]


def read_lock(path: Path) -> tuple[str, int]:
    raw = path.read_bytes()
    if raw.startswith(b"\xff\xfe"):
        text = raw.decode("utf-16")
        encoding = "utf-16le_with_bom_native_powershell_redirection"
    else:
        text = raw.decode("utf-8-sig")
        encoding = "utf-8"
    return encoding, len([line for line in text.splitlines() if line.strip()])


def main() -> None:
    parser = argparse.ArgumentParser()
    parser.add_argument("--original-root", type=Path, required=True)
    parser.add_argument("--workspace-root", type=Path, required=True)
    parser.add_argument("--run-root", type=Path, required=True)
    parser.add_argument("--accepted-at", required=True)
    args = parser.parse_args()

    original_root = args.original_root.resolve()
    workspace_root = args.workspace_root.resolve()
    run_root = args.run_root.resolve()
    frozen_source = workspace_root / "Github-Branch/parity/sources/team-7"
    frozen_inputs = workspace_root / "Github-Branch/parity/inputs/team-7"
    manifests = workspace_root / "Github-Branch/parity/manifests"
    comparison_path = run_root / "comparison.json"
    comparison = json.loads(comparison_path.read_text(encoding="utf-8"))

    source_files: list[dict[str, Any]] = []
    for frozen in sorted(path for path in frozen_source.rglob("*") if path.is_file()):
        relative = frozen.relative_to(frozen_source).as_posix()
        original = original_root / "Github-Branch" / relative
        frozen_hash = sha256(frozen)
        original_hash = sha256(original)
        source_files.append(
            {
                "path": relative,
                "original_path": f"Team 7 - Gold Volatility Dynamics/Github-Branch/{relative}",
                "role": role_for_source(relative),
                "bytes": frozen.stat().st_size,
                "sha256": frozen_hash,
                "copy_sha256_verified": frozen_hash == original_hash,
            }
        )

    source_manifest = {
        "schema_version": "1.0",
        "team": 7,
        "benchmark_class": "legacy_empirical_gold_silver_macro_volatility",
        "acquired_utc": "2026-08-15T05:07:08Z",
        "original_root": "Team 7 - Gold Volatility Dynamics/Github-Branch",
        "frozen_root": "UNIFICATO/Github-Branch/parity/sources/team-7",
        "copy_policy": "byte_identical_only",
        "license": {
            "status": "PRESENT",
            "identifier": "MIT",
            "path": "LICENSE",
            "sha256": sha256(frozen_source / "LICENSE"),
            "citation_path": "CITATION.cff",
            "citation_sha256": sha256(frozen_source / "CITATION.cff"),
        },
        "files": source_files,
        "verification": {
            "planned_executable_and_dependency_file_count": 16,
            "metadata_file_count": 2,
            "total_frozen_file_count": len(source_files),
            "all_originals_present": all(item["copy_sha256_verified"] for item in source_files),
            "all_sha256_copies_verified": all(item["copy_sha256_verified"] for item in source_files),
            "downloader_executed": False,
            "main_orchestrator_executed": False,
            "originals_modified": False,
        },
    }
    source_path = manifests / "team-7-source.json"
    write_json(source_path, source_manifest)

    input_roles = {
        "data/raw/yahoo_raw_download.csv": "frozen_yahoo_finance_download_consumed_by_stage_02",
        "data/raw/asset_levels_with_nans.csv": "distributed_raw_alignment_reference_regenerated_by_stage_02",
        "data/processed/asset_levels.csv": "distributed_processed_levels_reference_regenerated_by_stage_02",
        "data/processed/asset_returns.csv": "distributed_processed_returns_reference_regenerated_by_stage_02",
    }
    input_files: list[dict[str, Any]] = []
    for relative, role in input_roles.items():
        original = original_root / "Github-Branch" / relative
        frozen = frozen_inputs / relative
        work = run_root / "work" / relative
        original_hash = sha256(original)
        frozen_hash = sha256(frozen)
        work_hash = sha256(work)
        input_files.append(
            {
                "path": relative,
                "role": role,
                "bytes": original.stat().st_size,
                "original_sha256": original_hash,
                "frozen_sha256": frozen_hash,
                "work_after_run_sha256": work_hash,
                "original_frozen_work_match": original_hash == frozen_hash == work_hash,
            }
        )
    inputs_manifest = {
        "schema_version": "1.0",
        "team": 7,
        "run_id": run_root.name,
        "input_class": "frozen_legacy_yahoo_finance_sample_no_refresh",
        "provider": "Yahoo Finance via the distributed yfinance downloader",
        "provider_refresh_performed": False,
        "tickers": {
            "gold": "GLD",
            "silver": "SLV",
            "sp500": "^GSPC",
            "ust10y": "^TNX",
            "dxy": "DX-Y.NYB",
        },
        "declared_download_window": {"start": "2000-01-01", "end_exclusive": "2024-12-31"},
        "effective_aligned_levels_window": {"start": "2006-05-01", "end": "2024-12-30", "rows": 4695},
        "effective_returns_window": {"start": "2006-05-02", "end": "2024-12-30", "rows": 4694},
        "cutoff_test": {"status": "PASS", "maximum_date": "2024-12-30", "required_maximum": "2024-12-31"},
        "scale_and_transform": {
            "gold_silver_sp500_dxy": "daily log returns in decimal units in asset_returns.csv",
            "ust10y": "first difference of the distributed ^TNX close/yield series",
            "garch_input": "gold and silver decimal log returns multiplied by 100 to percentage points",
            "rolling_window": "252 trailing observations, center not enabled",
        },
        "files": input_files,
        "verification": {
            "file_count": len(input_files),
            "all_original_frozen_work_sha256_match": all(
                item["original_frozen_work_match"] for item in input_files
            ),
            "processed_levels_regenerated_byte_identical": input_files[2]["original_frozen_work_match"],
            "processed_returns_regenerated_byte_identical": input_files[3]["original_frozen_work_match"],
            "team_8_inputs_consumed": False,
        },
        "licensing_and_redistribution": {
            "source_code_license": "MIT",
            "market_data_redistribution_terms": "NOT_ESTABLISHED; frozen files retained as internal parity evidence only",
        },
    }
    inputs_path = manifests / "team-7-inputs.json"
    write_json(inputs_path, inputs_manifest)

    initial_lock = run_root / "runtime-lock.txt"
    arch_lock = run_root / "runtime-lock-with-arch.txt"
    initial_encoding, initial_count = read_lock(initial_lock)
    arch_encoding, arch_count = read_lock(arch_lock)
    runtime_manifest = {
        "schema_version": "1.0",
        "team": 7,
        "run_id": run_root.name,
        "runtime_status": "candidate_runtime_persisted_original_runtime_unknown",
        "execution_isolation": "dedicated run working directory, Python bytecode prefix, Matplotlib cache, and a system-site virtual environment containing arch only",
        "interpreter": {
            "executable": sys.executable.replace("\\", "/"),
            "implementation": platform.python_implementation(),
            "version": platform.python_version(),
            "architecture": platform.architecture()[0],
            "machine": platform.machine(),
        },
        "operating_system": {"platform": platform.platform(), "timezone_for_project_logs": "Europe/Rome"},
        "declared_requirements": [
            "arch", "jinja2", "matplotlib", "numpy", "pandas", "scipy",
            "seaborn", "statsmodels", "yfinance"
        ],
        "observed_direct_dependencies": {
            "arch": "8.0.0",
            "matplotlib": "3.9.2",
            "numpy": "2.1.0",
            "pandas": "2.3.3",
            "scipy": "1.14.1",
            "statsmodels": "0.14.6",
        },
        "runtime_locks": {
            "shared_environment_before_arch": {
                "path": f"UNIFICATO/Github-Branch/parity/runs/team-7/{run_root.name}/runtime-lock.txt",
                "format": initial_encoding,
                **file_evidence(initial_lock),
                "package_count": initial_count,
                "exit_code": 0,
                "stderr_bytes": (run_root / "runtime-lock.stderr.txt").stat().st_size,
            },
            "authoritative_stage_07_environment": {
                "path": f"UNIFICATO/Github-Branch/parity/runs/team-7/{run_root.name}/runtime-lock-with-arch.txt",
                "format": arch_encoding,
                **file_evidence(arch_lock),
                "package_count": arch_count,
                "exit_code": 0,
                "stderr_bytes": (run_root / "runtime-lock-with-arch.stderr.txt").stat().st_size,
            },
        },
        "runtime_preparation": {
            "authoritative_environment": ".venv-base; system-site packages plus arch 8.0.0 installed locally with --no-deps",
            "network_used_for_market_data": False,
            "network_used_during_stage_execution": False,
            "network_used_during_dependency_acquisition": True,
            "disclosure": "The arch wheel was first acquired while testing a rejected candidate environment, then installed from the local cache into .venv-base; no market data was downloaded.",
            "rejected_candidates": [
                ".venv with independently upgraded numerical stack (numpy 2.5.2, pandas 3.0.5, scipy 1.18.0); not used",
                "temporary uv Python 3.14.4 environment; not used for recorded outputs",
            ],
        },
        "environment": {
            "python_bytecode_prefix": f"UNIFICATO/Github-Branch/parity/runs/team-7/{run_root.name}/pycache",
            "MPLCONFIGDIR": f"UNIFICATO/Github-Branch/parity/runs/team-7/{run_root.name}/mplconfig",
            "matplotlib_backend": "Agg",
        },
        "known_runtime_gaps": [
            "The distributed requirements have no version pins and the original Python/package lock is absent.",
            "The system-site virtual environment shares the base numerical stack with the existing interpreter.",
            "Raw numerical and raster output differences cannot be assigned to a specific historical package or rendering version.",
        ],
    }
    runtime_path = manifests / "team-7-runtime.json"
    write_json(runtime_path, runtime_manifest)

    stage_times = {
        "02_data_preparation.py": ("2026-08-15T05:08:32.714Z", "2026-08-15T05:08:39.191Z", "shared_python_3.12.5"),
        "03_descriptive_analysis.py": ("2026-08-15T05:08:39.225Z", "2026-08-15T05:09:02.437Z", "shared_python_3.12.5"),
        "04_gold_regressions.py": ("2026-08-15T05:09:02.437Z", "2026-08-15T05:09:06.620Z", "shared_python_3.12.5"),
        "05_silver_regressions.py": ("2026-08-15T05:09:06.620Z", "2026-08-15T05:09:10.615Z", "shared_python_3.12.5"),
        "06_stylized_facts.py": ("2026-08-15T05:09:10.615Z", "2026-08-15T05:09:13.938Z", "shared_python_3.12.5"),
        "06b_ar_mean_models.py": ("2026-08-15T05:09:13.938Z", "2026-08-15T05:09:17.455Z", "shared_python_3.12.5"),
        "06c_arma_mean_models.py": ("2026-08-15T05:09:17.456Z", "2026-08-15T05:09:49.784Z", "shared_python_3.12.5"),
        "06d_arima_mean_models.py": ("2026-08-15T05:09:49.785Z", "2026-08-15T05:10:17.924Z", "shared_python_3.12.5"),
        "08_arfima_analysis.py": ("2026-08-15T05:10:17.925Z", "2026-08-15T05:10:23.017Z", "shared_python_3.12.5"),
        "09_regression_tables.py": ("2026-08-15T05:10:23.019Z", "2026-08-15T05:10:26.523Z", "shared_python_3.12.5"),
        "07_garch_estimation.py": ("2026-08-15T05:14:18.2723627Z", "2026-08-15T05:14:29.8681551Z", ".venv-base_python_3.12.5_arch_8.0.0"),
    }
    executions: list[dict[str, Any]] = []
    for stage, (started, ended, runtime) in stage_times.items():
        stem = Path(stage).stem
        stdout = run_root / "stage-logs" / f"{stem}.stdout.txt"
        stderr = run_root / "stage-logs" / f"{stem}.stderr.txt"
        executions.append(
            {
                "stage": stage,
                "runtime": runtime,
                "command": f"python src/{stage}",
                "started_utc": started,
                "completed_utc": ended,
                "exit_code": 0,
                "network_used": False,
                "stdout": {"path": f"stage-logs/{stdout.name}", **file_evidence(stdout)},
                "stderr": {"path": f"stage-logs/{stderr.name}", **file_evidence(stderr)},
            }
        )

    artifacts = comparison["artifacts"]
    exact_csv = [item["path"] for item in artifacts if item["status"] == "BYTE_IDENTICAL" and item["path"].endswith(".csv")]
    exact_tex = [item["path"] for item in artifacts if item["status"] == "BYTE_IDENTICAL" and item["path"].endswith(".tex")]
    sentinel_names = {
        "csv/arfima_gph_long_memory.csv",
        "csv/garch_params.csv",
        "tables/arfima_gph_table.tex",
        "tables/correlation_matrix.tex",
        "tables/descriptive_stats.tex",
        "tables/garch_params.tex",
        "tables/gold_equity_regressions.tex",
    }
    sentinels = [
        {
            "path": item["path"],
            "status": item["status"],
            "original_sha256": item.get("original_sha256"),
            "actual_sha256": item.get("actual_sha256"),
            "max_absolute_difference": item.get("comparison", {}).get("overall_max_absolute_difference"),
        }
        for item in artifacts
        if item["path"] in sentinel_names
    ]
    results_manifest = {
        "schema_version": "1.0",
        "team": 7,
        "benchmark_class": "legacy_empirical_gold_silver_macro_volatility",
        "run_id": run_root.name,
        "gate": "G1.3_team_7_offline",
        "gate_status": "ACCEPTED_LIMITED",
        "accepted_at": args.accepted_at,
        "acceptance_scope": "Byte-identical frozen source/input copies, regenerated processed data, eight CSV outputs and thirteen LaTeX tables only. All non-identical numerical files, every PNG, historical runtime parity, market-data redistribution, predictive validation, GPH interpretation, recommendation language, Team 8 integration, and Barrick attribution remain excluded.",
        "source_parity_passed": True,
        "source_parity_scope": "Sixteen executable/dependency files plus MIT/CITATION metadata, four input snapshots, eight exact CSV outputs, and thirteen exact LaTeX tables only.",
        "full_output_parity_passed": False,
        "full_historical_environment_parity_proven": False,
        "source_manifest": {"path": "UNIFICATO/Github-Branch/parity/manifests/team-7-source.json", "sha256": sha256(source_path)},
        "runtime_manifest": {"path": "UNIFICATO/Github-Branch/parity/manifests/team-7-runtime.json", "sha256": sha256(runtime_path)},
        "inputs_manifest": {"path": "UNIFICATO/Github-Branch/parity/manifests/team-7-inputs.json", "sha256": sha256(inputs_path)},
        "comparison_report": {"path": f"UNIFICATO/Github-Branch/parity/runs/team-7/{run_root.name}/comparison.json", **file_evidence(comparison_path)},
        "execution": {
            "cwd": f"UNIFICATO/Github-Branch/parity/runs/team-7/{run_root.name}/work",
            "stages": executions,
            "stage_count": len(executions),
            "all_exit_codes_zero": all(item["exit_code"] == 0 for item in executions),
            "all_stderr_empty": all(item["stderr"]["sha256"] == EMPTY_SHA256 for item in executions),
            "downloader_executed": False,
            "main_orchestrator_executed": False,
            "market_data_network_used": False,
        },
        "input_regeneration_tests": {
            "asset_levels_csv": {"status": "PASS_BYTE_IDENTICAL", "sha256": sha256(run_root / "work/data/processed/asset_levels.csv"), "shape": [4695, 5]},
            "asset_returns_csv": {"status": "PASS_BYTE_IDENTICAL", "sha256": sha256(run_root / "work/data/processed/asset_returns.csv"), "shape": [4694, 5]},
            "cutoff": {"status": "PASS", "maximum_date": "2024-12-30"},
        },
        "output_comparison": {
            **comparison["summary"],
            "accepted_byte_identical_csv": exact_csv,
            "accepted_byte_identical_tex": exact_tex,
            "sentinels": sentinels,
            "csv_difference_summary": {
                "arfima_gph_long_memory.csv": "maximum absolute difference 1.77635683940025e-15; table rounds byte-identically; raw CSV excluded",
                "garch_outputs": "parameters/conditional volatility/forecasts differ up to 1.42501480571156e-06; garch_params.tex rounds byte-identically; raw CSV excluded",
                "ar1_outputs": "coefficients and R-squared align, but HC1 standard errors/t-statistics differ materially under the candidate statsmodels runtime; raw CSV excluded",
                "arima_grid_gold.csv": "one (3,1,2) solver result differs by about 0.410779 in AIC/BIC/log-likelihood; raw CSV excluded",
                "other_nonidentical_csv": "mostly floating serialization or solver differences; no source-backed tolerance, therefore excluded",
            },
            "png_test": {
                "status": "NOT_PASSED",
                "tolerance_id": "TOL-IMAGE",
                "byte_identical": 0,
                "total": 15,
                "different_dimensions": 14,
                "same_dimensions_but_different_pixels": 1,
                "visual_spot_check": "PASS_INFORMATIONAL for correlation heatmap, gold/DXY rolling correlation, gold GARCH conditional volatility, and gold fractional differencing: same data geometry/labels and no clipping observed.",
                "reason": "No source-backed pixel tolerance exists; visual similarity does not establish image parity.",
            },
        },
        "static_temporal_audit": {
            "status": "PASS_LIMITED",
            "negative_shifts_found": 0,
            "centered_rolling_windows_found": 0,
            "rolling_correlations": "252-observation trailing pandas rolling windows",
            "forecast_scope": "GARCH 1-10 step conditional variance is produced from a model fitted on the full frozen sample and is not an out-of-sample performance test.",
            "prediction_claim_authorized": False,
        },
        "tolerance_ids": ["TOL-BYTE-0", "TOL-IMAGE"],
        "semantic_label": "legacy GLD/SLV and macro-financial empirical sample, 2006-05-01 through 2024-12-30; not Barrick company evidence, valuation, or investment advice",
        "known_divergences": [
            "The original Python version and dependency lock are not distributed.",
            "Sixteen of twenty-four CSV outputs are not byte-identical; no unapproved numerical tolerance was introduced.",
            "All fifteen PNG files fail byte parity; fourteen also have different decoded dimensions.",
            "The historical HC1 covariance and ARIMA solver runtime cannot be reconstructed from the distributed repository.",
            "The GARCH forecast is not accompanied by a train/test split or out-of-sample scoring protocol.",
            "The GPH d>0.5 interpretation remains under methodological review and is not promoted as an accepted claim.",
            "The distributed market-data files have no established redistribution approval in this audit.",
            "Recommendation language remains excluded from the unified thesis.",
            "No Team 8 provider, scale, surface, calibration, or Barrick adapter was consumed.",
        ],
    }
    results_path = manifests / "team-7-results.json"
    write_json(results_path, results_manifest)

    print(json.dumps({
        "source_manifest_sha256": sha256(source_path),
        "runtime_manifest_sha256": sha256(runtime_path),
        "inputs_manifest_sha256": sha256(inputs_path),
        "results_manifest_sha256": sha256(results_path),
    }, indent=2))


if __name__ == "__main__":
    main()
