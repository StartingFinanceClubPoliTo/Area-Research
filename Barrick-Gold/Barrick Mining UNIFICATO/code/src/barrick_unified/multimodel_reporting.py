"""Reporting and provenance for the conditional four-model valuation."""

from __future__ import annotations

import csv
import hashlib
import json
import os
from pathlib import Path
from typing import Any

import matplotlib

matplotlib.use("Agg")
import matplotlib.pyplot as plt
import numpy as np
from scipy.stats import gaussian_kde

from .multimodel_valuation import MODEL_ORDER, MultiModelRun
from .valuation import QUANTILE_LEVELS


def sha256(path: Path) -> str:
    digest = hashlib.sha256()
    with path.open("rb") as handle:
        for chunk in iter(lambda: handle.read(1024 * 1024), b""):
            digest.update(chunk)
    return digest.hexdigest().upper()


def _array_hash(array: np.ndarray) -> str:
    values = np.ascontiguousarray(np.asarray(array, dtype="<f8"))
    return hashlib.sha256(values.tobytes()).hexdigest().upper()


def _verify_declared_reference(path: Path, entry: dict[str, Any], declared: str) -> None:
    """Keep the raw digest; permit only an exact UTF-8 Python CRLF-to-LF match.

    Git's Windows checkout can change line endings of frozen source references.
    No whitespace, encoding, lone-CR or numerical-content normalization is allowed.
    The matching mode is explicit in newly generated manifests; old manifests
    and reference files are never rewritten.
    """
    entry["declared_sha256"] = declared
    if not declared:
        return
    if entry["sha256"] == declared:
        entry["declared_hash_match"] = "raw_bytes"
        return
    if path.suffix == ".py":
        raw = path.read_bytes()
        try:
            raw.decode("utf-8", errors="strict")
        except UnicodeDecodeError:
            pass
        else:
            canonical = raw.replace(b"\r\n", b"\n")
            normalized_hash = hashlib.sha256(canonical).hexdigest().upper()
            if b"\r\n" in raw and b"\r" not in canonical and normalized_hash == declared:
                entry["declared_hash_match"] = "utf8_python_crlf_to_lf"
                entry["lf_sha256"] = normalized_hash
                return
    raise ValueError(f"declared reference hash mismatch: {entry['path']}")


def _write_json(path: Path, payload: dict[str, Any]) -> None:
    path.parent.mkdir(parents=True, exist_ok=True)
    path.write_text(
        json.dumps(payload, indent=2, ensure_ascii=False) + "\n", encoding="utf-8"
    )


def _write_csv(path: Path, rows: list[dict[str, Any]], fields: list[str]) -> None:
    path.parent.mkdir(parents=True, exist_ok=True)
    with path.open("w", encoding="utf-8", newline="") as handle:
        writer = csv.DictWriter(handle, fieldnames=fields)
        writer.writeheader()
        writer.writerows(rows)


def multimodel_quantile_rows(run: MultiModelRun) -> list[dict[str, Any]]:
    rows: list[dict[str, Any]] = []
    for model_id in MODEL_ORDER:
        model = run.models[model_id]
        valuation = model.valuation
        for level in QUANTILE_LEVELS:
            rows.append(
                {
                    "model_id": model_id,
                    "model": model.label,
                    "quantile_pct": level,
                    "enterprise_value_proxy_usd_mn": float(
                        np.percentile(
                            valuation.enterprise_value_proxy_usd_mn, level
                        )
                    ),
                    "equity_value_proxy_usd_mn": float(
                        np.percentile(valuation.equity_value_proxy_usd_mn, level)
                    ),
                    "value_per_share_proxy_usd": float(
                        np.percentile(valuation.value_per_share_proxy_usd, level)
                    ),
                }
            )
    return rows


def multimodel_summary_rows(run: MultiModelRun) -> list[dict[str, Any]]:
    observed = run.inputs.observed_share_price_usd
    rows: list[dict[str, Any]] = []
    for model_id in MODEL_ORDER:
        model = run.models[model_id]
        values = model.valuation.value_per_share_proxy_usd
        median = float(np.median(values))
        market_percentile = float(100.0 * np.mean(values <= observed))
        probability_above = float(np.mean(values > observed))
        rows.append(
            {
                "model_id": model_id,
                "model": model.label,
                "median_value_per_share_usd": median,
                "p10_value_per_share_usd": float(np.percentile(values, 10)),
                "p90_value_per_share_usd": float(np.percentile(values, 90)),
                "observed_close_usd": observed,
                "observed_close_percentile": market_percentile,
                "probability_model_value_exceeds_observed": probability_above,
                "observed_minus_median_usd": observed - median,
                "observed_vs_median_pct": (observed / median - 1.0) * 100.0
                if median != 0.0
                else None,
                "conditional_median_reading": (
                    "observed close above conditional median"
                    if observed > median
                    else "observed close below conditional median"
                ),
            }
        )
    return rows


def _latex_quantile_table(
    path: Path, rows: list[dict[str, Any]], observed: float
) -> None:
    lookup = {
        (str(row["model_id"]), int(row["quantile_pct"])): float(
            row["value_per_share_proxy_usd"]
        )
        for row in rows
    }
    lines = [
        "% Generated by barrick_unified. Do not edit manually.",
        "\\begin{table}[H]",
        "\\centering",
        "\\caption{Barrick conditional value/share quantiles by gold-price model}",
        "\\label{tab:conditional-multimodel-valuation-quantiles}",
        "\\setlength{\\tabcolsep}{8pt}",
        "\\resizebox{\\textwidth}{!}{%",
        "\\begin{tabular}{@{}r r r@{\\hspace{10pt}}r r@{}}",
        "\\toprule",
        "Quantile & BS/GBM & Heston & Bates--Poisson & Full Bates--Hawkes \\\\",
        "\\midrule",
    ]
    for level in QUANTILE_LEVELS:
        values = [lookup[(model_id, level)] for model_id in MODEL_ORDER]
        lines.append(
            f"P{level:02d} & "
            + " & ".join(f"{value:,.2f}" for value in values)
            + " \\\\"
        )
    lines.extend(
        [
            "\\bottomrule",
            "\\end{tabular}%",
            "}",
            "\\par\\smallskip",
            f"\\footnotesize Values in USD/share; observed B close: USD {observed:,.2f}. "
            "Only the gold-price layer changes. Production/cost use Team 4; DCF/WACC "
            "and equity bridge use Team 5. This is a conditional transfer of GLD/Q "
            "distributional shape to gold USD/oz, not a validated physical forecast, "
            "fair value or target price.",
            "\\end{table}",
            "",
        ]
    )
    path.parent.mkdir(parents=True, exist_ok=True)
    path.write_text("\n".join(lines), encoding="utf-8")


def _bh_distribution_figure(run: MultiModelRun, output: Path) -> None:
    values = run.models["full_bates_hawkes"].valuation.value_per_share_proxy_usd
    observed = run.inputs.observed_share_price_usd
    lower, upper = np.percentile(values, [0.5, 99.5])
    lower, upper = min(float(lower), observed), max(float(upper), observed)
    shown = values[(values >= lower) & (values <= upper)]
    median = float(np.median(values))
    p10, p90 = np.percentile(values, [10, 90])
    figure, axis = plt.subplots(figsize=(10.2, 5.6))
    axis.hist(
        shown,
        bins=90,
        density=True,
        color="#D9E4F0",
        edgecolor="white",
        linewidth=0.35,
    )
    axis.axvspan(
        p10, p90, color="#D4AF37", alpha=0.18, label="Conditional P10--P90"
    )
    axis.axvline(
        median,
        color="#17365D",
        linewidth=2.1,
        label=f"Full Bates--Hawkes median: ${median:,.2f}",
    )
    axis.axvline(
        observed,
        color="#B33A3A",
        linewidth=2.1,
        linestyle="--",
        label=f"Observed B close: ${observed:,.2f}",
    )
    axis.set(
        title="Barrick conditional value distribution — Full Bates--Hawkes gold layer",
        xlabel="Conditional value (USD/share)",
        ylabel="Density",
    )
    axis.grid(axis="y", alpha=0.2)
    axis.legend(frameon=False)
    figure.text(
        0.01,
        0.01,
        "Gold only: conditional transfer of GLD/Q distributional shape to gold USD/oz. "
        "Team 4 production/cost vectors; Team 5 DCF/WACC. Not a validated physical "
        "forecast, fair value, target or recommendation.",
        fontsize=8,
    )
    figure.tight_layout(rect=(0, 0.04, 1, 1))
    output.parent.mkdir(parents=True, exist_ok=True)
    figure.savefig(output, dpi=220, bbox_inches="tight")
    plt.close(figure)


def _comparison_figure(run: MultiModelRun, output: Path) -> None:
    observed = run.inputs.observed_share_price_usd
    all_values = np.concatenate(
        [run.models[model_id].valuation.value_per_share_proxy_usd for model_id in MODEL_ORDER]
    )
    lower, upper = np.percentile(all_values, [0.5, 99.5])
    lower, upper = min(float(lower), observed), max(float(upper), observed)
    grid = np.linspace(lower, upper, 320)
    colors = ("#1F77B4", "#2CA02C", "#D07A00", "#6F42C1")
    figure, axis = plt.subplots(figsize=(10.6, 5.9))
    for model_id, color in zip(MODEL_ORDER, colors):
        model = run.models[model_id]
        values = model.valuation.value_per_share_proxy_usd
        shown = values[(values >= lower) & (values <= upper)]
        density = gaussian_kde(shown, bw_method="scott")(grid)
        median = float(np.median(values))
        axis.plot(
            grid,
            density,
            color=color,
            linewidth=1.9,
            label=f"{model.label} (P50 ${median:,.2f})",
        )
    axis.axvline(
        observed,
        color="#B33A3A",
        linestyle="--",
        linewidth=2.0,
        label=f"Observed B close: ${observed:,.2f}",
    )
    axis.set(
        title="Barrick conditional value distributions by gold-price model",
        xlabel="Conditional value (USD/share)",
        ylabel="Density",
    )
    axis.grid(axis="y", alpha=0.2)
    axis.legend(frameon=False, fontsize=9)
    figure.text(
        0.01,
        0.01,
        "Gold-price layer only: conditional transfer of GLD/Q distributional shape to gold USD/oz.\n"
        "Common deterministic Team 4 production/cost vectors and Team 5 DCF/WACC; "
        "densities use the pooled P0.5--P99.5 display range.",
        fontsize=8,
    )
    figure.tight_layout(rect=(0, 0.065, 1, 1))
    output.parent.mkdir(parents=True, exist_ok=True)
    figure.savefig(output, dpi=220, bbox_inches="tight")
    plt.close(figure)


def _input_entry(path: Path, project_root: Path, role: str) -> dict[str, Any]:
    resolved = path.resolve()
    try:
        logical = resolved.relative_to(project_root.resolve()).as_posix()
    except ValueError:
        logical = Path(os.path.relpath(resolved, project_root.resolve())).as_posix()
    return {
        "role": role,
        "path": logical,
        "bytes": resolved.stat().st_size,
        "sha256": sha256(resolved),
    }


def write_multimodel_outputs(
    run: MultiModelRun,
    experiment_config: dict[str, Any],
    experiment_config_path: Path,
    base_config_path: Path,
    output_dir: Path,
    figure_dir: Path,
    manifest_path: Path,
    project_root: Path,
    run_id: str,
) -> dict[str, Any]:
    """Write aggregate outputs and a complete source/artifact manifest."""

    output_dir.mkdir(parents=True, exist_ok=True)
    figure_dir.mkdir(parents=True, exist_ok=True)
    artifact_root = output_dir.parents[2]
    quantile_rows = multimodel_quantile_rows(run)
    summary_rows = multimodel_summary_rows(run)

    quantiles_csv = output_dir / "valuation_quantiles_by_model.csv"
    _write_csv(
        quantiles_csv,
        quantile_rows,
        [
            "model_id",
            "model",
            "quantile_pct",
            "enterprise_value_proxy_usd_mn",
            "equity_value_proxy_usd_mn",
            "value_per_share_proxy_usd",
        ],
    )
    quantiles_tex = output_dir / "valuation_quantiles_by_model.tex"
    _latex_quantile_table(
        quantiles_tex, quantile_rows, run.inputs.observed_share_price_usd
    )
    summary_csv = output_dir / "valuation_summary_by_model.csv"
    _write_csv(summary_csv, summary_rows, list(summary_rows[0]))

    common_contract = {
        "scope": "GOLD_PRICE_LAYER_ONLY",
        "production_and_cost": "Barrick actuals for 2026Q1-Q2 followed by the frozen Team 4 forecast for 2026Q3-2030; vectors are identical across all four valuations",
        "temporary_v3_gold_driver": "Team 4 GBM/NSS was temporary and is not a production/cost model",
        "dcf_wacc_equity_bridge": "Unified DCF methodology; identical across all four valuations",
        "common_input_hashes": run.common_input_hashes,
        "wacc_shocks_sha256": run.wacc_shocks_sha256,
        "wacc_identical_across_models": True,
        "gold_engine_base_seed_alignment": experiment_config["simulation"]["common_random_numbers_note"],
    }
    summary_payload = {
        "schema_version": str(experiment_config["schema_version"]),
        "status": run.inputs.status,
        "run_id": run_id,
        "valuation_as_of_utc": run.inputs.valuation_as_of_utc,
        "observed_market_price": {
            "ticker": "B",
            "price_usd": run.inputs.observed_share_price_usd,
            "timestamp_utc": run.inputs.observed_share_price_timestamp_utc,
            "source": run.inputs.observed_share_price_source,
        },
        "primary_structural_scenario": "full_bates_hawkes",
        "selection_basis": experiment_config["selection_basis"],
        "models": summary_rows,
        "common_contract": common_contract,
        "conditional_bridge": run.bridge,
        "calibration_snapshot": run.calibration_snapshot,
        "asynchronous_dates": run.asynchronous_dates,
        "interpretation": "conditional transfer of GLD/Q distributional shape to gold USD/oz; research sensitivity only, not a validated physical forecast, fair value, target price or investment recommendation",
    }
    summary_json = output_dir / "valuation_summary_by_model.json"
    _write_json(summary_json, summary_payload)

    gold_quantile_rows: list[dict[str, Any]] = []
    for model_id in MODEL_ORDER:
        terminal = run.models[model_id].quarterly_gold_paths[:, -1]
        for level in QUANTILE_LEVELS:
            gold_quantile_rows.append(
                {
                    "model_id": model_id,
                    "model": run.models[model_id].label,
                    "quantile_pct": level,
                    "terminal_gold_usd_per_oz": float(
                        np.percentile(terminal, level)
                    ),
                }
            )
    gold_quantiles_csv = output_dir / "terminal_gold_quantiles_by_model.csv"
    _write_csv(
        gold_quantiles_csv,
        gold_quantile_rows,
        ["model_id", "model", "quantile_pct", "terminal_gold_usd_per_oz"],
    )

    bh_figure = figure_dir / "full_bates_hawkes_value_distribution.png"
    comparison_figure = figure_dir / "multimodel_value_distribution_comparison.png"
    _bh_distribution_figure(run, bh_figure)
    _comparison_figure(run, comparison_figure)

    code_paths = [
        project_root / "src" / "barrick_unified" / "valuation.py",
        project_root / "src" / "barrick_unified" / "multimodel_valuation.py",
        project_root / "src" / "barrick_unified" / "multimodel_reporting.py",
        project_root / "run_multimodel_valuation.py",
    ]
    layer = experiment_config["gold_price_layer"]
    source_dir = project_root / layer["team8_source_dir"]
    input_entries = [
        _input_entry(experiment_config_path, project_root, "experiment_config"),
        _input_entry(base_config_path, project_root, "base_team4_team5_contract"),
        _input_entry(project_root / "pyproject.toml", project_root, "runtime_dependency_contract"),
        _input_entry(project_root / "requirements.txt", project_root, "runtime_dependency_contract"),
    ]
    if str(experiment_config.get("schema_version")) == "4.0":
        input_entries.extend(
            [
                _input_entry(source_dir / "path_simulation.py", project_root, "team8_20260902_path_adapter"),
                _input_entry(project_root / layer["calibration_manifest"], project_root, "team8_calibration_manifest"),
                _input_entry(project_root / layer["nss_curve_file"], project_root, "team8_nss_curve"),
                _input_entry(project_root / layer["calibration_summary_file"], project_root, "team8_calibration_summary"),
                _input_entry(project_root / layer["surface_diagnostics_file"], project_root, "team8_surface_diagnostics"),
                _input_entry(project_root / layer["oos_manifest"], project_root, "team8_oos_manifest"),
                _input_entry(project_root / layer["oos_model_summary"], project_root, "team8_oos_model_summary"),
            ]
        )
        for path in sorted(source_dir.rglob("*.py")):
            if path.name != "path_simulation.py":
                input_entries.append(_input_entry(path, project_root, "vendored_team8_source"))
    else:
        input_entries.extend(
            [
                _input_entry(source_dir / "path_simulation.py", project_root, "frozen_team8_path_engine"),
                _input_entry(source_dir / "BatesHawkesExact.py", project_root, "frozen_team8_compensator"),
                _input_entry(source_dir / "Hawkes.py", project_root, "frozen_team8_hawkes_engine"),
                _input_entry(source_dir / "Data" / "lse_publication_manifest.json", project_root, "team8_lse_snapshot_manifest"),
                _input_entry(source_dir / "Data" / "baseline_calibration_metrics.csv", project_root, "team8_fit_metrics"),
                _input_entry(source_dir / "Data" / "bates_hawkes_calibration_metrics.csv", project_root, "team8_full_fit_metrics"),
            ]
        )
    for model_id in MODEL_ORDER:
        input_entries.append(
            _input_entry(
                run.models[model_id].parameter_path,
                project_root,
                f"team8_parameters_{model_id}",
            )
        )
    for reference in run.inputs.reference_files:
        reference_path = (project_root / str(reference["path"])).resolve()
        entry = _input_entry(reference_path, project_root, "base_declared_reference")
        declared = str(reference.get("sha256", "")).upper()
        _verify_declared_reference(reference_path, entry, declared)
        input_entries.append(entry)

    artifacts = [
        quantiles_csv,
        quantiles_tex,
        summary_csv,
        summary_json,
        gold_quantiles_csv,
        bh_figure,
        comparison_figure,
    ]
    manifest = {
        "schema_version": str(experiment_config["schema_version"]),
        "status": run.inputs.status,
        "run_id": run_id,
        "valuation_as_of_utc": run.inputs.valuation_as_of_utc,
        "model_boundary": common_contract,
        "calibration_snapshot": run.calibration_snapshot,
        "asynchronous_dates": run.asynchronous_dates,
        "conditional_bridge": run.bridge,
        "selection_basis": experiment_config["selection_basis"],
        "path_grid": run.path_grid,
        "simulation": {
            "paths": run.inputs.n_simulations,
            "engine_seeds": {
                model_id: run.models[model_id].engine_seed for model_id in MODEL_ORDER
            },
            "wacc_seed": int(experiment_config["simulation"]["wacc_seed"]),
            "wacc_shocks_sha256": run.wacc_shocks_sha256,
        },
        "observed_market_price": summary_payload["observed_market_price"],
        "model_summaries": summary_rows,
        "model_array_hashes": {
            model_id: {
                "quarterly_gold_paths_sha256": _array_hash(
                    run.models[model_id].quarterly_gold_paths
                ),
                "annual_wacc_sha256": _array_hash(
                    run.models[model_id].valuation.annual_wacc
                ),
                "value_per_share_sha256": _array_hash(
                    run.models[model_id].valuation.value_per_share_proxy_usd
                ),
                "common_input_hashes": run.common_input_hashes,
            }
            for model_id in MODEL_ORDER
        },
        "inputs": input_entries,
        "code": [
            _input_entry(path, project_root, "code") for path in code_paths
        ],
        "artifacts": [
            {
                "path": path.relative_to(artifact_root).as_posix(),
                "bytes": path.stat().st_size,
                "sha256": sha256(path),
            }
            for path in artifacts
        ],
        "unresolved_corporate_inputs": list(run.inputs.unresolved_corporate_inputs),
        "not_investment_advice": True,
        "validated_physical_gold_forecast": False,
    }
    _write_json(manifest_path, manifest)
    return manifest
