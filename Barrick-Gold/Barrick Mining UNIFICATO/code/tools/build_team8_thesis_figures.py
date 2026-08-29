"""Rebuild every Team 8 thesis image from code held in this branch.

Current option, rate and GLD-history inputs are transformed into a temporary
Team 8 working copy.  The original parity snapshot is never mutated and no
standalone-article PNG is promoted.  Row-level LSE data stay under the
Git-ignored local raw-data tree; only aggregate outputs and figures are copied
to the run directory.

The rolling Welch--Goyal figure is a separate, explicitly frozen layer because
its historical option-bar inputs cannot be redistributed.  It is nevertheless
rendered afresh from the committed aggregate cumulative-loss table.
"""

from __future__ import annotations

import argparse
import hashlib
import importlib
import json
import os
import shutil
import subprocess
import sys
from datetime import datetime, timezone
from pathlib import Path

import matplotlib

matplotlib.use("Agg")
import matplotlib.pyplot as plt
import numpy as np
import pandas as pd


ROOT = Path(__file__).resolve().parents[1]
TEAM8_SOURCE = ROOT / "parity" / "sources" / "team-8"

CURRENT_FIGURES = (
    "usd_treasury_curve.png",
    "sampling_comparison.png",
    "gld_return_normality.png",
    "black_scholes_residual_heatmap.png",
    "heston_residual_heatmap.png",
    "bates_residual_heatmap.png",
    "bates_hawkes_residual_heatmap.png",
    "bates_hawkes_volatility_smile.png",
    "bates_hawkes_vs_bates.png",
    "terminal_return_percentiles.png",
    "volatility_state_paths.png",
    "bates_poisson_jump_paths.png",
    "hawkes_jump_paths.png",
    "hawkes_intensity_comparison.png",
)
FROZEN_FIGURES = ("online_welch_goyal_cumulative.png",)
AGGREGATE_OUTPUTS = (
    "baseline_calibration_metrics.csv",
    "baseline_calibration_metrics.json",
    "black_scholes_calibrated_params.json",
    "heston_calibrated_params.json",
    "bates_calibrated_params.json",
    "bates_hawkes_calibrated_params.json",
    "bates_hawkes_calibration_metrics.csv",
    "bates_hawkes_calibration_metrics.json",
    "calibration_residual_normality.csv",
    "gld_return_normality_tests.csv",
    "gld_return_normality_tests.json",
    "terminal_return_percentiles_0_100.csv",
    "path_simulation_summary.csv",
    "model_parameters_long.csv",
    "model_appendix_summary.csv",
    "lse_publication_manifest.json",
    "usd_treasury_curve.csv",
    "usd_treasury_nss_fit.json",
    "hawkes_intensity_comparison.csv",
    "online_welch_goyal_cumulative.csv",
)


def sha256(path: Path) -> str:
    return hashlib.sha256(path.read_bytes()).hexdigest().upper()


def run_script(python: Path, script: Path, cwd: Path) -> None:
    environment = os.environ.copy()
    environment["PYTHONDONTWRITEBYTECODE"] = "1"
    subprocess.run(
        [str(python), str(script)], cwd=str(cwd), env=environment, check=True
    )


def prepare_local_dataset(staged_root: Path, raw_path: Path) -> dict[str, object]:
    payload = json.loads(raw_path.read_text(encoding="utf-8"))
    rows = payload["rows"]
    daily = rows["daily_stock_candles"]
    history = daily["GLD"] if isinstance(daily, dict) else daily
    sys.path.insert(0, str(staged_root))
    try:
        module = importlib.import_module("lse_dataset")
        _, metadata = module.write_local_dataset(
            rows["gld_option_calls"],
            rows["usd_treasury_yields"],
            history,
            staged_root / "Data" / "lse_local",
        )
        shutil.copy2(
            staged_root / "Data" / "lse_local" / "usd_treasury_nss_fit.json",
            staged_root / "Data" / "usd_treasury_nss_fit.json",
        )
    finally:
        sys.path.pop(0)
        sys.modules.pop("lse_dataset", None)
    return metadata


def build_hawkes_intensity_figure(data_dir: Path, seed: int = 20260827) -> None:
    parameters = json.loads(
        (data_dir / "bates_hawkes_calibrated_params.json").read_text(encoding="utf-8")
    )["parameters"]
    baseline = float(parameters["lambda_bar"])
    alpha = float(parameters["alpha"])
    beta = float(parameters["beta"])
    horizon = 5.0
    times = np.linspace(0.0, horizon, 601)
    dt = float(times[1] - times[0])
    rng = np.random.default_rng(seed)
    hawkes_intensity = np.empty_like(times)
    poisson_intensity = np.full_like(times, baseline)
    hawkes_counts = np.zeros_like(times)
    poisson_counts = np.zeros_like(times)
    hawkes_intensity[0] = baseline
    for index in range(1, len(times)):
        decayed = baseline + (hawkes_intensity[index - 1] - baseline) * np.exp(-beta * dt)
        hawkes_event = rng.random() < 1.0 - np.exp(-max(decayed, 0.0) * dt)
        poisson_event = rng.random() < 1.0 - np.exp(-baseline * dt)
        hawkes_intensity[index] = decayed + (alpha if hawkes_event else 0.0)
        hawkes_counts[index] = hawkes_counts[index - 1] + int(hawkes_event)
        poisson_counts[index] = poisson_counts[index - 1] + int(poisson_event)

    table = pd.DataFrame(
        {
            "time": times,
            "poisson_intensity": poisson_intensity,
            "hawkes_intensity": hawkes_intensity,
            "poisson_cumulative_jumps": poisson_counts,
            "hawkes_cumulative_jumps": hawkes_counts,
        }
    )
    table.to_csv(data_dir / "hawkes_intensity_comparison.csv", index=False)
    figure, axes = plt.subplots(2, 1, figsize=(9.4, 6.4), sharex=True)
    axes[0].plot(times, poisson_intensity, label="Poisson intensity", linewidth=1.8)
    axes[0].plot(times, hawkes_intensity, label="Hawkes intensity", linewidth=1.4)
    axes[0].set(ylabel="Annual jump intensity", title="Calibrated Poisson and self-exciting intensities")
    axes[0].legend()
    axes[1].step(times, poisson_counts, where="post", label="Poisson jumps", linewidth=1.6)
    axes[1].step(times, hawkes_counts, where="post", label="Hawkes jumps", linewidth=1.6)
    axes[1].set(xlabel="Years", ylabel="Cumulative jumps")
    axes[1].legend()
    for axis in axes:
        axis.grid(True, alpha=0.22)
    figure.tight_layout()
    figure.savefig(data_dir / "hawkes_intensity_comparison.png", dpi=220, bbox_inches="tight")
    plt.close(figure)


def build_frozen_online_figure(data_dir: Path) -> None:
    source = TEAM8_SOURCE / "Data" / "online_welch_goyal_cumulative.csv"
    frame = pd.read_csv(source)
    frame.to_csv(data_dir / source.name, index=False)
    frame["date"] = pd.to_datetime(frame["date"], utc=True)
    model_names = ("Black-Scholes", "Heston", "Bates", "Full Bates-Hawkes")
    groups = [
        [f"{model} vs Prevailing mean" for model in model_names],
        [f"{model} vs Random walk" for model in model_names],
        [f"{model} vs Black-Scholes" for model in model_names if model != "Black-Scholes"],
        ["Full Bates-Hawkes vs Heston", "Full Bates-Hawkes vs Bates"],
    ]
    titles = (
        "Models against the node-specific prevailing mean",
        "Models against yesterday's observed IV surface",
        "Added structure against Black--Scholes",
        "Self-excitation against Heston and Bates",
    )
    figure, axes = plt.subplots(2, 2, figsize=(12.5, 8.4), sharex=True)
    for panel_index, (axis, labels, title) in enumerate(zip(axes.ravel(), groups, titles)):
        for label in labels:
            if label in frame:
                legend = f"vs {label.split(' vs ')[1]}" if panel_index == 3 else label.split(" vs ")[0]
                axis.plot(frame["date"], frame[label], linewidth=1.8, label=legend)
        axis.axhline(0.0, color="black", linestyle="--", linewidth=0.9)
        axis.set_title(title)
        axis.set_ylabel("Cumulative loss difference (IV pp$^2$)")
        axis.grid(True, alpha=0.25)
        axis.legend(fontsize=8)
    for axis in axes[-1]:
        axis.set_xlabel("Target trading date")
    figure.autofmt_xdate(rotation=25)
    figure.suptitle(
        "Frozen rolling Welch--Goyal diagnostics: upward favors the named model",
        y=1.01,
        fontsize=13,
    )
    figure.tight_layout()
    figure.savefig(data_dir / "online_welch_goyal_cumulative.png", dpi=220, bbox_inches="tight")
    plt.close(figure)


def main() -> None:
    parser = argparse.ArgumentParser()
    parser.add_argument("--raw-input", type=Path, required=True)
    parser.add_argument("--run-id", required=True)
    args = parser.parse_args()
    raw_path = args.raw_input.resolve()
    output_dir = ROOT / "outputs" / "thesis" / args.run_id / "team8_empirical"
    output_dir.mkdir(parents=True, exist_ok=True)
    scratch_dir = ROOT / "outputs" / "tmp" / f"team8-{args.run_id}"
    if scratch_dir.exists():
        raise FileExistsError(f"Refusing to overwrite existing scratch directory: {scratch_dir}")
    scratch_dir.mkdir(parents=True)
    staged_root = scratch_dir / "team8"
    try:
        shutil.copytree(TEAM8_SOURCE, staged_root)
        metadata = prepare_local_dataset(staged_root, raw_path)
        python = Path(sys.executable)
        run_script(python, staged_root / "tools" / "rebuild_lse_benchmarks.py", staged_root)
        run_script(python, staged_root / "tools" / "rebuild_exact_hawkes_outputs.py", staged_root)
        run_script(python, staged_root / "tools" / "rebuild_path_outputs.py", staged_root)
        staged_data = staged_root / "Data"
        build_hawkes_intensity_figure(staged_data)
        build_frozen_online_figure(staged_data)

        missing = [name for name in (*CURRENT_FIGURES, *FROZEN_FIGURES) if not (staged_data / name).is_file()]
        if missing:
            raise RuntimeError(f"Team 8 rebuild did not create: {missing}")
        for name in (*CURRENT_FIGURES, *FROZEN_FIGURES, *AGGREGATE_OUTPUTS):
            source = staged_data / name
            if source.is_file():
                shutil.copy2(source, output_dir / name)
    finally:
        shutil.rmtree(scratch_dir)

    entries = []
    for name in (*CURRENT_FIGURES, *FROZEN_FIGURES):
        path = output_dir / name
        entries.append(
            {
                "name": name,
                "classification": "FROZEN_LEGACY" if name in FROZEN_FIGURES else "CURRENT",
                "bytes": path.stat().st_size,
                "sha256": sha256(path),
            }
        )
    manifest = {
        "schema_version": "1.0",
        "run_id": args.run_id,
        "generated_at_utc": datetime.now(timezone.utc).isoformat(),
        "status": "MIXED_CURRENT_AND_EXPLICITLY_FROZEN_TEAM8_FIGURES",
        "raw_input": str(raw_path.relative_to(ROOT)).replace("\\", "/"),
        "raw_input_sha256": sha256(raw_path),
        "row_level_data_committed": False,
        "team8_source": str(TEAM8_SOURCE.relative_to(ROOT)).replace("\\", "/"),
        "team8_source_files": {
            str(path.relative_to(ROOT)).replace("\\", "/"): sha256(path)
            for path in sorted(TEAM8_SOURCE.rglob("*.py"))
        },
        "dataset_metadata": metadata,
        "figures": entries,
        "figure_count": len(entries),
        "frozen_boundary": (
            "online_welch_goyal_cumulative.png is freshly rendered from frozen aggregate "
            "loss data because historical LSE option bars are not distributable in this branch"
        ),
        "not_investment_advice": True,
    }
    (output_dir / "figure_manifest.json").write_text(
        json.dumps(manifest, indent=2), encoding="utf-8"
    )
    print(json.dumps({"run_id": args.run_id, "output": str(output_dir), "figure_count": len(entries)}, indent=2))


if __name__ == "__main__":
    main()
