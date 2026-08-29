"""Build the primary rolling one-step-ahead, no-look-ahead validation."""

import argparse
import hashlib
import json
from pathlib import Path
import subprocess
import sys
from time import perf_counter

import matplotlib

matplotlib.use("Agg")
import matplotlib.pyplot as plt
import numpy as np
import pandas as pd

ROOT = Path(__file__).resolve().parents[1]
if str(ROOT) not in sys.path:
    sys.path.insert(0, str(ROOT))

from Sampling import Sampling  # noqa: E402
from historical_validation import (  # noqa: E402
    MODEL_NAMES,
    cumulative_loss_analysis,
    fixed_surface_nodes,
    interpolate_observed_iv_grid,
    oos_r2_metrics,
    prepare_historical_eligible_surface,
)
from online_validation import (  # noqa: E402
    MODEL_SLUGS,
    merge_online_chains,
    parameter_stability,
    run_model_chain,
)


SEED = 20260813
MODEL_COLUMNS = {
    model: f"pred_iv_{MODEL_SLUGS[model]}" for model in MODEL_NAMES
}


def _sha256(path):
    """Fingerprint a local input so caches cannot survive silent data revisions."""
    digest = hashlib.sha256()
    with Path(path).open("rb") as handle:
        for block in iter(lambda: handle.read(1024 * 1024), b""):
            digest.update(block)
    return digest.hexdigest()


def _input_dates(option_history, stock_history):
    option_dates = set(pd.to_datetime(option_history["ts"], utc=True).dt.normalize())
    stock_dates = set(pd.to_datetime(stock_history["timestamp"], utc=True).dt.normalize())
    return sorted(option_dates.intersection(stock_dates))


def build_surface_cache(local_dir, force=False):
    """Build local-only observed grids and calibration samples once."""
    grid_path = local_dir / "online_observed_iv_grid.parquet"
    calibration_path = local_dir / "online_calibration_surfaces.parquet"
    design_path = local_dir / "online_surface_cache_design.json"
    option_source = local_dir / "options_GLD_1d.parquet"
    stock_source = local_dir / "gld_daily_history.csv"
    rate_source = local_dir / "usd_treasury_history.csv"
    option_history = pd.read_parquet(option_source)
    stock_history = pd.read_csv(stock_source)
    rate_history = pd.read_csv(rate_source)
    dates = _input_dates(option_history, stock_history)
    end = dates[-1]
    evaluation_start = end - pd.DateOffset(months=6)
    training_start = evaluation_start - pd.DateOffset(months=6)
    selected_dates = [date for date in dates if training_start <= date <= end]
    expected = {
        "training_start": training_start.strftime("%Y-%m-%d"),
        "evaluation_start": evaluation_start.strftime("%Y-%m-%d"),
        "evaluation_end": end.strftime("%Y-%m-%d"),
        "grid_nodes": 25,
        "calibration_nodes": 36,
        "option_source_sha256": _sha256(option_source),
        "stock_source_sha256": _sha256(stock_source),
        "rate_source_sha256": _sha256(rate_source),
    }
    if not force and grid_path.exists() and calibration_path.exists() and design_path.exists():
        cached = json.loads(design_path.read_text(encoding="utf-8"))
        if all(cached.get(key) == value for key, value in expected.items()):
            return grid_path, calibration_path, cached

    nodes = fixed_surface_nodes(
        n_moneyness=5,
        n_maturities=5,
        moneyness_bounds=(0.95, 1.05),
        maturity_days=(90, 270),
    )
    grids = []
    calibrations = []
    failures = []
    started = perf_counter()
    for number, date in enumerate(selected_dates, start=1):
        try:
            eligible = prepare_historical_eligible_surface(
                option_history,
                stock_history,
                rate_history,
                date,
                min_volume=25,
                moneyness_bounds=(0.85, 1.20),
                maturity_days=(21, 730),
            )
            grid = interpolate_observed_iv_grid(eligible, nodes)
            grids.append(grid)
            if date >= evaluation_start:
                sample = Sampling.sample_chebyshev(eligible, n_T=6, n_K=6)
                sample = sample.iloc[:36].copy()
                calibrations.append(sample)
        except (ValueError, IndexError) as error:
            failures.append({"date": date.strftime("%Y-%m-%d"), "error": str(error)})
        if number % 50 == 0:
            print(
                f"  prepared {number}/{len(selected_dates)} historical surfaces",
                flush=True,
            )
    grid_frame = pd.concat(grids, ignore_index=True)
    calibration_frame = pd.concat(calibrations, ignore_index=True)
    grid_frame.to_parquet(grid_path, index=False)
    calibration_frame.to_parquet(calibration_path, index=False)
    design = {
        **expected,
        "available_dates": len(selected_dates),
        "evaluation_dates": int(calibration_frame["date"].nunique()),
        "grid_moneyness_bounds": [0.95, 1.05],
        "grid_maturity_days": [90, 270],
        "grid_interpolation": "linear inside the daily observed convex hull; unavailable nodes remain missing",
        "calibration_filter": "calls; volume >= 25; 0.85 <= K/S <= 1.20; 21--730 DTE; valid traded-close IV",
        "failures": failures,
        "elapsed_seconds": float(perf_counter() - started),
    }
    design_path.write_text(json.dumps(design, indent=2), encoding="utf-8")
    return grid_path, calibration_path, design


def _plot_online_cumulative(cumulative, output_path):
    frame = pd.DataFrame(cumulative).copy()
    frame["date"] = pd.to_datetime(frame["date"], utc=True)
    figure, axes = plt.subplots(2, 2, figsize=(12.5, 8.4), sharex=True)
    groups = [
        [f"{model} vs Prevailing mean" for model in MODEL_NAMES],
        [f"{model} vs Random walk" for model in MODEL_NAMES],
        [f"{model} vs Black-Scholes" for model in MODEL_NAMES if model != "Black-Scholes"],
        ["Full Bates-Hawkes vs Heston", "Full Bates-Hawkes vs Bates"],
    ]
    titles = [
        "Models against the node-specific prevailing mean",
        "Models against yesterday's observed IV surface",
        "Added structure against Black--Scholes",
        "Self-excitation against Heston and Bates",
    ]
    for panel_index, (axis, labels, title) in enumerate(zip(axes.ravel(), groups, titles)):
        for label in labels:
            if label in frame:
                legend_label = (
                    f"vs {label.split(' vs ')[1]}"
                    if panel_index == 3 else label.split(" vs ")[0]
                )
                axis.plot(
                    frame["date"], frame[label], linewidth=1.8,
                    label=legend_label,
                )
        axis.axhline(0.0, color="black", linestyle="--", linewidth=0.9)
        axis.set_title(title)
        axis.set_ylabel("Cumulative loss difference (IV pp$^2$)")
        axis.grid(True, alpha=0.25)
        axis.legend(fontsize=8)
    for axis in axes[-1]:
        axis.set_xlabel("Target trading date")
    figure.autofmt_xdate(rotation=25)
    figure.suptitle(
        "Rolling one-step-ahead Welch--Goyal diagnostics: upward favors the named model",
        y=1.01,
        fontsize=13,
    )
    figure.tight_layout()
    figure.savefig(output_path, dpi=220, bbox_inches="tight")
    plt.close(figure)


def _launch_parallel_chains(calibration_path, grid_path, rate_path, local_dir):
    """Launch one independent Python process per chronological model chain."""
    workers = []
    result_paths = []
    for model in MODEL_NAMES:
        result_path = local_dir / f"online_{MODEL_SLUGS[model]}_worker_result.json"
        result_paths.append(result_path)
        command = [
            sys.executable,
            str(Path(__file__).resolve()),
            "--worker-model", model,
            "--calibration-cache", str(calibration_path),
            "--grid-cache", str(grid_path),
            "--rate-history", str(rate_path),
            "--checkpoint-dir", str(local_dir),
            "--result-path", str(result_path),
        ]
        workers.append((model, subprocess.Popen(command)))
    failures = []
    for model, process in workers:
        return_code = process.wait()
        if return_code != 0:
            failures.append(f"{model} exited with code {return_code}")
    if failures:
        raise RuntimeError("; ".join(failures))
    return [json.loads(path.read_text(encoding="utf-8")) for path in result_paths]


def build_public_outputs(grid_path, chain_results, design, output_dir, local_dir):
    merged = merge_online_chains(
        grid_path, chain_results, design["evaluation_start"]
    )
    required = [
        "implied_vol", "mean_forecast", "random_walk_forecast",
        *MODEL_COLUMNS.values(),
    ]
    common = merged.dropna(subset=required).copy()
    if common.empty:
        raise ValueError("No common rolling OOS observations remain after interpolation.")
    common = common.rename(columns={"target_date": "date"})
    _, mean_metrics, pairwise = oos_r2_metrics(
        common,
        MODEL_COLUMNS,
        benchmark_column="mean_forecast",
        benchmark_name="Prevailing mean",
    )
    _, random_walk_metrics, _ = oos_r2_metrics(
        common,
        MODEL_COLUMNS,
        benchmark_column="random_walk_forecast",
        benchmark_name="Random walk",
    )
    metrics = pd.concat([mean_metrics, random_walk_metrics], ignore_index=True)
    metrics = metrics.rename(columns={"weeks": "target_dates"})
    _, cumulative, loss_tests = cumulative_loss_analysis(
        common,
        MODEL_COLUMNS,
        benchmark_columns={
            "Prevailing mean": "mean_forecast",
            "Random walk": "random_walk_forecast",
        },
    )
    loss_tests = loss_tests.rename(columns={"weeks": "target_dates"})
    stability, convergence = parameter_stability(chain_results)

    target_dates = pd.to_datetime(common["date"], utc=True)
    metadata = {
        "status": "primary online predictive validation",
        "purpose": "simulate the information set available to a daily production run",
        "training_mean_start": design["training_start"],
        "first_forecast_origin": min(pd.to_datetime(common["origin_date"], utc=True)).strftime("%Y-%m-%d"),
        "first_target": target_dates.min().strftime("%Y-%m-%d"),
        "last_target": target_dates.max().strftime("%Y-%m-%d"),
        "forecast_origins": int(common["origin_date"].nunique()),
        "target_dates": int(target_dates.nunique()),
        "common_node_date_observations": int(len(common)),
        "nodes_per_complete_date": design["grid_nodes"],
        "forecast_rule": "calibrate on t, evolve current variance/intensity states over the calendar gap, forecast the fixed normalized IV grid at the next available trading date",
        "parameter_update": "first-date global-plus-local calibration followed by daily warm-start SLSQP; previous parameters retained only after an invalid/worse update",
        "parallel_policy": "four independent chronological model chains run in parallel processes; each chain remains sequential",
        "spot_policy": "S_t is used only in the t calibration; S_(t+1) is used only to construct the realized normalized target and never enters the forecast",
        "rate_policy": "forecast uses the latest complete LSE Treasury curve available on or before t",
        "mean_benchmark": "node-specific expanding mean using observations available through t",
        "random_walk_benchmark": "observed normalized IV at t for the same fixed node",
        "r2_definition": "1 - SSE_model / SSE_benchmark on common one-step-ahead observations",
        "cumulative_loss_definition": "sum through target t of comparator daily MSE minus model daily MSE",
        "systematic_test": "two-sided Newey-West HAC test of daily mean loss differences, four lags",
        "historical_data_limitation": "daily option close and daily GLD close may be asynchronous; liquidity, arbitrage, and vega filters mitigate but do not eliminate this noise",
        "valuation_separation": "the current Barrick/GLD point valuation continues to use the current LSE snapshot and is not replaced by historical rolling parameters",
        "seed": SEED,
    }
    for frame in (metrics, pairwise, loss_tests, convergence):
        frame["first_target"] = metadata["first_target"]
        frame["last_target"] = metadata["last_target"]

    metrics.to_csv(output_dir / "online_validation_metrics.csv", index=False)
    pairwise.to_csv(output_dir / "online_pairwise_r2.csv", index=False)
    loss_tests.to_csv(output_dir / "online_loss_differential_tests.csv", index=False)
    cumulative.to_csv(output_dir / "online_welch_goyal_cumulative.csv", index=False)
    stability.to_csv(output_dir / "online_parameter_stability.csv", index=False)
    convergence.to_csv(output_dir / "online_calibration_convergence.csv", index=False)
    _plot_online_cumulative(
        cumulative, output_dir / "online_welch_goyal_cumulative.png"
    )
    (output_dir / "online_validation_design.json").write_text(
        json.dumps(metadata, indent=2), encoding="utf-8"
    )
    payload = {
        "design": metadata,
        "metrics": metrics.to_dict(orient="records"),
        "pairwise": pairwise.to_dict(orient="records"),
        "loss_tests": loss_tests.to_dict(orient="records"),
        "convergence": convergence.to_dict(orient="records"),
    }
    (output_dir / "online_validation_metrics.json").write_text(
        json.dumps(payload, indent=2), encoding="utf-8"
    )
    merged.to_csv(local_dir / "online_validation_predictions.csv", index=False)
    return payload


def main(force_cache=False):
    local_dir = ROOT / "Data" / "lse_local"
    output_dir = ROOT / "Data"
    print("Preparing the ex-ante normalized surface cache...", flush=True)
    grid_path, calibration_path, design = build_surface_cache(
        local_dir, force=force_cache
    )
    print(
        "Launching four parallel chronological workers: BS, Heston, Bates, and full Bates--Hawkes...",
        flush=True,
    )
    chain_results = _launch_parallel_chains(
        calibration_path,
        grid_path,
        local_dir / "usd_treasury_history.csv",
        local_dir,
    )
    payload = build_public_outputs(
        grid_path, chain_results, design, output_dir, local_dir
    )
    print(json.dumps(payload, indent=2), flush=True)
    return payload


def parse_args():
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--force-cache", action="store_true")
    parser.add_argument("--worker-model", choices=MODEL_NAMES)
    parser.add_argument("--calibration-cache")
    parser.add_argument("--grid-cache")
    parser.add_argument("--rate-history")
    parser.add_argument("--checkpoint-dir")
    parser.add_argument("--result-path")
    return parser.parse_args()


if __name__ == "__main__":
    arguments = parse_args()
    if arguments.worker_model:
        result = run_model_chain(
            arguments.worker_model,
            arguments.calibration_cache,
            arguments.grid_cache,
            arguments.rate_history,
            arguments.checkpoint_dir,
            seed=SEED,
        )
        Path(arguments.result_path).write_text(
            json.dumps(result, indent=2), encoding="utf-8"
        )
    else:
        main(force_cache=arguments.force_cache)
