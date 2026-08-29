"""Build the publication chart for Team 8 one-step-ahead OOS R-squared.

The chart is derived only from the byte-frozen aggregate metrics.  It does not
re-run the licensed historical LSE calibration and never mutates the Team 8
source freeze.
"""

from __future__ import annotations

import argparse
import csv
import json
import math
from pathlib import Path

import matplotlib

matplotlib.use("Agg")

import matplotlib.pyplot as plt
import numpy as np
from matplotlib.ticker import PercentFormatter


PROJECT_ROOT = Path(__file__).resolve().parents[1]
TEAM8_DATA = PROJECT_ROOT / "parity" / "sources" / "team-8" / "Data"
DEFAULT_METRICS = TEAM8_DATA / "online_validation_metrics.csv"
DEFAULT_DESIGN = TEAM8_DATA / "online_validation_design.json"
DEFAULT_OUTPUT = (
    PROJECT_ROOT
    / "outputs"
    / "figures"
    / "team-8"
    / "online_oos_r2_by_benchmark.png"
)

MODEL_ORDER = ("Black-Scholes", "Heston", "Bates", "Full Bates-Hawkes")
BENCHMARK_ORDER = ("Prevailing mean", "Random walk")
BENCHMARK_LABELS = {
    "Prevailing mean": "Expanding mean",
    "Random walk": "Previous-day IV",
}


def load_validated_metrics(
    metrics_path: Path = DEFAULT_METRICS,
    design_path: Path = DEFAULT_DESIGN,
) -> tuple[list[dict[str, object]], dict[str, object]]:
    """Load the frozen aggregates and verify their internal R-squared identity."""

    with metrics_path.open("r", encoding="utf-8-sig", newline="") as handle:
        raw_rows = list(csv.DictReader(handle))
    design = json.loads(design_path.read_text(encoding="utf-8"))

    expected_pairs = {
        (model, benchmark)
        for model in MODEL_ORDER
        for benchmark in BENCHMARK_ORDER
    }
    observed_pairs: set[tuple[str, str]] = set()
    rows: list[dict[str, object]] = []

    for raw in raw_rows:
        model = raw.get("model", "")
        benchmark = raw.get("benchmark", "")
        pair = (model, benchmark)
        if pair not in expected_pairs:
            raise ValueError(f"Unexpected model/benchmark row: {pair!r}")
        if pair in observed_pairs:
            raise ValueError(f"Duplicate model/benchmark row: {pair!r}")
        observed_pairs.add(pair)

        observations = int(raw["observations"])
        target_dates = int(raw["target_dates"])
        sse = float(raw["sse_iv_decimal"])
        benchmark_sse = float(raw["benchmark_sse_iv_decimal"])
        r2_oos = float(raw["r2_oos_vs_benchmark"])
        recomputed = 1.0 - sse / benchmark_sse
        if not math.isclose(r2_oos, recomputed, rel_tol=0.0, abs_tol=1e-12):
            raise ValueError(
                f"R-squared identity fails for {model} vs {benchmark}: "
                f"stored={r2_oos}, recomputed={recomputed}"
            )

        rows.append(
            {
                "model": model,
                "benchmark": benchmark,
                "r2_oos": r2_oos,
                "observations": observations,
                "target_dates": target_dates,
                "first_target": raw["first_target"],
                "last_target": raw["last_target"],
            }
        )

    missing = expected_pairs - observed_pairs
    if missing:
        raise ValueError(f"Missing model/benchmark rows: {sorted(missing)!r}")

    expected_observations = int(design["common_node_date_observations"])
    expected_dates = int(design["target_dates"])
    expected_first = str(design["first_target"])
    expected_last = str(design["last_target"])
    for row in rows:
        if row["observations"] != expected_observations:
            raise ValueError("Metric observations disagree with the frozen design")
        if row["target_dates"] != expected_dates:
            raise ValueError("Metric target-date count disagrees with the frozen design")
        if row["first_target"] != expected_first or row["last_target"] != expected_last:
            raise ValueError("Metric target range disagrees with the frozen design")

    if design.get("r2_definition") != (
        "1 - SSE_model / SSE_benchmark on common one-step-ahead observations"
    ):
        raise ValueError("Unexpected OOS R-squared definition in the frozen design")

    return rows, design


def build_figure(
    metrics_path: Path = DEFAULT_METRICS,
    design_path: Path = DEFAULT_DESIGN,
    output_path: Path = DEFAULT_OUTPUT,
) -> Path:
    """Render the two-benchmark OOS R-squared comparison as a PNG."""

    rows, design = load_validated_metrics(metrics_path, design_path)
    lookup = {
        (str(row["model"]), str(row["benchmark"])): float(row["r2_oos"])
        for row in rows
    }

    x = np.arange(len(MODEL_ORDER), dtype=float)
    width = 0.34
    colors = ("#0B3C5D", "#D69E2E")
    fig, ax = plt.subplots(figsize=(12.2, 7.2))

    for index, benchmark in enumerate(BENCHMARK_ORDER):
        values = [lookup[(model, benchmark)] for model in MODEL_ORDER]
        offset = (index - 0.5) * width
        bars = ax.bar(
            x + offset,
            values,
            width=width,
            color=colors[index],
            label=BENCHMARK_LABELS[benchmark],
            edgecolor="white",
            linewidth=0.8,
        )
        for bar, value in zip(bars, values, strict=True):
            vertical_alignment = "bottom" if value >= 0.0 else "top"
            label_y = value + 0.022 if value >= 0.0 else value - 0.022
            ax.text(
                bar.get_x() + bar.get_width() / 2.0,
                label_y,
                f"{value:+.1%}",
                ha="center",
                va=vertical_alignment,
                fontsize=9.5,
                fontweight="semibold",
            )

    ax.axhline(0.0, color="#222222", linewidth=1.2)
    ax.set_xticks(x)
    ax.set_xticklabels(
        ("Black-Scholes", "Heston", "Bates", "Full Bates-Hawkes"),
        fontsize=10.5,
    )
    ax.set_ylabel(r"Out-of-sample $R^2$", fontsize=11)
    ax.yaxis.set_major_formatter(PercentFormatter(xmax=1.0, decimals=0))
    ax.set_ylim(-0.38, 1.02)
    ax.grid(axis="y", color="#D9D9D9", linewidth=0.8, alpha=0.75)
    ax.set_axisbelow(True)
    ax.spines["top"].set_visible(False)
    ax.spines["right"].set_visible(False)
    ax.legend(loc="lower left", frameon=False, ncol=2, fontsize=10)

    fig.suptitle(
        "Team 8: one-step-ahead predictive performance",
        x=0.5,
        y=0.965,
        fontsize=16,
        fontweight="bold",
        color="#0B3C5D",
    )
    ax.set_title(
        "OOS R-squared by option model and benchmark",
        fontsize=12,
        pad=14,
    )

    sample_note = (
        f"{int(design['target_dates']):,} target dates · "
        f"{int(design['common_node_date_observations']):,} common node-date observations · "
        f"{design['first_target']} to {design['last_target']}"
    )
    fig.text(0.5, 0.075, sample_note, ha="center", fontsize=9.5, color="#333333")
    fig.text(
        0.5,
        0.040,
        (
            r"$R^2_{OOS}=1-SSE_{model}/SSE_{benchmark}$; positive values favour the model. "
            "Previous-day IV is the same-node random-walk benchmark."
        ),
        ha="center",
        fontsize=9,
        color="#444444",
    )
    fig.text(
        0.5,
        0.014,
        (
            "Source: byte-frozen Team 8 aggregate. Figure reproducible offline; "
            "historical market parity was not independently re-run in UNIFICATO."
        ),
        ha="center",
        fontsize=8.2,
        color="#666666",
    )

    output_path.parent.mkdir(parents=True, exist_ok=True)
    fig.tight_layout(rect=(0.045, 0.105, 0.99, 0.91))
    fig.savefig(output_path, dpi=200, bbox_inches="tight", facecolor="white")
    plt.close(fig)
    return output_path


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(
        description="Build the Team 8 online OOS R-squared publication chart."
    )
    parser.add_argument("--metrics", type=Path, default=DEFAULT_METRICS)
    parser.add_argument("--design", type=Path, default=DEFAULT_DESIGN)
    parser.add_argument("--output", type=Path, default=DEFAULT_OUTPUT)
    return parser.parse_args()


def main() -> None:
    args = parse_args()
    output = build_figure(args.metrics, args.design, args.output)
    print(output)


if __name__ == "__main__":
    main()
