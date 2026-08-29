"""Current Barrick operating figures derived from public Q1/Q2 2026 tables.

This module reuses the operational decomposition developed by Team 4 but does
not import its NSS, implied-volatility, GBM, gold-price or illustrative EBITDA
artifacts.  All figures are rebuilt from branch-local tabular data.
"""

from __future__ import annotations

import csv
import hashlib
import json
from pathlib import Path
import platform
from typing import Any

import matplotlib

matplotlib.use("Agg")
import matplotlib.pyplot as plt
import numpy as np


FIGURE_NAMES = (
    "team4_mine_production_qoq",
    "team4_mine_cost_qoq",
    "team4_process_components_qoq",
    "team4_operating_contract",
)

COLORS = {"2026Q1": "#718096", "2026Q2": "#D4AF37"}


def _sha256(path: Path) -> str:
    return hashlib.sha256(path.read_bytes()).hexdigest().upper()


def _read_rows(path: Path) -> list[dict[str, str]]:
    with path.open(encoding="utf-8", newline="") as handle:
        rows = list(csv.DictReader(handle))
    if not rows:
        raise ValueError(f"empty operating input: {path}")
    return rows


def _save(figure: plt.Figure, output_dir: Path, name: str) -> list[Path]:
    png = output_dir / f"{name}.png"
    pdf = output_dir / f"{name}.pdf"
    figure.savefig(png, dpi=240, bbox_inches="tight", facecolor="white")
    figure.savefig(pdf, bbox_inches="tight", facecolor="white")
    plt.close(figure)
    return [png, pdf]


def _grouped_bars(
    rows: list[dict[str, str]],
    field: str,
    ylabel: str,
    title: str,
    output_dir: Path,
    name: str,
) -> list[Path]:
    mines = list(dict.fromkeys(row["mine"] for row in rows))
    x = np.arange(len(mines), dtype=float)
    width = 0.36
    figure, axis = plt.subplots(figsize=(11.6, 6.2))
    for offset, quarter in zip((-width / 2, width / 2), ("2026Q1", "2026Q2")):
        indexed = {row["mine"]: float(row[field]) for row in rows if row["quarter"] == quarter}
        values = [indexed[mine] for mine in mines]
        bars = axis.bar(x + offset, values, width, label=quarter, color=COLORS[quarter])
        axis.bar_label(bars, fmt="%.0f", padding=2, fontsize=7)
    axis.set_xticks(x, mines, rotation=24, ha="right")
    axis.set_ylabel(ylabel)
    axis.set_title(title)
    axis.grid(axis="y", alpha=0.22)
    axis.legend(frameon=False)
    figure.text(
        0.01,
        0.01,
        "Source: Barrick Q1/Q2 2026 Mine Statistics. Selected Team 4 mine perimeter; company totals also include other operations.",
        fontsize=8,
    )
    figure.tight_layout(rect=(0, 0.05, 1, 1))
    return _save(figure, output_dir, name)


def _process_components(
    rows: list[dict[str, str]], output_dir: Path
) -> list[Path]:
    mines = list(dict.fromkeys(row["mine"] for row in rows))
    x = np.arange(len(mines), dtype=float)
    width = 0.36
    panels = (
        ("ore_processed_kt", "Ore processed (kt)"),
        ("processed_grade_g_per_t", "Processed grade (g/t)"),
        ("recovery_pct", "Recovery (%)"),
    )
    figure, axes = plt.subplots(3, 1, figsize=(11.8, 10.2), sharex=True)
    for axis, (field, label) in zip(axes, panels):
        for offset, quarter in zip((-width / 2, width / 2), ("2026Q1", "2026Q2")):
            indexed = {
                row["mine"]: float(row[field])
                for row in rows
                if row["quarter"] == quarter
            }
            axis.bar(
                x + offset,
                [indexed[mine] for mine in mines],
                width,
                label=quarter,
                color=COLORS[quarter],
            )
        axis.set_ylabel(label)
        axis.grid(axis="y", alpha=0.22)
    axes[0].legend(frameon=False, ncol=2)
    axes[-1].set_xticks(x, mines, rotation=24, ha="right")
    figure.suptitle("Team 4 operating decomposition refreshed with Barrick 2026 actuals")
    figure.text(
        0.01,
        0.01,
        "Ore processed, grade and recovery are physical operating drivers; no gold-price simulation enters this figure.",
        fontsize=8,
    )
    figure.tight_layout(rect=(0, 0.04, 1, 0.97))
    return _save(figure, output_dir, "team4_process_components_qoq")


def _operating_contract(config: dict[str, Any], output_dir: Path) -> list[Path]:
    model = config["model"]
    production = np.asarray(model["production_koz"], dtype=float)
    costs = np.asarray(model["cost_usd_per_oz"], dtype=float)
    status = list(model["operating_vector_status"])
    if production.shape != (20,) or costs.shape != (20,) or len(status) != 20:
        raise ValueError("the operating contract must contain 20 quarterly observations")
    if status[:2] != ["ACTUAL_BARRICK_Q1_2026", "ACTUAL_BARRICK_Q2_2026"]:
        raise ValueError("the first two quarters must be current Barrick actuals")
    if any(value != "TEAM4_FORECAST" for value in status[2:]):
        raise ValueError("Team 4 forecasts must begin at 2026Q3")

    quarters = np.arange(1, 21)
    labels = [f"{2026 + (quarter - 1) // 4}Q{(quarter - 1) % 4 + 1}" for quarter in quarters]
    figure, axes = plt.subplots(2, 1, figsize=(11.5, 7.7), sharex=True)
    for axis, values, ylabel, color in (
        (axes[0], production, "Production (koz/quarter)", "#2A6F97"),
        (axes[1], costs, "Cost of Sales (USD/oz)", "#B5651D"),
    ):
        axis.plot(quarters[:2], values[:2], marker="o", linewidth=2.5, color="#1B4332", label="Barrick actual")
        axis.plot(quarters[1:], values[1:], marker="o", markersize=4, linestyle="--", linewidth=1.8, color=color, label="Team 4 forecast (from 2026Q3)")
        axis.axvline(2.5, color="#B33A3A", linestyle=":", linewidth=1.4)
        axis.axvspan(0.5, 2.5, color="#95D5B2", alpha=0.16)
        axis.set_ylabel(ylabel)
        axis.grid(alpha=0.22)
        axis.legend(frameon=False, loc="best")
    axes[-1].set_xticks(quarters, labels, rotation=55, ha="right", fontsize=8)
    figure.suptitle("Unified operating contract: current actuals and Team 4 forecast kept separate")
    figure.text(
        0.01,
        0.01,
        "The price layer is not shown: valuation gold paths are generated independently by the current Team 8 models.",
        fontsize=8,
    )
    figure.tight_layout(rect=(0, 0.05, 1, 0.97))
    return _save(figure, output_dir, "team4_operating_contract")


def build_team4_operating_figures(
    *,
    mine_csv: Path,
    totals_csv: Path,
    source_manifest: Path,
    valuation_config: Path,
    output_dir: Path,
    run_id: str,
) -> dict[str, Any]:
    """Build four operating figures and a hash-complete run manifest."""

    if output_dir.exists():
        raise FileExistsError(output_dir)
    output_dir.mkdir(parents=True)
    rows = _read_rows(mine_csv)
    totals = _read_rows(totals_csv)
    if len(rows) != 16 or {row["quarter"] for row in rows} != {"2026Q1", "2026Q2"}:
        raise ValueError("expected eight mines for each of two quarters")
    if len(totals) != 2:
        raise ValueError("expected exactly two company-quarter totals")
    for row in rows:
        for field in (
            "attributable_production_koz",
            "ore_processed_kt",
            "processed_grade_g_per_t",
            "recovery_pct",
            "cost_of_sales_usd_per_oz",
        ):
            if not np.isfinite(float(row[field])) or float(row[field]) <= 0:
                raise ValueError(f"invalid {field} for {row['mine']} {row['quarter']}")
    for quarter in ("2026Q1", "2026Q2"):
        subset = sum(
            float(row["attributable_production_koz"])
            for row in rows
            if row["quarter"] == quarter
        )
        total = float(next(row for row in totals if row["quarter"] == quarter)["total_attributable_gold_production_koz"])
        if subset > total:
            raise ValueError(f"selected-mine production exceeds Barrick total in {quarter}")

    artifacts: list[Path] = []
    artifacts += _grouped_bars(
        rows,
        "attributable_production_koz",
        "Attributable production (koz)",
        "Selected-mine attributable gold production: Q1 vs Q2 2026",
        output_dir,
        "team4_mine_production_qoq",
    )
    artifacts += _grouped_bars(
        rows,
        "cost_of_sales_usd_per_oz",
        "Cost of Sales (USD/oz)",
        "Selected-mine Cost of Sales per ounce: Q1 vs Q2 2026",
        output_dir,
        "team4_mine_cost_qoq",
    )
    artifacts += _process_components(rows, output_dir)
    config = json.loads(valuation_config.read_text(encoding="utf-8"))
    artifacts += _operating_contract(config, output_dir)

    manifest = {
        "schema_version": "1.0",
        "run_id": run_id,
        "status": "CURRENT_OPERATIONS_WITH_EXPLICIT_FORECAST_HANDOFF",
        "separation_policy": {
            "team4_operating_method_used": True,
            "team4_price_simulation_used": False,
            "team4_illustrative_valuation_used": False,
            "valuation_price_layer": "current Team 8 calibration",
        },
        "upstream_method_code": {
            "repository": "StartingFinanceClubPoliTo/Research",
            "commit": "f0c77feff23944819f8bd63abe7b6244b348cc77",
            "path": "Barrick-Gold/Component-Driven-EBITDA-Barrick",
            "note": "The upstream notebooks/R script informed the retained operating decomposition; this module is the autonomous current-data renderer.",
        },
        "inputs": [
            {"path": str(path), "sha256": _sha256(path)}
            for path in (mine_csv, totals_csv, source_manifest, valuation_config)
        ],
        "runtime": {
            "python": platform.python_version(),
            "numpy": np.__version__,
            "matplotlib": matplotlib.__version__,
        },
        "artifacts": [
            {
                "path": path.name,
                "bytes": path.stat().st_size,
                "sha256": _sha256(path),
            }
            for path in artifacts
        ],
    }
    manifest_path = output_dir / "figure_manifest.json"
    manifest_path.write_text(
        json.dumps(manifest, indent=2, ensure_ascii=False) + "\n",
        encoding="utf-8",
    )
    return {**manifest, "manifest_path": str(manifest_path)}
