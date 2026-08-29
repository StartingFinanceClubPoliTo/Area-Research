"""Reproducible, sidecar-first figures for the Barrick article."""

from __future__ import annotations

import csv
from dataclasses import dataclass
import json
from pathlib import Path
import platform
import sys
from typing import Any

import matplotlib

matplotlib.use("Agg")
import matplotlib.pyplot as plt
import numpy as np
import scipy
from scipy.stats import gaussian_kde

from barrick_unified.multimodel_valuation import MODEL_ORDER, MultiModelRun
from barrick_unified.valuation import QUANTILE_LEVELS

from ..domain.provenance import file_sha256


COLORS = {
    "black_scholes": "#1F77B4",
    "heston": "#2A9D8F",
    "bates_poisson": "#D17B0F",
    "full_bates_hawkes": "#6F42C1",
}


@dataclass(frozen=True)
class FigureSpec:
    figure_id: str
    score: int
    chapter: str
    title: str


SPECS = (
    FigureSpec("fig_val_primary", 10, "Discoveries", "Full Bates--Hawkes conditional Barrick value"),
    FigureSpec("fig_gold_bands", 10, "Chapter 9", "Conditional gold path bands by model"),
    FigureSpec("fig_dcf_drivers", 10, "Chapter 11", "Growth, ROIC and reinvestment assumptions"),
    FigureSpec("fig_wacc_assumptions", 10, "Chapter 11", "Common mean-reverting WACC contract"),
    FigureSpec("fig_terminal_value", 10, "Chapter 11", "Terminal-value mechanics and contribution"),
    FigureSpec("fig_val_multi", 10, "Chapter 12", "Conditional Barrick value by gold-price model"),
)


def _write_csv(path: Path, rows: list[dict[str, Any]]) -> None:
    if not rows:
        raise ValueError("figure sidecar cannot be empty")
    with path.open("w", encoding="utf-8", newline="") as handle:
        writer = csv.DictWriter(handle, fieldnames=list(rows[0]))
        writer.writeheader()
        writer.writerows(rows)


def _write_json(path: Path, payload: dict[str, Any]) -> None:
    path.write_text(json.dumps(payload, indent=2, ensure_ascii=False) + "\n", encoding="utf-8")


def _save(figure, root: Path, figure_id: str) -> tuple[Path, Path]:
    png = root / f"{figure_id}.png"
    pdf = root / f"{figure_id}.pdf"
    figure.savefig(png, dpi=240, bbox_inches="tight", facecolor="white")
    figure.savefig(pdf, bbox_inches="tight", facecolor="white")
    plt.close(figure)
    return png, pdf


def _summary(run: MultiModelRun, model_id: str) -> dict[str, Any]:
    values = run.models[model_id].valuation.value_per_share_proxy_usd
    observed = run.inputs.observed_share_price_usd
    median = float(np.median(values))
    return {
        "model_id": model_id,
        "model": run.models[model_id].label,
        "p10_usd_per_share": float(np.percentile(values, 10)),
        "median_usd_per_share": median,
        "p90_usd_per_share": float(np.percentile(values, 90)),
        "observed_close_usd_per_share": observed,
        "observed_close_timestamp_utc": run.inputs.observed_share_price_timestamp_utc,
        "observed_close_percentile": float(100.0 * np.mean(values <= observed)),
        "probability_value_above_close": float(np.mean(values > observed)),
        "observed_minus_median_usd": observed - median,
        "observed_vs_median_pct": (observed / median - 1.0) * 100.0,
    }


class ThesisFigureService:
    def build(
        self,
        *,
        run: MultiModelRun,
        output_dir: Path,
        run_id: str,
        input_paths: list[Path],
        code_paths: list[Path],
    ) -> dict[str, Any]:
        output_dir.mkdir(parents=True, exist_ok=False)
        artifacts: list[dict[str, Any]] = []
        for spec in SPECS:
            if spec.figure_id == "fig_val_primary":
                paths = self._primary(run, output_dir, spec)
            elif spec.figure_id == "fig_gold_bands":
                paths = self._gold_bands(run, output_dir, spec)
            elif spec.figure_id == "fig_val_multi":
                paths = self._multi(run, output_dir, spec)
            elif spec.figure_id == "fig_dcf_drivers":
                paths = self._dcf_drivers(run, output_dir, spec)
            elif spec.figure_id == "fig_wacc_assumptions":
                paths = self._wacc_assumptions(run, output_dir, spec)
            elif spec.figure_id == "fig_terminal_value":
                paths = self._terminal_value(run, output_dir, spec)
            else:  # pragma: no cover
                raise AssertionError(spec.figure_id)
            for path in paths:
                artifacts.append(self._record(path, output_dir.parent))
        manifest = {
            "schema_version": "1.0",
            "run_id": run_id,
            "status": "FIGURE_OUTPUT_CANDIDATE",
            "figure_specs": [spec.__dict__ for spec in SPECS],
            "model_boundary": {
                "gold_models_only": list(MODEL_ORDER),
                "operations": "BARRICK_2026Q1_Q2_ACTUAL_PLUS_TEAM4_FORECAST_FROM_2026Q3",
                "dcf_and_bridge": "UNIFIED_CONDITIONAL_WITH_UNRESOLVED_CORPORATE_PROXIES",
                "team4_price_demonstrator": "EXCLUDED",
                "conditional_bridge": run.bridge,
            },
            "market_overlay": {
                "provider": "London Strategic Edge",
                "ticker": "B",
                "close_usd": run.inputs.observed_share_price_usd,
                "timestamp_utc": run.inputs.observed_share_price_timestamp_utc,
            },
            "calibration_snapshot": run.calibration_snapshot,
            "asynchronous_dates": run.asynchronous_dates,
            "runtime": {
                "python": platform.python_version(),
                "numpy": np.__version__,
                "scipy": scipy.__version__,
                "matplotlib": matplotlib.__version__,
                "platform": sys.platform,
            },
            "inputs": [self._external_record(path) for path in input_paths],
            "code": [self._external_record(path) for path in code_paths],
            "artifacts": artifacts,
            "publication_gate": {
                "raw_lse_included": False,
                "updated_figures": [spec.figure_id for spec in SPECS],
                "excluded_not_current": [
                    "team4_price_iv_gbm_and_illustrative_ebitda_demonstrator",
                    "fig_bh_residuals",
                    "fig_bh_smile",
                    "fig_bh_vs_bates",
                ],
            },
        }
        manifest_path = output_dir.parent / "figure_manifest.json"
        _write_json(manifest_path, manifest)
        return {**manifest, "manifest_path": str(manifest_path), "manifest_sha256": file_sha256(manifest_path)}

    @staticmethod
    def _record(path: Path, root: Path) -> dict[str, Any]:
        return {
            "path": path.relative_to(root).as_posix(),
            "bytes": path.stat().st_size,
            "sha256": file_sha256(path),
        }

    @staticmethod
    def _external_record(path: Path) -> dict[str, Any]:
        resolved = path.resolve()
        return {"path": str(resolved), "bytes": resolved.stat().st_size, "sha256": file_sha256(resolved)}

    def _primary(self, run: MultiModelRun, root: Path, spec: FigureSpec) -> list[Path]:
        model = run.models["full_bates_hawkes"]
        values = model.valuation.value_per_share_proxy_usd
        observed = run.inputs.observed_share_price_usd
        lower, upper = np.percentile(values, [0.5, 99.5])
        lower, upper = min(float(lower), observed), max(float(upper), observed)
        shown = values[(values >= lower) & (values <= upper)]
        facts = _summary(run, "full_bates_hawkes")
        figure, axis = plt.subplots(figsize=(10.4, 5.7))
        axis.hist(shown, bins=90, density=True, color="#DCE6F1", edgecolor="white", linewidth=0.35)
        axis.axvspan(facts["p10_usd_per_share"], facts["p90_usd_per_share"], color="#D4AF37", alpha=0.18, label="Conditional P10--P90")
        axis.axvline(facts["median_usd_per_share"], color="#17365D", linewidth=2.2, label=f"Median ${facts['median_usd_per_share']:.2f}")
        axis.axvline(observed, color="#B33A3A", linestyle="--", linewidth=2.2, label=f"B close ${observed:.2f} ({run.inputs.observed_share_price_timestamp_utc[:10]})")
        axis.set(title="Barrick conditional value distribution - Full Bates--Hawkes gold layer", xlabel="Conditional value (USD/share)", ylabel="Density")
        axis.grid(axis="y", alpha=0.2)
        axis.legend(frameon=False)
        figure.text(0.01, 0.01, "Gold layer: current Team 8 GLD/Q calibration. Operations use Barrick Q1--Q2 actuals plus Team 4 forecasts from Q3; the Team 4 price demonstrator is excluded. Not fair value or a target price.", fontsize=8)
        figure.tight_layout(rect=(0, 0.04, 1, 1))
        png, pdf = _save(figure, root, spec.figure_id)
        csv_path = root / f"{spec.figure_id}.csv"
        _write_csv(csv_path, [{"path_id": index, "value_usd_per_share": float(value)} for index, value in enumerate(values)])
        json_path = root / f"{spec.figure_id}.json"
        _write_json(json_path, {"figure_spec": spec.__dict__, "facts": facts, "display_range_usd_per_share": [float(lower), float(upper)], "n_paths": int(values.size), "operations_status": "CURRENT_Q1_Q2_PLUS_TEAM4_FORECAST_Q3_ONWARD", "team4_price_demonstrator": "EXCLUDED", "corporate_bridge_status": "UNRESOLVED", "interpretation": "conditional sensitivity; observed close above conditional median does not establish overvaluation"})
        return [png, pdf, csv_path, json_path]

    def _multi(self, run: MultiModelRun, root: Path, spec: FigureSpec) -> list[Path]:
        observed = run.inputs.observed_share_price_usd
        all_values = np.concatenate([run.models[mid].valuation.value_per_share_proxy_usd for mid in MODEL_ORDER])
        lower, upper = np.percentile(all_values, [0.5, 99.5])
        lower, upper = min(float(lower), observed), max(float(upper), observed)
        grid = np.linspace(lower, upper, 360)
        rows: list[dict[str, Any]] = []
        summaries = []
        figure, axis = plt.subplots(figsize=(10.7, 6.0))
        for model_id in MODEL_ORDER:
            model = run.models[model_id]
            values = model.valuation.value_per_share_proxy_usd
            shown = values[(values >= lower) & (values <= upper)]
            density = gaussian_kde(shown, bw_method="scott")(grid)
            summary = _summary(run, model_id)
            summaries.append(summary)
            axis.plot(grid, density, color=COLORS[model_id], linewidth=2.0, label=f"{model.label} (P50 ${summary['median_usd_per_share']:.2f})")
            rows.extend({"model_id": model_id, "path_id": index, "value_usd_per_share": float(value)} for index, value in enumerate(values))
        axis.axvline(observed, color="#B33A3A", linestyle="--", linewidth=2.1, label=f"B close ${observed:.2f} ({run.inputs.observed_share_price_timestamp_utc[:10]})")
        axis.set(title="Barrick conditional value distributions by gold-price model", xlabel="Conditional value (USD/share)", ylabel="Density")
        axis.grid(axis="y", alpha=0.2)
        axis.legend(frameon=False, fontsize=9)
        figure.text(0.01, 0.01, "Only the Team 8 gold-price engine changes. The separated operating contract, unified DCF/WACC and unresolved corporate bridge are common.", fontsize=8)
        figure.tight_layout(rect=(0, 0.04, 1, 1))
        png, pdf = _save(figure, root, spec.figure_id)
        csv_path = root / f"{spec.figure_id}.csv"
        _write_csv(csv_path, rows)
        json_path = root / f"{spec.figure_id}.json"
        _write_json(json_path, {"figure_spec": spec.__dict__, "summaries": summaries, "display_range_usd_per_share": [float(lower), float(upper)], "n_paths_per_model": int(run.inputs.n_simulations), "interpretation": "controlled model-risk comparison, not four independent company models"})
        return [png, pdf, csv_path, json_path]

    def _gold_bands(self, run: MultiModelRun, root: Path, spec: FigureSpec) -> list[Path]:
        rows: list[dict[str, Any]] = []
        figure, axes = plt.subplots(2, 2, figsize=(11.0, 7.8), sharex=True, sharey=True)
        quarters = np.arange(1, run.inputs.n_quarters + 1)
        for axis, model_id in zip(axes.flat, MODEL_ORDER):
            model = run.models[model_id]
            paths = model.quarterly_gold_paths
            p10, p50, p90 = np.percentile(paths, [10, 50, 90], axis=0)
            axis.fill_between(quarters, p10, p90, color=COLORS[model_id], alpha=0.18, label="P10--P90")
            axis.plot(quarters, p50, color=COLORS[model_id], linewidth=2.0, label="P50")
            axis.axhline(run.inputs.gold_price_0_usd_per_oz, color="#555555", linestyle=":", linewidth=1.0, label="Common start")
            axis.set_title(model.label)
            axis.grid(alpha=0.2)
            axis.legend(frameon=False, fontsize=8)
            for quarter, lo, med, hi in zip(quarters, p10, p50, p90):
                rows.append({"model_id": model_id, "quarter": int(quarter), "p10_gold_usd_per_oz": float(lo), "p50_gold_usd_per_oz": float(med), "p90_gold_usd_per_oz": float(hi)})
        for axis in axes[-1, :]:
            axis.set_xlabel("Quarter")
        for axis in axes[:, 0]:
            axis.set_ylabel("Gold scenario (USD/troy oz)")
        figure.suptitle("Conditional gold path bands - four option-implied engines", fontsize=14)
        figure.text(0.01, 0.01, "Current risk-neutral GLD option shape transferred conditionally to an independently sourced Barrick realized-price anchor; no Team 4 price path is used. Not a physical forecast. 8,192 paths, 20 quarters.", fontsize=8)
        figure.tight_layout(rect=(0, 0.035, 1, 0.965))
        png, pdf = _save(figure, root, spec.figure_id)
        csv_path = root / f"{spec.figure_id}.csv"
        _write_csv(csv_path, rows)
        json_path = root / f"{spec.figure_id}.json"
        _write_json(json_path, {"figure_spec": spec.__dict__, "n_paths": int(run.inputs.n_simulations), "quarters": int(run.inputs.n_quarters), "start_gold_usd_per_oz": run.inputs.gold_price_0_usd_per_oz, "calibration_snapshot": run.calibration_snapshot, "measure": "conditional risk-neutral shape; Q-to-P NOT_VALIDATED", "bands": [10, 50, 90]})
        return [png, pdf, csv_path, json_path]

    def _dcf_drivers(self, run: MultiModelRun, root: Path, spec: FigureSpec) -> list[Path]:
        inputs = run.inputs
        years = np.arange(1, inputs.n_years + 1)
        growth = np.linspace(inputs.high_growth, inputs.stable_growth, inputs.n_years)
        roic = np.linspace(inputs.roic_high, inputs.roic_stable, inputs.n_years)
        reinvestment = growth / roic
        rows = [
            {
                "year": int(year),
                "growth_decimal": float(g),
                "roic_decimal": float(r),
                "reinvestment_rate_decimal": float(rr),
                "tax_rate_decimal": float(inputs.tax_rate),
            }
            for year, g, r, rr in zip(years, growth, roic, reinvestment)
        ]
        figure, axes = plt.subplots(1, 2, figsize=(11.0, 4.9))
        axes[0].plot(years, growth * 100.0, marker="o", linewidth=2.2, label="Growth")
        axes[0].plot(years, roic * 100.0, marker="o", linewidth=2.2, label="ROIC")
        axes[0].set(title="Linear convergence assumptions", xlabel="Explicit forecast year", ylabel="Annual rate (%)")
        axes[0].set_xticks(years)
        axes[0].grid(alpha=0.25)
        axes[0].legend(frameon=False)
        axes[1].plot(years, reinvestment * 100.0, color="#D17B0F", marker="o", linewidth=2.2)
        axes[1].set(title="Implied reinvestment = growth / ROIC", xlabel="Explicit forecast year", ylabel="Reinvestment rate (%)")
        axes[1].set_xticks(years)
        axes[1].grid(alpha=0.25)
        figure.suptitle("Unified DCF driver schedule read from the current valuation config", fontsize=14)
        figure.text(
            0.01,
            0.01,
            "Editable fields: model.high_growth, model.stable_growth, model.roic_high, model.roic_stable and model.tax_rate. "
            "These are methodological assumptions shared by all four gold engines, not filing-calibrated Barrick estimates.",
            fontsize=8,
        )
        figure.tight_layout(rect=(0, 0.06, 1, 0.94))
        png, pdf = _save(figure, root, spec.figure_id)
        csv_path = root / f"{spec.figure_id}.csv"
        _write_csv(csv_path, rows)
        json_path = root / f"{spec.figure_id}.json"
        _write_json(
            json_path,
            {
                "figure_spec": spec.__dict__,
                "config_fields": {
                    "model.high_growth": inputs.high_growth,
                    "model.stable_growth": inputs.stable_growth,
                    "model.roic_high": inputs.roic_high,
                    "model.roic_stable": inputs.roic_stable,
                    "model.tax_rate": inputs.tax_rate,
                },
                "formula": "reinvestment_rate_t = growth_t / ROIC_t; FCFF_t = after_tax_margin_t - max(after_tax_margin_t, 0) * reinvestment_rate_t",
                "scope": "common DCF layer, identical across gold-price models",
            },
        )
        return [png, pdf, csv_path, json_path]

    def _wacc_assumptions(self, run: MultiModelRun, root: Path, spec: FigureSpec) -> list[Path]:
        inputs = run.inputs
        years = np.arange(0, inputs.n_years + 1)
        common = run.models[MODEL_ORDER[0]].valuation.annual_wacc
        p10, p50, p90 = np.percentile(common, [10, 50, 90], axis=0)
        p10 = np.r_[inputs.wacc_0, p10]
        p50 = np.r_[inputs.wacc_0, p50]
        p90 = np.r_[inputs.wacc_0, p90]
        expected = np.empty(inputs.n_years + 1)
        expected[0] = inputs.wacc_0
        persistence = np.exp(-inputs.wacc_reversion)
        floor = inputs.stable_growth + inputs.terminal_spread_floor
        for year in range(inputs.n_years):
            expected[year + 1] = inputs.wacc_long_run + (
                expected[year] - inputs.wacc_long_run
            ) * persistence
        expected[1:] = np.clip(expected[1:], floor, 0.25)
        rows = [
            {
                "year": int(year),
                "expected_wacc_decimal": float(mean),
                "p10_wacc_decimal": float(lo),
                "p50_wacc_decimal": float(med),
                "p90_wacc_decimal": float(hi),
                "long_run_wacc_decimal": float(inputs.wacc_long_run),
                "wacc_floor_decimal": float(floor),
            }
            for year, mean, lo, med, hi in zip(years, expected, p10, p50, p90)
        ]
        figure, axis = plt.subplots(figsize=(10.7, 5.2))
        axis.fill_between(years, p10 * 100.0, p90 * 100.0, color="#5B8DB8", alpha=0.22, label="P10--P90 simulated WACC")
        axis.plot(years, p50 * 100.0, color="#17365D", marker="o", linewidth=2.2, label="Simulated median")
        axis.plot(years, expected * 100.0, color="#D17B0F", linestyle="--", marker="s", linewidth=2.0, label="Zero-shock expected path")
        axis.axhline(inputs.wacc_long_run * 100.0, color="#2A9D8F", linestyle=":", linewidth=2.0, label="Long-run WACC")
        axis.axhline(floor * 100.0, color="#B33A3A", linestyle=":", linewidth=1.8, label="Floor = stable growth + spread floor")
        axis.set(title="Common mean-reverting WACC used in every valuation branch", xlabel="Year (0 is valuation date)", ylabel="WACC (%)")
        axis.set_xticks(years)
        axis.grid(alpha=0.25)
        axis.legend(frameon=False, ncol=2)
        figure.text(
            0.01,
            0.01,
            "Editable fields: model.wacc_0, model.wacc_long_run, model.wacc_reversion, model.wacc_volatility and model.terminal_spread_floor. "
            "All four gold models reuse the same 8,192-path WACC shock array.",
            fontsize=8,
        )
        figure.tight_layout(rect=(0, 0.06, 1, 1))
        png, pdf = _save(figure, root, spec.figure_id)
        csv_path = root / f"{spec.figure_id}.csv"
        _write_csv(csv_path, rows)
        json_path = root / f"{spec.figure_id}.json"
        _write_json(
            json_path,
            {
                "figure_spec": spec.__dict__,
                "config_fields": {
                    "model.wacc_0": inputs.wacc_0,
                    "model.wacc_long_run": inputs.wacc_long_run,
                    "model.wacc_reversion": inputs.wacc_reversion,
                    "model.wacc_volatility": inputs.wacc_volatility,
                    "model.terminal_spread_floor": inputs.terminal_spread_floor,
                },
                "annual_transition": "WACC[t+1] = long_run + (WACC[t] - long_run) * exp(-reversion) + conditional_std * shock[t]",
                "conditional_std": float(
                    inputs.wacc_volatility
                    * np.sqrt(
                        (1.0 - np.exp(-2.0 * inputs.wacc_reversion))
                        / (2.0 * inputs.wacc_reversion)
                    )
                ),
                "clip": {"minimum": float(floor), "maximum": 0.25},
                "wacc_shocks_sha256": run.wacc_shocks_sha256,
            },
        )
        return [png, pdf, csv_path, json_path]

    def _terminal_value(self, run: MultiModelRun, root: Path, spec: FigureSpec) -> list[Path]:
        inputs = run.inputs
        growth_grid = np.linspace(0.005, 0.045, 81)
        wacc_grid = np.linspace(0.06, 0.12, 91)
        g_mesh, w_mesh = np.meshgrid(growth_grid, wacc_grid)
        multiple = (1.0 + g_mesh) * (1.0 - g_mesh / inputs.roic_stable) / (w_mesh - g_mesh)
        multiple[w_mesh <= g_mesh + inputs.terminal_spread_floor] = np.nan
        terminal_wacc = run.models[MODEL_ORDER[0]].valuation.annual_wacc[:, -1]
        central_terminal_wacc = float(np.median(terminal_wacc))
        contribution_rows = []
        for model_id in MODEL_ORDER:
            valuation = run.models[model_id].valuation
            explicit = valuation.pv_explicit_fcff_proxy_usd_mn
            terminal = valuation.pv_terminal_proxy_usd_mn
            enterprise = valuation.enterprise_value_proxy_usd_mn
            admitted = (enterprise > 0.0) & (explicit >= 0.0) & (terminal >= 0.0)
            shares = terminal[admitted] / enterprise[admitted]
            lo, med, hi = np.percentile(shares, [10, 50, 90])
            contribution_rows.append(
                {
                    "model_id": model_id,
                    "label": run.models[model_id].label,
                    "eligible_paths": int(admitted.sum()),
                    "eligible_fraction": float(np.mean(admitted)),
                    "p10_terminal_share": float(lo),
                    "p50_terminal_share": float(med),
                    "p90_terminal_share": float(hi),
                }
            )
        figure, axes = plt.subplots(1, 2, figsize=(11.2, 5.2))
        levels = np.linspace(np.nanpercentile(multiple, 5), np.nanpercentile(multiple, 95), 18)
        contour = axes[0].contourf(growth_grid * 100.0, wacc_grid * 100.0, multiple, levels=levels, cmap="viridis")
        figure.colorbar(contour, ax=axes[0], label="Terminal value / final after-tax margin")
        axes[0].scatter([inputs.stable_growth * 100.0], [central_terminal_wacc * 100.0], color="#B33A3A", s=60, marker="x", linewidths=2.5, label="Current g and median terminal WACC")
        axes[0].set(title="Terminal-value multiple sensitivity", xlabel="Stable growth (%)", ylabel="Terminal WACC (%)")
        axes[0].legend(frameon=False, fontsize=8)
        labels = [row["label"] for row in contribution_rows]
        x = np.arange(len(labels))
        medians = np.array([row["p50_terminal_share"] for row in contribution_rows])
        lows = medians - np.array([row["p10_terminal_share"] for row in contribution_rows])
        highs = np.array([row["p90_terminal_share"] for row in contribution_rows]) - medians
        axes[1].bar(x, medians * 100.0, color=[COLORS[mid] for mid in MODEL_ORDER], alpha=0.85)
        axes[1].errorbar(x, medians * 100.0, yerr=np.vstack([lows, highs]) * 100.0, fmt="none", ecolor="#222222", capsize=4)
        axes[1].set_xticks(x, ["BS/GBM", "Heston", "Bates", "Full BH"])
        axes[1].set(title="PV terminal share of positive EV paths", ylabel="Terminal share of EV (%)")
        axes[1].grid(axis="y", alpha=0.25)
        figure.suptitle("How the terminal block enters the unified Barrick valuation", fontsize=14)
        figure.text(
            0.01,
            0.01,
            "TV = final after-tax margin × (1+g) × (1-g/ROIC stable) / (terminal WACC-g), then discounted to present. "
            "The left panel is a normalized sensitivity; the right panel uses the actual four-model run and excludes non-positive component paths.",
            fontsize=8,
        )
        figure.tight_layout(rect=(0, 0.075, 1, 0.94))
        png, pdf = _save(figure, root, spec.figure_id)
        csv_path = root / f"{spec.figure_id}.csv"
        _write_csv(csv_path, contribution_rows)
        json_path = root / f"{spec.figure_id}.json"
        current_multiple = (
            (1.0 + inputs.stable_growth)
            * (1.0 - inputs.stable_growth / inputs.roic_stable)
            / (central_terminal_wacc - inputs.stable_growth)
        )
        _write_json(
            json_path,
            {
                "figure_spec": spec.__dict__,
                "config_fields": {
                    "model.stable_growth": inputs.stable_growth,
                    "model.roic_stable": inputs.roic_stable,
                    "model.terminal_spread_floor": inputs.terminal_spread_floor,
                },
                "median_terminal_wacc": central_terminal_wacc,
                "current_normalized_terminal_multiple": float(current_multiple),
                "formula": "terminal_fcff = max(final_after_tax_margin, 0) * (1 + stable_growth) * (1 - stable_growth / roic_stable); terminal_value = terminal_fcff / (terminal_wacc - stable_growth)",
                "contribution_summary": contribution_rows,
                "interpretation": "methodological sensitivity and actual-run decomposition; not a filing-reconciled Barrick fair value",
            },
        )
        return [png, pdf, csv_path, json_path]
