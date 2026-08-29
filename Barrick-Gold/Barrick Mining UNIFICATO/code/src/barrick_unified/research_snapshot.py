"""Build publication-safe aggregate tables, figures and a result manifest."""

from __future__ import annotations

import hashlib
import json
from datetime import datetime, timezone
from pathlib import Path
from typing import Any

import matplotlib

matplotlib.use("Agg")
import matplotlib.pyplot as plt
import numpy as np
import pandas as pd

from .market_data import (
    canonicalise_lse_candles,
    compute_market_summary,
    rolling_correlation,
    rolling_statistics,
)
from .team8_option_audit import audit_current_surface


SERIES_ROLES = {
    "B": "current Barrick Mining Corporation NYSE equity; USD/share",
    "GOLD": "historical NYSE ticker candidate; kept separate pending approved issuer/corporate-action stitching",
    "GLD": "SPDR Gold Shares ETF proxy; USD/share; never relabelled gold USD/oz or realized gold price",
    "ABX": "TSX Barrick listing candidate; currency/FX bridge not applied",
}


def sha256(path: Path) -> str:
    digest = hashlib.sha256()
    with path.open("rb") as handle:
        for chunk in iter(lambda: handle.read(1024 * 1024), b""):
            digest.update(chunk)
    return digest.hexdigest().upper()


def _write_json(path: Path, payload: dict[str, Any]) -> None:
    path.parent.mkdir(parents=True, exist_ok=True)
    path.write_text(json.dumps(payload, indent=2, ensure_ascii=False) + "\n", encoding="utf-8")


def _display_path(path: Path, project_root: Path) -> str:
    try:
        return path.relative_to(project_root).as_posix()
    except ValueError:
        return path.as_posix()


def _market_context_figure(
    bars_by_symbol: dict[str, pd.DataFrame], output: Path, window: int
) -> None:
    fig, axes = plt.subplots(2, 1, figsize=(10.5, 7.2), sharex=False)
    colors = {"B": "#0B5FA5", "GOLD": "#A86F00", "GLD": "#D4AF37", "ABX": "#6B4C9A"}
    for symbol, bars in bars_by_symbol.items():
        normalized = bars["close"] / bars["close"].iloc[0] * 100.0
        axes[0].plot(bars["timestamp"], normalized, label=symbol, color=colors.get(symbol), linewidth=1.5)
        rolling = rolling_statistics(bars, window=window)
        axes[1].plot(
            rolling["timestamp"],
            rolling["annualized_volatility"],
            label=symbol,
            color=colors.get(symbol),
            linewidth=1.3,
        )
    axes[0].set_title("LSE market context: separately indexed series")
    axes[0].set_ylabel("Index (first observation = 100)")
    axes[1].set_title(f"Trailing {window}-observation realized volatility")
    axes[1].set_ylabel("Annualized volatility")
    axes[1].set_xlabel("UTC observation date")
    for axis in axes:
        axis.grid(alpha=0.22)
        axis.legend(ncol=min(4, len(bars_by_symbol)), frameon=False)
    fig.text(
        0.01,
        0.005,
        "B/GOLD are not stitched; GLD is an ETF proxy in USD/share, not gold in USD/oz. Source: London Strategic Edge.",
        fontsize=8,
    )
    fig.tight_layout(rect=(0, 0.025, 1, 1))
    output.parent.mkdir(parents=True, exist_ok=True)
    fig.savefig(output, dpi=180, bbox_inches="tight")
    plt.close(fig)


def _correlation_figure(
    bars_by_symbol: dict[str, pd.DataFrame], output: Path, window: int
) -> int:
    fig, axis = plt.subplots(figsize=(10.5, 4.6))
    line_count = 0
    for symbol in ("GOLD", "B"):
        if symbol not in bars_by_symbol or "GLD" not in bars_by_symbol:
            continue
        frame = rolling_correlation(
            bars_by_symbol[symbol], bars_by_symbol["GLD"], window=window
        )
        valid = frame.dropna(subset=["rolling_correlation"])
        if valid.empty:
            continue
        axis.plot(
            valid["timestamp"],
            valid["rolling_correlation"],
            label=f"{symbol} vs GLD",
            linewidth=1.4,
        )
        line_count += 1
    axis.axhline(0.0, color="black", linewidth=0.8, alpha=0.5)
    axis.set_ylim(-1.0, 1.0)
    axis.set_title(f"Trailing {window}-observation equity/GLD return correlation")
    axis.set_ylabel("Correlation")
    axis.set_xlabel("UTC observation date")
    axis.grid(alpha=0.22)
    if line_count:
        axis.legend(frameon=False)
    else:
        axis.text(0.5, 0.5, "Insufficient overlapping observations", ha="center", va="center", transform=axis.transAxes)
    fig.text(0.01, 0.01, "No issuer-series stitching or FX conversion is applied.", fontsize=8)
    fig.tight_layout(rect=(0, 0.03, 1, 1))
    output.parent.mkdir(parents=True, exist_ok=True)
    fig.savefig(output, dpi=180, bbox_inches="tight")
    plt.close(fig)
    return line_count


def _option_surface_figure(sample: pd.DataFrame, spot: float, output: Path) -> None:
    fig, axis = plt.subplots(figsize=(8.5, 5.2))
    moneyness = sample["K"].to_numpy(dtype=float) / float(spot)
    scatter = axis.scatter(
        moneyness,
        sample["T"],
        c=sample["implied_vol"],
        cmap="viridis",
        s=42,
        edgecolors="black",
        linewidths=0.25,
    )
    axis.set_title("Current LSE GLD option surface: Team 8 filtered nodes")
    axis.set_xlabel("Strike / current GLD spot")
    axis.set_ylabel("Maturity (years)")
    axis.grid(alpha=0.18)
    colorbar = fig.colorbar(scatter, ax=axis)
    colorbar.set_label("LSE implied volatility")
    fig.text(
        0.01,
        0.01,
        "Visualization of derived filtered nodes; raw LSE rows remain local and are not distributed.",
        fontsize=8,
    )
    fig.tight_layout(rect=(0, 0.03, 1, 1))
    output.parent.mkdir(parents=True, exist_ok=True)
    fig.savefig(output, dpi=180, bbox_inches="tight")
    plt.close(fig)


def build_public_outputs(
    raw_snapshot_paths: dict[str, Path],
    snapshot: dict[str, Any],
    output_dir: Path,
    figure_dir: Path,
    manifest_path: Path,
    team8_root: Path,
    project_root: Path,
    cutoff: str,
    rolling_window: int = 63,
) -> dict[str, Any]:
    """Create aggregates only; never serialize canonical row-level bars."""

    candle_payload = snapshot.get("rows", {}).get("daily_stock_candles", {})
    catalog_entries = snapshot.get("catalog_entries", {})
    bars_by_symbol: dict[str, pd.DataFrame] = {}
    pre_entity_rows_discarded: dict[str, int] = {}
    for symbol, rows in candle_payload.items():
        bars = canonicalise_lse_candles(rows, symbol)
        first = catalog_entries.get(symbol, {}).get("first")
        if first:
            valid_from = pd.Timestamp(first, tz="UTC") if pd.Timestamp(first).tzinfo is None else pd.Timestamp(first).tz_convert("UTC")
            valid_from_date = valid_from.normalize()
            discarded = int(bars["timestamp"].lt(valid_from_date).sum())
            bars = bars.loc[bars["timestamp"].ge(valid_from_date)].reset_index(drop=True)
            pre_entity_rows_discarded[symbol] = discarded
            if bars.empty:
                raise ValueError(f"No {symbol} candles remain after catalog entity-validity cutoff {valid_from_date.date()}.")
        bars_by_symbol[symbol] = bars
    if not bars_by_symbol:
        raise ValueError("Snapshot contains no daily stock candles.")
    cutoff_date = pd.Timestamp(cutoff).date()
    post_cutoff = {
        symbol: bars["timestamp"].max().date().isoformat()
        for symbol, bars in bars_by_symbol.items()
        if bars["timestamp"].max().date() > cutoff_date
    }
    if post_cutoff:
        raise ValueError(f"LSE candles exceed the requested cutoff {cutoff}: {post_cutoff}")

    summaries = pd.DataFrame(
        [compute_market_summary(bars) for bars in bars_by_symbol.values()]
    ).sort_values("symbol")
    summaries["semantic_role"] = summaries["symbol"].map(SERIES_ROLES).fillna("unclassified LSE stock series")
    output_dir.mkdir(parents=True, exist_ok=True)
    summary_path = output_dir / "lse_market_summary.csv"
    summaries.to_csv(summary_path, index=False, float_format="%.10g")

    market_figure = figure_dir / "lse_market_context.png"
    correlation_figure = figure_dir / "lse_equity_gld_rolling_correlation.png"
    _market_context_figure(bars_by_symbol, market_figure, rolling_window)
    correlation_lines = _correlation_figure(
        bars_by_symbol, correlation_figure, rolling_window
    )

    option_payload = snapshot.get("rows", {}).get("gld_option_calls")
    yield_payload = snapshot.get("rows", {}).get("usd_treasury_yields")
    option_gate: dict[str, Any]
    option_figure = figure_dir / "gld_lse_option_surface_audit.png"
    if option_payload and yield_payload:
        try:
            option_audit, sample, spot = audit_current_surface(
                option_payload, yield_payload, team8_root
            )
            option_as_of = pd.Timestamp(option_audit.as_of_utc).date()
            curve_as_of = pd.Timestamp(option_audit.treasury_curve_date).date()
            if option_as_of > cutoff_date or curve_as_of > cutoff_date:
                option_gate = {
                    "status": "BLOCKED_POST_CUTOFF",
                    "reason": "Current-only LSE endpoints returned observations after the requested cutoff.",
                    **option_audit.as_dict(),
                }
            else:
                _option_surface_figure(sample, spot, option_figure)
                option_gate = {"status": "PASS_CURRENT_REFRESH_ONLY", **option_audit.as_dict()}
        except (ValueError, RuntimeError, KeyError) as exc:
            option_gate = {"status": "BLOCKED", "reason": str(exc)}
    else:
        option_gate = {
            "status": "BLOCKED",
            "reason": "Current LSE option or Treasury endpoint returned no rows.",
        }

    artifacts = [summary_path, market_figure, correlation_figure]
    if option_figure.exists() and option_gate["status"] == "PASS_CURRENT_REFRESH_ONLY":
        artifacts.append(option_figure)
    raw_evidence = {
        name: {
            "path": path.relative_to(project_root).as_posix(),
            "sha256": sha256(path),
            "bytes": path.stat().st_size,
            "redistribution": "row-level data are local-only and Git-ignored",
        }
        for name, path in raw_snapshot_paths.items()
    }
    run_id = str(snapshot.get("run_id"))
    common_source = {
        "provider": "London Strategic Edge",
        "lse_data_version": snapshot.get("lse_data_version"),
        "pipeline_version": snapshot.get("pipeline_version"),
        "pipeline_script_sha256": snapshot.get("pipeline_script_sha256"),
        "run_id": run_id,
        "source_run_id": snapshot.get("source_run_id", run_id),
        "cutoff_utc_date": cutoff,
    }
    module_paths = {
        "market_data": Path(__file__).with_name("market_data.py"),
        "research_snapshot": Path(__file__),
        "team8_option_audit": Path(__file__).with_name("team8_option_audit.py"),
        "lse_adapter_current": Path(__file__).with_name("lse_adapter.py"),
    }
    frozen_team8_path = team8_root / "lse_dataset.py"
    code_provenance = {
        "acquisition_runner_sha256": snapshot.get("pipeline_script_sha256"),
        "transformation_modules": {
            name: {
                "path": _display_path(path, project_root),
                "sha256": sha256(path),
            }
            for name, path in module_paths.items()
        },
        "frozen_team8_lse_dataset": {
            "path": _display_path(frozen_team8_path, project_root),
            "sha256": sha256(frozen_team8_path) if frozen_team8_path.is_file() else None,
        },
    }
    equity_symbols = [symbol for symbol in summaries["symbol"] if symbol != "GLD"]
    equity_manifest_path = manifest_path.parent / "barrick_equity_manifest.json"
    equity_manifest = {
        "schema_version": "1.0",
        "dataset": "Barrick_equity_series",
        **common_source,
        "endpoint": {
            "method": "candles",
            "dataset": "stocks",
            "timeframe": "1d",
            "symbols": equity_symbols,
            "start": snapshot.get("request", {}).get("start"),
            "end_inclusive_cutoff": cutoff,
            "order": "asc",
            "limit": 5000,
        },
        "raw_snapshot": raw_evidence.get("barrick_equity"),
        "series": [
            {
                "symbol": row["symbol"],
                "bars": int(row["bars"]),
                "start_utc": row["start_utc"],
                "end_utc": row["end_utc"],
                "semantic_role": row["semantic_role"],
            }
            for row in summaries.to_dict(orient="records")
            if row["symbol"] != "GLD"
        ],
        "stitching": "NOT_APPLIED; B/GOLD/ABX remain separate until issuer, corporate-action and FX rules are approved",
        "aggregate_output": {
            "path": summary_path.relative_to(project_root).as_posix(),
            "sha256": sha256(summary_path),
        },
    }
    _write_json(equity_manifest_path, equity_manifest)
    option_manifest_path = manifest_path.parent / "gld_option_surface_manifest.json"
    option_manifest = {
        "schema_version": "1.0",
        "dataset": "GLD_current_option_surface_audit",
        **common_source,
        "endpoint": {
            "method": "options",
            "underlying": "GLD",
            "type": "call",
            "max_dte": snapshot.get("request", {}).get("gld_max_dte"),
            "limit": 5000,
            "historical_cutoff_parameter_available": False,
            "observed_as_of_must_not_exceed_cutoff": True,
        },
        "treasury_endpoint": {
            "method": "bond_yields",
            "symbols": ["US1M", "US2M", "US3M", "US6M", "US1Y", "US2Y", "US3Y", "US5Y"],
            "lookback_days": snapshot.get("request", {}).get("yield_lookback_days"),
            "order": "asc",
            "limit_per_tenor": 5000,
        },
        "raw_snapshot": raw_evidence.get("gld_market_inputs"),
        "gate": option_gate,
        "historical_G1_5": "NOT_PROVEN; a current endpoint call cannot replace the frozen 2026-08-12 raw snapshot",
    }
    _write_json(option_manifest_path, option_manifest)

    manifest = {
        "schema_version": "1.0",
        **common_source,
        "generated_at_utc": snapshot.get("generated_at_utc"),
        "transformed_at_utc": datetime.now(timezone.utc).isoformat(),
        "code_provenance": code_provenance,
        "request": snapshot.get("request", {}),
        "raw_snapshots": raw_evidence,
        "dataset_manifests": {
            "barrick_equity": {
                "path": equity_manifest_path.relative_to(project_root).as_posix(),
                "sha256": sha256(equity_manifest_path),
            },
            "gld_option_surface": {
                "path": option_manifest_path.relative_to(project_root).as_posix(),
                "sha256": sha256(option_manifest_path),
            },
        },
        "market_series": [
            {
                "symbol": row["symbol"],
                "bars": int(row["bars"]),
                "start_utc": row["start_utc"],
                "end_utc": row["end_utc"],
                "semantic_role": row["semantic_role"],
                "pre_entity_rows_discarded": pre_entity_rows_discarded.get(row["symbol"], 0),
            }
            for row in summaries.to_dict(orient="records")
        ],
        "statistics_contract": {
            "returns": "close-to-close log returns",
            "annualization": 252,
            "rolling_window": rolling_window,
            "alignment": "trailing and inclusive at t; no look-ahead",
            "missing_data": "no price or volume forward fill",
            "historical_var": "empirical 5% log-return quantile; descriptive, not a forecast",
            "ticker_reuse_guard": "stock rows before the current catalog entry's first timestamp are excluded",
        },
        "option_surface_gate": option_gate,
        "historical_parity_boundary": {
            "G1_4": "Team 8 unit/benchmark parity is independent of this refresh",
            "G1_5": "NOT_PROVEN; current data cannot reproduce the historical 2026-08-12 LSE snapshot",
            "valuation_adapter": "NOT_STARTED; GLD is not converted to gold USD/oz or Barrick realized price",
        },
        "figures": {
            "correlation_series_rendered": correlation_lines,
            "publication_note": "derived figures only; no row-level LSE dataset is distributed",
        },
        "artifacts": [
            {
                "path": path.relative_to(project_root).as_posix(),
                "bytes": path.stat().st_size,
                "sha256": sha256(path),
            }
            for path in artifacts
        ],
    }
    _write_json(manifest_path, manifest)
    return manifest
