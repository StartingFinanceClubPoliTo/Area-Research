"""Generate simulated three-stage DCF results and article figures.

The model is deliberately synthetic. It is designed to show how a
three-stage DCF works under uncertainty; it is not a valuation of
Barrick Gold and it does not compute a share price.
"""

from __future__ import annotations

import csv
from dataclasses import dataclass, replace
from pathlib import Path

import matplotlib.pyplot as plt
import numpy as np


ROOT = Path(__file__).resolve().parents[1]
IMG_DIR = ROOT / "img"
OUT_DIR = ROOT / "output"


COLORS = {
    "navy": "#1F3A5F",
    "teal": "#1B8A7A",
    "gold": "#D29E2A",
    "red": "#B84A4A",
    "gray": "#6F7782",
    "light": "#DDE7F0",
}


@dataclass(frozen=True)
class ModelParams:
    """Synthetic assumptions for the three-stage DCF example."""

    n_sim: int = 20_000
    seed: int = 42

    high_years: int = 3
    transition_years: int = 4

    revenue_0: float = 1_000.0
    high_growth: float = 0.085
    stable_growth: float = 0.025
    revenue_volatility: float = 0.115

    margin_start: float = 0.205
    margin_stable: float = 0.170
    margin_volatility: float = 0.015

    roic_high: float = 0.145
    roic_stable: float = 0.105
    tax_rate: float = 0.25

    wacc_0: float = 0.091
    wacc_long_run: float = 0.083
    wacc_reversion: float = 0.45
    wacc_volatility: float = 0.009
    corr_growth_wacc: float = -0.30


@dataclass
class SimulationResult:
    params: ModelParams
    revenue: np.ndarray
    margin: np.ndarray
    wacc: np.ndarray
    fcff: np.ndarray
    ev: np.ndarray
    pv_fcff: np.ndarray
    pv_terminal: np.ndarray


def year_count(p: ModelParams) -> int:
    return p.high_years + p.transition_years


def growth_path(p: ModelParams) -> np.ndarray:
    """Expected growth path from high growth to stable growth."""
    t = np.arange(1, year_count(p) + 1)
    transition = (t - 1) / max(year_count(p) - 1, 1)
    return p.high_growth + (p.stable_growth - p.high_growth) * transition


def roic_path(p: ModelParams) -> np.ndarray:
    t = np.arange(1, year_count(p) + 1)
    transition = (t - 1) / max(year_count(p) - 1, 1)
    return p.roic_high + (p.roic_stable - p.roic_high) * transition


def simulate(p: ModelParams) -> SimulationResult:
    rng = np.random.default_rng(p.seed)
    t_total = year_count(p)

    corr = np.array([[1.0, p.corr_growth_wacc], [p.corr_growth_wacc, 1.0]])
    chol = np.linalg.cholesky(corr)
    shocks = rng.standard_normal((p.n_sim, t_total, 2)) @ chol.T
    margin_shocks = rng.standard_normal((p.n_sim, t_total))

    expected_growth = growth_path(p)
    log_growth = (
        np.log1p(expected_growth)[None, :]
        - 0.5 * p.revenue_volatility**2
        + p.revenue_volatility * shocks[:, :, 0]
    )
    revenue = p.revenue_0 * np.exp(np.cumsum(log_growth, axis=1))

    margin_trend = p.margin_start + (
        p.margin_stable - p.margin_start
    ) * (np.arange(1, t_total + 1) / t_total)
    margin = margin_trend[None, :] + p.margin_volatility * margin_shocks
    margin = np.clip(margin, 0.06, 0.35)

    wacc = np.empty((p.n_sim, t_total + 1))
    wacc[:, 0] = p.wacc_0
    e_kdt = np.exp(-p.wacc_reversion)
    conditional_std = p.wacc_volatility * np.sqrt(
        (1.0 - np.exp(-2.0 * p.wacc_reversion)) / (2.0 * p.wacc_reversion)
    )
    for i in range(t_total):
        wacc[:, i + 1] = (
            p.wacc_long_run
            + (wacc[:, i] - p.wacc_long_run) * e_kdt
            + conditional_std * shocks[:, i, 1]
        )
    wacc = np.clip(wacc[:, 1:], 0.035, 0.18)

    nopat = revenue * margin * (1.0 - p.tax_rate)
    reinvestment_rate = np.minimum(growth_path(p) / roic_path(p), 0.95)
    fcff = nopat * (1.0 - reinvestment_rate[None, :])

    discount_factors = 1.0 / np.cumprod(1.0 + wacc, axis=1)
    pv_fcff = np.sum(fcff * discount_factors, axis=1)

    nopat_terminal = nopat[:, -1] * (1.0 + p.stable_growth)
    fcff_terminal = nopat_terminal * (1.0 - p.stable_growth / p.roic_stable)
    terminal_spread = np.maximum(wacc[:, -1] - p.stable_growth, 0.01)
    pv_terminal = (fcff_terminal / terminal_spread) * discount_factors[:, -1]
    ev = pv_fcff + pv_terminal

    return SimulationResult(p, revenue, margin, wacc, fcff, ev, pv_fcff, pv_terminal)


def percentile_summary(values: np.ndarray) -> dict[str, float]:
    return {
        "mean": float(np.mean(values)),
        "median": float(np.median(values)),
        "std": float(np.std(values, ddof=0)),
        "p5": float(np.percentile(values, 5)),
        "p25": float(np.percentile(values, 25)),
        "p75": float(np.percentile(values, 75)),
        "p95": float(np.percentile(values, 95)),
        "mc_se": float(np.std(values, ddof=0) / np.sqrt(values.size)),
    }


def rankdata(x: np.ndarray) -> np.ndarray:
    """Average ranks for one-dimensional arrays, implemented without SciPy."""
    order = np.argsort(x, kind="mergesort")
    sorted_x = x[order]
    ranks = np.empty_like(x, dtype=float)
    start = 0
    while start < len(x):
        end = start + 1
        while end < len(x) and sorted_x[end] == sorted_x[start]:
            end += 1
        avg_rank = 0.5 * (start + end - 1) + 1.0
        ranks[order[start:end]] = avg_rank
        start = end
    return ranks


def spearman(x: np.ndarray, y: np.ndarray) -> float:
    rx = rankdata(x)
    ry = rankdata(y)
    rx -= rx.mean()
    ry -= ry.mean()
    denom = np.sqrt(np.sum(rx**2) * np.sum(ry**2))
    return float(np.sum(rx * ry) / denom)


def sensitivity(result: SimulationResult) -> list[dict[str, float | str]]:
    drivers = {
        "Average revenue": result.revenue.mean(axis=1),
        "Terminal revenue": result.revenue[:, -1],
        "Average WACC": result.wacc.mean(axis=1),
        "Terminal WACC": result.wacc[:, -1],
        "Average operating margin": result.margin.mean(axis=1),
        "Terminal value share": result.pv_terminal / result.ev,
    }
    rows = []
    for name, values in drivers.items():
        rows.append({"driver": name, "rho": spearman(values, result.ev)})
    rows.sort(key=lambda row: abs(float(row["rho"])), reverse=True)
    return rows


def write_csv(path: Path, rows: list[dict[str, object]], fieldnames: list[str]) -> None:
    path.parent.mkdir(parents=True, exist_ok=True)
    with path.open("w", newline="", encoding="utf-8") as handle:
        writer = csv.DictWriter(handle, fieldnames=fieldnames)
        writer.writeheader()
        writer.writerows(rows)


def plot_base(result: SimulationResult) -> None:
    IMG_DIR.mkdir(parents=True, exist_ok=True)
    years = np.arange(0, year_count(result.params) + 1)
    revenue_paths = np.column_stack(
        [np.full(result.params.n_sim, result.params.revenue_0), result.revenue]
    )
    revenue_pct = np.percentile(revenue_paths, [5, 25, 50, 75, 95], axis=0)
    wacc_pct = np.percentile(result.wacc * 100.0, [5, 25, 50, 75, 95], axis=0)

    fig, axes = plt.subplots(2, 2, figsize=(12.5, 8.5))
    fig.suptitle("Three-Stage DCF with Simulated Data", fontsize=14, fontweight="bold")

    ev = result.ev
    axes[0, 0].hist(ev, bins=90, color=COLORS["light"], edgecolor="white", density=True)
    for q, color, label in [
        (5, COLORS["red"], "P5"),
        (50, COLORS["gold"], "Median"),
        (95, COLORS["teal"], "P95"),
    ]:
        axes[0, 0].axvline(np.percentile(ev, q), color=color, lw=1.8, label=label)
    axes[0, 0].set_title("Enterprise value distribution")
    axes[0, 0].set_xlabel("Simulated value units, millions")
    axes[0, 0].set_ylabel("Density")
    axes[0, 0].legend()

    axes[0, 1].fill_between(years, revenue_pct[0], revenue_pct[4], color=COLORS["teal"], alpha=0.15, label="P5-P95")
    axes[0, 1].fill_between(years, revenue_pct[1], revenue_pct[3], color=COLORS["teal"], alpha=0.28, label="P25-P75")
    axes[0, 1].plot(years, revenue_pct[2], color=COLORS["teal"], lw=2.0, label="Median")
    axes[0, 1].set_title("Revenue-driver fan chart")
    axes[0, 1].set_xlabel("Projection year")
    axes[0, 1].set_ylabel("Simulated revenue")
    axes[0, 1].legend()

    axes[1, 0].fill_between(years[1:], wacc_pct[0], wacc_pct[4], color=COLORS["navy"], alpha=0.15, label="P5-P95")
    axes[1, 0].fill_between(years[1:], wacc_pct[1], wacc_pct[3], color=COLORS["navy"], alpha=0.28, label="P25-P75")
    axes[1, 0].plot(years[1:], wacc_pct[2], color=COLORS["navy"], lw=2.0, label="Median")
    axes[1, 0].axhline(result.params.wacc_long_run * 100.0, color=COLORS["gray"], lw=1.0, ls=":", label="Long-run WACC")
    axes[1, 0].set_title("Discount-rate fan chart")
    axes[1, 0].set_xlabel("Projection year")
    axes[1, 0].set_ylabel("WACC (%)")
    axes[1, 0].legend()

    sens = sensitivity(result)
    names = [row["driver"] for row in sens]
    values = [float(row["rho"]) for row in sens]
    colors = [COLORS["teal"] if value > 0 else COLORS["red"] for value in values]
    y = np.arange(len(names))
    axes[1, 1].barh(y, values, color=colors)
    axes[1, 1].set_yticks(y, names)
    axes[1, 1].invert_yaxis()
    axes[1, 1].axvline(0, color="black", lw=0.8)
    axes[1, 1].set_title("Rank-correlation sensitivity")
    axes[1, 1].set_xlabel("Spearman rho")

    fig.tight_layout()
    fig.savefig(IMG_DIR / "mc_dcf_results.png", dpi=180, bbox_inches="tight")
    plt.close(fig)


def scenario_params() -> dict[str, ModelParams]:
    base = ModelParams()
    return {
        "Conservative": replace(
            base,
            seed=100,
            high_growth=0.055,
            stable_growth=0.018,
            revenue_volatility=0.14,
            margin_start=0.180,
            margin_stable=0.145,
            wacc_0=0.105,
            wacc_long_run=0.095,
            wacc_volatility=0.012,
            roic_high=0.120,
            roic_stable=0.090,
        ),
        "Base": base,
        "Expansion": replace(
            base,
            seed=300,
            high_growth=0.110,
            stable_growth=0.030,
            revenue_volatility=0.10,
            margin_start=0.225,
            margin_stable=0.185,
            wacc_0=0.082,
            wacc_long_run=0.076,
            wacc_volatility=0.007,
            roic_high=0.165,
            roic_stable=0.120,
        ),
    }


def plot_scenarios(results: dict[str, SimulationResult]) -> None:
    fig, axes = plt.subplots(1, 2, figsize=(12.5, 4.8))
    palette = {
        "Conservative": COLORS["red"],
        "Base": COLORS["navy"],
        "Expansion": COLORS["teal"],
    }

    for name, result in results.items():
        axes[0].hist(
            result.ev,
            bins=80,
            density=True,
            histtype="step",
            lw=2.0,
            color=palette[name],
            label=name,
        )
        axes[0].axvline(np.median(result.ev), color=palette[name], lw=1.2, ls="--")
    axes[0].set_title("Scenario distributions")
    axes[0].set_xlabel("Simulated value units, millions")
    axes[0].set_ylabel("Density")
    axes[0].legend()

    x = np.arange(len(results))
    names = list(results.keys())
    p5 = [np.percentile(results[name].ev, 5) for name in names]
    med = [np.percentile(results[name].ev, 50) for name in names]
    p95 = [np.percentile(results[name].ev, 95) for name in names]
    axes[1].vlines(x, p5, p95, color=[palette[name] for name in names], lw=5, alpha=0.35)
    axes[1].scatter(x, med, color=[palette[name] for name in names], s=85, zorder=3)
    axes[1].set_xticks(x, names)
    axes[1].set_title("P5, median, and P95 by scenario")
    axes[1].set_ylabel("Simulated value units, millions")
    axes[1].grid(axis="y", alpha=0.25)

    fig.tight_layout()
    fig.savefig(IMG_DIR / "mc_scenarios.png", dpi=180, bbox_inches="tight")
    plt.close(fig)


def main() -> None:
    IMG_DIR.mkdir(parents=True, exist_ok=True)
    OUT_DIR.mkdir(parents=True, exist_ok=True)

    base = simulate(ModelParams())
    plot_base(base)

    base_summary = percentile_summary(base.ev)
    base_summary["terminal_value_share"] = float(np.mean(base.pv_terminal / base.ev))
    base_summary["explicit_cash_flow_share"] = float(np.mean(base.pv_fcff / base.ev))
    write_csv(
        OUT_DIR / "base_summary.csv",
        [{"metric": key, "value": f"{value:.6f}"} for key, value in base_summary.items()],
        ["metric", "value"],
    )

    sens_rows = sensitivity(base)
    write_csv(
        OUT_DIR / "sensitivity.csv",
        [{"driver": row["driver"], "spearman_rho": f"{float(row['rho']):.6f}"} for row in sens_rows],
        ["driver", "spearman_rho"],
    )

    scenarios = {name: simulate(params) for name, params in scenario_params().items()}
    plot_scenarios(scenarios)

    scenario_rows = []
    for name, result in scenarios.items():
        summary = percentile_summary(result.ev)
        scenario_rows.append(
            {
                "scenario": name,
                "mean": f"{summary['mean']:.6f}",
                "median": f"{summary['median']:.6f}",
                "std": f"{summary['std']:.6f}",
                "p5": f"{summary['p5']:.6f}",
                "p95": f"{summary['p95']:.6f}",
                "terminal_value_share": f"{float(np.mean(result.pv_terminal / result.ev)):.6f}",
            }
        )
    write_csv(
        OUT_DIR / "scenario_summary.csv",
        scenario_rows,
        ["scenario", "mean", "median", "std", "p5", "p95", "terminal_value_share"],
    )

    print("Generated simulated three-stage DCF outputs:")
    print(f"- {IMG_DIR / 'mc_dcf_results.png'}")
    print(f"- {IMG_DIR / 'mc_scenarios.png'}")
    print(f"- {OUT_DIR / 'base_summary.csv'}")
    print(f"- {OUT_DIR / 'scenario_summary.csv'}")
    print(f"- {OUT_DIR / 'sensitivity.csv'}")


if __name__ == "__main__":
    main()
