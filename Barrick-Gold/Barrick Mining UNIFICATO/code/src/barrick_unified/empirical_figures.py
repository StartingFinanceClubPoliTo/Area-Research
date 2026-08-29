"""Generate the Chapter 7 empirical figures from one current LSE snapshot."""

from __future__ import annotations

import hashlib
import json
from pathlib import Path
from typing import Any

import matplotlib

matplotlib.use("Agg")
import matplotlib.pyplot as plt
import numpy as np
import pandas as pd
from scipy.optimize import minimize
from scipy.stats import norm, probplot


FIGURE_NAMES = (
    "correlation_heatmap.png",
    "rolling_corr_gold_ust10y.png",
    "rolling_corr_gold_sp500.png",
    "rolling_corr_gold_dxy.png",
    "rolling_corr_silver_ust10y.png",
    "rolling_corr_silver_sp500.png",
    "rolling_corr_silver_dxy.png",
    "hist_normal_gold.png",
    "hist_normal_silver.png",
    "qqplot_normal_gold.png",
    "qqplot_normal_silver.png",
    "returns_rollingvol_gold.png",
    "returns_rollingvol_silver.png",
    "acf_grid_gold_silver.png",
    "garch_condvol_gold.png",
    "garch_condvol_silver.png",
    "acf_abs_returns_gold_silver.png",
    "arfima_fd_gold.png",
    "arfima_fd_silver.png",
)

LABELS = {
    "gold": "Gold ETF (GLD)",
    "silver": "Silver ETF (SLV)",
    "sp500": "S&P 500 ETF proxy (SPY)",
    "dxy": "DXY formula proxy from LSE FX",
    "ust10y_change": "10Y Treasury yield change",
}

COLORS = {
    "gold": "#C58B18",
    "silver": "#5B6770",
    "sp500": "#1F77B4",
    "dxy": "#2E8B57",
    "yield": "#7A5195",
}


def sha256(path: Path) -> str:
    digest = hashlib.sha256()
    with path.open("rb") as handle:
        for chunk in iter(lambda: handle.read(1024 * 1024), b""):
            digest.update(chunk)
    return digest.hexdigest().upper()


def _number(value: Any) -> float:
    if isinstance(value, str):
        value = value.replace(" ", "").replace(",", ".")
    return float(value)


def _series(rows: list[dict[str, Any]], name: str) -> pd.Series:
    records: list[tuple[pd.Timestamp, float]] = []
    for row in rows:
        timestamp = row.get("timestamp") or row.get("ts") or row.get("date")
        close = row.get("close")
        if timestamp in (None, "") or close in (None, ""):
            continue
        records.append((pd.to_datetime(timestamp, utc=True).normalize(), _number(close)))
    if not records:
        raise ValueError(f"No usable rows for {name}")
    frame = pd.DataFrame(records, columns=["date", name]).drop_duplicates("date", keep="last")
    return frame.set_index("date")[name].sort_index()


def load_current_panel(raw_path: Path) -> tuple[pd.DataFrame, pd.DataFrame]:
    payload = json.loads(raw_path.read_text(encoding="utf-8"))
    rows = payload["rows"]
    market = rows["market_candles"]
    fx = rows["fx_candles"]
    yields = rows["us10y_yields"]

    parts = [
        _series(market["GLD"], "gold"),
        _series(market["SLV"], "silver"),
        _series(market["SPY"], "sp500"),
        _series(fx["EUR/USD"], "eurusd"),
        _series(fx["USD/JPY"], "usdjpy"),
        _series(fx["GBP/USD"], "gbpusd"),
        _series(fx["USD/CAD"], "usdcad"),
        _series(fx["USD/SEK"], "usdsek"),
        _series(fx["USD/CHF"], "usdchf"),
        _series(yields, "ust10y_yield"),
    ]
    panel = pd.concat(parts, axis=1, join="inner").replace([np.inf, -np.inf], np.nan).dropna()
    if len(panel) < 300:
        raise ValueError(f"Aligned LSE panel is too short: {len(panel)} observations")

    log_dxy = (
        -0.576 * np.log(panel["eurusd"])
        + 0.136 * np.log(panel["usdjpy"])
        - 0.119 * np.log(panel["gbpusd"])
        + 0.091 * np.log(panel["usdcad"])
        + 0.042 * np.log(panel["usdsek"])
        + 0.036 * np.log(panel["usdchf"])
    )
    panel["dxy"] = 100.0 * np.exp(log_dxy - log_dxy.iloc[0])
    levels = panel[["gold", "silver", "sp500", "dxy", "ust10y_yield"]].copy()
    returns = np.log(levels[["gold", "silver", "sp500", "dxy"]]).diff()
    returns["ust10y_change"] = levels["ust10y_yield"].diff()
    returns = returns.dropna()
    return levels.loc[returns.index], returns


def _save(fig: plt.Figure, path: Path) -> None:
    path.parent.mkdir(parents=True, exist_ok=True)
    fig.tight_layout()
    fig.savefig(path, dpi=240, bbox_inches="tight", facecolor="white")
    plt.close(fig)


def _acf(values: pd.Series | np.ndarray, nlags: int) -> np.ndarray:
    x = np.asarray(values, dtype=float)
    x = x[np.isfinite(x)]
    x = x - x.mean()
    denominator = float(np.dot(x, x))
    if denominator <= 0:
        return np.full(nlags + 1, np.nan)
    result = [1.0]
    result.extend(float(np.dot(x[:-lag], x[lag:]) / denominator) for lag in range(1, nlags + 1))
    return np.asarray(result)


def _plot_correlation(returns: pd.DataFrame, output: Path) -> None:
    columns = ["gold", "silver", "sp500", "dxy", "ust10y_change"]
    corr = returns[columns].corr().to_numpy()
    labels = [LABELS[col] for col in columns]
    fig, ax = plt.subplots(figsize=(9.2, 7.2))
    image = ax.imshow(corr, cmap="coolwarm", vmin=-1, vmax=1)
    ax.set_xticks(range(len(labels)), labels, rotation=34, ha="right")
    ax.set_yticks(range(len(labels)), labels)
    for i in range(len(labels)):
        for j in range(len(labels)):
            ax.text(j, i, f"{corr[i, j]:.2f}", ha="center", va="center", fontsize=9)
    fig.colorbar(image, ax=ax, label="Pearson correlation")
    ax.set_title("Current LSE-aligned daily-return correlation matrix")
    _save(fig, output)


def _plot_rolling_corr(returns: pd.DataFrame, left: str, right: str, output: Path, window: int) -> None:
    series = returns[left].rolling(window).corr(returns[right])
    fig, ax = plt.subplots(figsize=(10.5, 4.2))
    ax.plot(series.index, series, color=COLORS.get(left, "#1F77B4"), linewidth=1.25)
    ax.axhline(0.0, color="black", linestyle="--", linewidth=0.8)
    ax.set(
        title=f"Rolling {window}-observation correlation: {LABELS[left]} vs {LABELS[right]}",
        xlabel="Date",
        ylabel="Correlation",
        ylim=(-1.02, 1.02),
    )
    ax.grid(True, alpha=0.22)
    _save(fig, output)


def _plot_histogram(series: pd.Series, asset: str, output: Path) -> None:
    values = series.dropna().to_numpy(dtype=float) * 100.0
    mean = float(values.mean())
    std = float(values.std(ddof=1))
    grid = np.linspace(float(values.min()), float(values.max()), 400)
    fig, ax = plt.subplots(figsize=(8.8, 5.0))
    ax.hist(values, bins=55, density=True, alpha=0.62, color=COLORS[asset], edgecolor="white")
    ax.plot(grid, norm.pdf(grid, mean, std), color="#B22222", linewidth=2.0, label="Matched normal density")
    ax.set(title=f"{LABELS[asset]} daily log returns", xlabel="Return (%)", ylabel="Density")
    ax.legend()
    ax.grid(True, alpha=0.18)
    _save(fig, output)


def _plot_qq(series: pd.Series, asset: str, output: Path) -> None:
    values = series.dropna().to_numpy(dtype=float) * 100.0
    (theoretical, ordered), (slope, intercept, _) = probplot(values, dist="norm")
    fig, ax = plt.subplots(figsize=(6.5, 5.6))
    ax.scatter(theoretical, ordered, s=16, alpha=0.60, color=COLORS[asset])
    ax.plot(theoretical, slope * theoretical + intercept, color="#B22222", linewidth=1.8)
    ax.set(title=f"Normal Q-Q plot: {LABELS[asset]}", xlabel="Theoretical quantile", ylabel="Observed return (%)")
    ax.grid(True, alpha=0.22)
    _save(fig, output)


def _plot_returns_vol(series: pd.Series, asset: str, output: Path, window: int) -> None:
    pct = series * 100.0
    rolling = pct.rolling(window).std() * np.sqrt(252.0)
    fig, axes = plt.subplots(2, 1, figsize=(11.2, 6.4), sharex=True)
    axes[0].plot(pct.index, pct, color=COLORS[asset], linewidth=0.75)
    axes[0].set(title=f"{LABELS[asset]} daily log returns", ylabel="Return (%)")
    axes[1].plot(rolling.index, rolling, color="#7A5195", linewidth=1.2)
    axes[1].set(title=f"Rolling {window}-observation annualised volatility", xlabel="Date", ylabel="Volatility (%)")
    for axis in axes:
        axis.grid(True, alpha=0.20)
    _save(fig, output)


def _plot_acf_grid(returns: pd.DataFrame, output: Path, nlags: int = 40) -> None:
    fig, axes = plt.subplots(2, 2, figsize=(11.5, 7.6), sharex=True)
    for column, asset in enumerate(("gold", "silver")):
        for row, transform in enumerate((lambda x: x, lambda x: x**2)):
            values = _acf(transform(returns[asset]), nlags)
            axes[row, column].bar(np.arange(nlags + 1), values, color=COLORS[asset], width=0.75)
            bound = 1.96 / np.sqrt(len(returns))
            axes[row, column].axhline(bound, color="#B22222", linestyle="--", linewidth=0.8)
            axes[row, column].axhline(-bound, color="#B22222", linestyle="--", linewidth=0.8)
            axes[row, column].set_title(f"{LABELS[asset]}: {'return' if row == 0 else 'squared-return'} ACF")
            axes[row, column].grid(True, alpha=0.16)
    axes[1, 0].set_xlabel("Lag")
    axes[1, 1].set_xlabel("Lag")
    _save(fig, output)


def _garch_filter(values: np.ndarray, theta: np.ndarray) -> tuple[np.ndarray, tuple[float, float, float, float]]:
    mu = float(theta[0])
    omega = float(np.exp(theta[1]))
    alpha = float(0.30 / (1.0 + np.exp(-theta[2])))
    beta_share = float(1.0 / (1.0 + np.exp(-theta[3])))
    beta = (0.999 - alpha) * beta_share
    residual = values - mu
    variance = np.empty_like(values)
    variance[0] = max(float(np.var(values, ddof=1)), 1e-8)
    for i in range(1, len(values)):
        variance[i] = omega + alpha * residual[i - 1] ** 2 + beta * variance[i - 1]
    return variance, (mu, omega, alpha, beta)


def _fit_garch(series: pd.Series) -> tuple[pd.Series, dict[str, float]]:
    values = series.dropna().to_numpy(dtype=float) * 100.0
    initial_variance = max(float(np.var(values, ddof=1)), 1e-6)
    initial = np.array([float(values.mean()), np.log(initial_variance * 0.02), -1.4, 3.8])

    def objective(theta: np.ndarray) -> float:
        variance, _ = _garch_filter(values, theta)
        residual = values - theta[0]
        if not np.all(np.isfinite(variance)) or np.any(variance <= 0):
            return 1e30
        return float(0.5 * np.sum(np.log(variance) + residual**2 / variance))

    result = minimize(
        objective,
        initial,
        method="L-BFGS-B",
        bounds=[(-2.0, 2.0), (-20.0, 4.0), (-8.0, 8.0), (-8.0, 8.0)],
        options={"maxiter": 3000, "ftol": 1e-12},
    )
    variance, params = _garch_filter(values, result.x)
    mu, omega, alpha, beta = params
    metadata = {
        "mu": mu,
        "omega": omega,
        "alpha": alpha,
        "beta": beta,
        "alpha_plus_beta": alpha + beta,
        "success": bool(result.success),
        "objective": float(result.fun),
    }
    return pd.Series(np.sqrt(variance), index=series.dropna().index), metadata


def _plot_garch(series: pd.Series, asset: str, output: Path) -> dict[str, float]:
    volatility, metadata = _fit_garch(series)
    returns = series.loc[volatility.index] * 100.0
    tail = min(1000, len(returns))
    fig, axes = plt.subplots(2, 1, figsize=(11.5, 6.2), sharex=True)
    axes[0].plot(returns.index[-tail:], returns.iloc[-tail:], color=COLORS[asset], linewidth=0.75)
    axes[0].set(title=f"{LABELS[asset]} daily returns", ylabel="Return (%)")
    axes[1].plot(volatility.index[-tail:], volatility.iloc[-tail:], color="#7A5195", linewidth=1.15)
    axes[1].set(title="Gaussian GARCH(1,1) conditional volatility", xlabel="Date", ylabel="Volatility (%)")
    for axis in axes:
        axis.grid(True, alpha=0.20)
    _save(fig, output)
    return metadata


def _plot_abs_acf(returns: pd.DataFrame, output: Path, nlags: int = 60) -> None:
    fig, axes = plt.subplots(1, 2, figsize=(11.5, 4.6), sharey=True)
    for axis, asset in zip(axes, ("gold", "silver")):
        values = _acf(np.abs(returns[asset]), nlags)
        axis.bar(np.arange(nlags + 1), values, color=COLORS[asset], width=0.75)
        bound = 1.96 / np.sqrt(len(returns))
        axis.axhline(bound, color="#B22222", linestyle="--", linewidth=0.8)
        axis.axhline(-bound, color="#B22222", linestyle="--", linewidth=0.8)
        axis.set(title=f"Absolute-return ACF: {LABELS[asset]}", xlabel="Lag")
        axis.grid(True, alpha=0.16)
    axes[0].set_ylabel("Autocorrelation")
    _save(fig, output)


def _gph_d(values: pd.Series) -> float:
    x = np.asarray(values.dropna(), dtype=float)
    x = x - x.mean()
    n = len(x)
    m = max(12, int(np.sqrt(n)))
    spectrum = np.abs(np.fft.fft(x)) ** 2 / (2.0 * np.pi * n)
    frequencies = 2.0 * np.pi * np.arange(1, m + 1) / n
    regressor = -2.0 * np.log(2.0 * np.sin(frequencies / 2.0))
    response = np.log(np.maximum(spectrum[1 : m + 1], 1e-18))
    estimate = float(np.polyfit(regressor, response, 1)[0])
    return float(np.clip(estimate, 0.0, 0.49))


def _fractional_difference(series: pd.Series, d: float, threshold: float = 1e-5) -> pd.Series:
    weights = [1.0]
    for k in range(1, min(600, len(series))):
        weight = -weights[-1] * (d - k + 1.0) / k
        weights.append(weight)
        if abs(weight) < threshold and k > 30:
            break
    array = series.to_numpy(dtype=float)
    output = np.full(len(array), np.nan)
    width = len(weights)
    reversed_weights = np.asarray(weights[::-1])
    for index in range(width - 1, len(array)):
        output[index] = float(np.dot(reversed_weights, array[index - width + 1 : index + 1]))
    return pd.Series(output, index=series.index)


def _plot_fractional_difference(series: pd.Series, asset: str, output: Path) -> float:
    absolute = np.abs(series.dropna())
    d = _gph_d(absolute)
    transformed = _fractional_difference(absolute, d)
    tail = min(1000, len(absolute))
    fig, axes = plt.subplots(2, 1, figsize=(11.4, 6.3), sharex=True)
    axes[0].plot(absolute.index[-tail:], absolute.iloc[-tail:] * 100.0, color=COLORS[asset], linewidth=0.8)
    axes[0].set(title=f"Absolute returns: {LABELS[asset]}", ylabel="Absolute return (%)")
    axes[1].plot(transformed.index[-tail:], transformed.iloc[-tail:] * 100.0, color="#E07B39", linewidth=0.8)
    axes[1].set(title=f"Fractionally differenced absolute returns (GPH d={d:.3f})", xlabel="Date", ylabel="Filtered value")
    for axis in axes:
        axis.grid(True, alpha=0.18)
    _save(fig, output)
    return d


def generate_empirical_figures(raw_path: Path, output_dir: Path, rolling_window: int = 252) -> dict[str, Any]:
    output_dir.mkdir(parents=True, exist_ok=False)
    levels, returns = load_current_panel(raw_path)
    _plot_correlation(returns, output_dir / "correlation_heatmap.png")
    for asset in ("gold", "silver"):
        _plot_rolling_corr(returns, asset, "ust10y_change", output_dir / f"rolling_corr_{asset}_ust10y.png", rolling_window)
        _plot_rolling_corr(returns, asset, "sp500", output_dir / f"rolling_corr_{asset}_sp500.png", rolling_window)
        _plot_rolling_corr(returns, asset, "dxy", output_dir / f"rolling_corr_{asset}_dxy.png", rolling_window)
        _plot_histogram(returns[asset], asset, output_dir / f"hist_normal_{asset}.png")
        _plot_qq(returns[asset], asset, output_dir / f"qqplot_normal_{asset}.png")
        _plot_returns_vol(returns[asset], asset, output_dir / f"returns_rollingvol_{asset}.png", rolling_window)
    _plot_acf_grid(returns, output_dir / "acf_grid_gold_silver.png")
    garch = {
        asset: _plot_garch(returns[asset], asset, output_dir / f"garch_condvol_{asset}.png")
        for asset in ("gold", "silver")
    }
    _plot_abs_acf(returns, output_dir / "acf_abs_returns_gold_silver.png")
    gph = {
        asset: _plot_fractional_difference(returns[asset], asset, output_dir / f"arfima_fd_{asset}.png")
        for asset in ("gold", "silver")
    }

    missing = sorted(set(FIGURE_NAMES) - {path.name for path in output_dir.glob("*.png")})
    if missing:
        raise AssertionError(f"Missing generated figures: {missing}")
    figures = [
        {"name": name, "bytes": (output_dir / name).stat().st_size, "sha256": sha256(output_dir / name)}
        for name in FIGURE_NAMES
    ]
    manifest = {
        "schema_version": "1.0",
        "status": "CURRENT_LSE_EMPIRICAL_FIGURES",
        "input_path": raw_path.as_posix(),
        "input_sha256": sha256(raw_path),
        "sample": {
            "start": returns.index.min().date().isoformat(),
            "end": returns.index.max().date().isoformat(),
            "observations": int(len(returns)),
            "rolling_window": int(rolling_window),
        },
        "asset_semantics": {
            "gold": "GLD ETF close, USD/share",
            "silver": "SLV ETF close, USD/share",
            "sp500": "SPY ETF close, USD/share, used as S&P 500 market proxy",
            "dxy": "transparent six-currency DXY formula proxy normalised to 100",
            "ust10y": "LSE US10Y close/yield level; first differences used in returns panel",
        },
        "garch": garch,
        "gph_d": gph,
        "figures": figures,
        "figure_count": len(figures),
        "raw_data_committed": False,
        "not_investment_advice": True,
    }
    (output_dir / "figure_manifest.json").write_text(json.dumps(manifest, indent=2), encoding="utf-8")
    return manifest
