"""No-look-ahead helpers for historical GLD option-model validation.

The backtest is deliberately separate from the current point-in-time valuation.
It freezes structural model parameters at a historical cutoff and evaluates
future option-implied-volatility forecasts on dates never used for calibration.
"""

from itertools import permutations

import matplotlib.pyplot as plt
import numpy as np
import pandas as pd
from scipy.interpolate import griddata
from scipy.stats import norm

from BnS import BnS
from Sampling import Sampling
from lse_dataset import interpolate_zero_rates


MODEL_NAMES = (
    "Black-Scholes",
    "Heston",
    "Bates",
    "Full Bates-Hawkes",
)


def weekly_last_dates(dates, start, end):
    """Choose the last available observation in each Friday-ending week."""
    values = pd.Series(pd.to_datetime(pd.Series(dates), utc=True).dropna().unique())
    values = values.sort_values()
    start = pd.Timestamp(start)
    end = pd.Timestamp(end)
    if start.tzinfo is None:
        start = start.tz_localize("UTC")
    else:
        start = start.tz_convert("UTC")
    if end.tzinfo is None:
        end = end.tz_localize("UTC")
    else:
        end = end.tz_convert("UTC")
    values = values.loc[values.between(start, end)]
    if values.empty:
        return []
    periods = values.dt.tz_localize(None).dt.to_period("W-FRI")
    return list(values.groupby(periods).max())


def latest_curve_without_lookahead(rate_history, observation_date):
    """Return the latest complete curve dated no later than an observation."""
    rates = pd.DataFrame(rate_history).copy()
    rates["date"] = pd.to_datetime(rates["date"], errors="coerce", utc=True)
    observation_date = pd.Timestamp(observation_date)
    if observation_date.tzinfo is None:
        observation_date = observation_date.tz_localize("UTC")
    else:
        observation_date = observation_date.tz_convert("UTC")
    eligible = rates.loc[rates["date"].le(observation_date)]
    if eligible.empty:
        raise ValueError(f"No Treasury curve is available by {observation_date.date()}.")
    curve_date = eligible["date"].max()
    curve = eligible.loc[eligible["date"].eq(curve_date)].copy()
    if curve["symbol"].nunique() < 2:
        raise ValueError(f"Treasury curve on {curve_date.date()} has too few tenors.")
    return curve.sort_values("maturity_years").reset_index(drop=True), curve_date


def _spot_on_date(stock_history, observation_date):
    history = pd.DataFrame(stock_history).copy()
    timestamp_column = "timestamp" if "timestamp" in history else "ts"
    history["_date"] = pd.to_datetime(
        history[timestamp_column], errors="coerce", utc=True
    ).dt.normalize()
    target = pd.Timestamp(observation_date)
    if target.tzinfo is None:
        target = target.tz_localize("UTC")
    else:
        target = target.tz_convert("UTC")
    matches = history.loc[history["_date"].eq(target.normalize()), "close"]
    if matches.empty:
        raise ValueError(f"No GLD close is available on {target.date()}.")
    spot = float(pd.to_numeric(matches, errors="coerce").dropna().iloc[-1])
    if not np.isfinite(spot) or spot <= 0.0:
        raise ValueError(f"Invalid GLD close on {target.date()}.")
    return spot


def prepare_historical_eligible_surface(
    option_history,
    stock_history,
    rate_history,
    observation_date,
    min_volume=25,
    moneyness_bounds=(0.85, 1.20),
    maturity_days=(21, 730),
):
    """Build all liquid, synchronized historical call observations.

    Daily option bars contain actual traded closes, not LSE snapshot IVs.  We
    invert those closes with the same Black--Scholes routine used elsewhere,
    and discard no-arbitrage violations caused by nonsynchronous last trades.
    """
    options = pd.DataFrame(option_history).copy()
    required = {"ts", "expiry", "opt_type", "strike", "close", "volume"}
    missing = sorted(required.difference(options.columns))
    if missing:
        raise ValueError(f"Historical option bars are missing columns: {missing}")
    target = pd.Timestamp(observation_date)
    if target.tzinfo is None:
        target = target.tz_localize("UTC")
    else:
        target = target.tz_convert("UTC")
    target = target.normalize()
    options["date"] = pd.to_datetime(options["ts"], errors="coerce", utc=True).dt.normalize()
    options = options.loc[
        options["date"].eq(target)
        & options["opt_type"].astype(str).str.upper().eq("C")
    ].copy()
    if options.empty:
        raise ValueError(f"No GLD call bars are available on {target.date()}.")

    spot = _spot_on_date(stock_history, target)
    curve, curve_date = latest_curve_without_lookahead(rate_history, target)
    options["expiry_dt"] = pd.to_datetime(options["expiry"], errors="coerce", utc=True)
    options["T"] = (options["expiry_dt"] - target).dt.days / 365.25
    options["K"] = pd.to_numeric(options["strike"], errors="coerce")
    options["price"] = pd.to_numeric(options["close"], errors="coerce")
    options["volume"] = pd.to_numeric(options["volume"], errors="coerce")
    options["moneyness"] = options["K"] / spot
    options = options.loc[
        options["T"].between(maturity_days[0] / 365.25, maturity_days[1] / 365.25)
        & options["moneyness"].between(*moneyness_bounds)
        & options["price"].gt(0.10)
        & options["volume"].ge(float(min_volume))
    ].copy()
    if len(options) < 4:
        raise ValueError(
            f"Only {len(options)} liquid historical calls remain on {target.date()}."
        )
    options["rate"] = interpolate_zero_rates(options["T"], curve)
    options["implied_vol"] = [
        BnS.implied_vol_call(row.price, spot, row.K, row.T, row.rate)
        for row in options.itertuples(index=False)
    ]
    options = options.loc[
        options["implied_vol"].between(0.03, 1.50)
    ].copy()
    options["vega"] = [
        BnS.calculate_bs_vega(spot, row.K, row.T, row.rate, 0.0, row.implied_vol)
        for row in options.itertuples(index=False)
    ]
    options = options.loc[options["vega"].ge(0.50)].copy()
    options = options.drop_duplicates(["expiry_dt", "K"], keep="last")
    if len(options) < 4 or options["T"].nunique() < 2:
        raise ValueError(
            f"Only {len(options)} valid-IV calls remain on {target.date()}."
        )
    options["expiry"] = options["expiry_dt"].dt.strftime("%Y-%m-%d")
    options["source_updated_at"] = target.strftime("%Y-%m-%dT00:00:00Z")
    options["source_age_hours"] = 0.0
    options["price_method"] = "LSE historical daily traded close"
    columns = [
        "date", "expiry", "T", "K", "price", "rate", "implied_vol", "vega",
        "moneyness", "volume", "source_updated_at", "source_age_hours", "price_method",
    ]
    eligible = options.loc[:, columns].sort_values(["T", "K"]).reset_index(drop=True)
    eligible["spot"] = spot
    eligible["rate_curve_date"] = curve_date.strftime("%Y-%m-%d")
    return eligible


def prepare_historical_surface(
    option_history,
    stock_history,
    rate_history,
    observation_date,
    sample_points=64,
    min_volume=25,
    moneyness_bounds=(0.85, 1.20),
    maturity_days=(21, 730),
):
    """Select deterministic Chebyshev nodes from a historical call surface."""
    eligible = prepare_historical_eligible_surface(
        option_history,
        stock_history,
        rate_history,
        observation_date,
        min_volume=min_volume,
        moneyness_bounds=moneyness_bounds,
        maturity_days=maturity_days,
    )
    if len(eligible) < sample_points:
        raise ValueError(
            f"Only {len(eligible)} valid historical calls remain on {pd.Timestamp(observation_date).date()}."
        )
    side = max(2, int(round(np.sqrt(sample_points))))
    sample = Sampling.sample_chebyshev(eligible, n_T=side, n_K=side)
    sample = sample.iloc[:sample_points].copy().reset_index(drop=True)
    return sample


def fixed_surface_nodes(
    n_moneyness=6,
    n_maturities=6,
    moneyness_bounds=(0.92, 1.08),
    maturity_days=(60, 365),
):
    """Return an ex-ante normalized grid shared by every forecast origin."""
    moneyness = Sampling.chebyshev_roots(
        int(n_moneyness), float(moneyness_bounds[0]), float(moneyness_bounds[1])
    )
    maturities = Sampling.chebyshev_roots(
        int(n_maturities),
        float(maturity_days[0]) / 365.25,
        float(maturity_days[1]) / 365.25,
    )
    rows = [
        {"node_id": i * len(moneyness) + j, "T": maturity, "moneyness": money}
        for i, maturity in enumerate(maturities)
        for j, money in enumerate(moneyness)
    ]
    return pd.DataFrame(rows)


def interpolate_observed_iv_grid(eligible_surface, nodes):
    """Interpolate observed IV onto a fixed grid without using model outputs."""
    surface = pd.DataFrame(eligible_surface)
    grid = pd.DataFrame(nodes).copy()
    points = surface[["T", "moneyness"]].to_numpy(dtype=float)
    targets = grid[["T", "moneyness"]].to_numpy(dtype=float)
    values = surface["implied_vol"].to_numpy(dtype=float)
    grid["implied_vol"] = griddata(points, values, targets, method="linear")
    grid["date"] = pd.to_datetime(surface["date"].iloc[0], utc=True)
    grid["spot"] = float(surface["spot"].iloc[0])
    grid["rate_curve_date"] = str(surface["rate_curve_date"].iloc[0])
    return grid


def build_weekly_panel(
    option_history,
    stock_history,
    rate_history,
    dates,
    **surface_kwargs,
):
    """Build identically sampled weekly surfaces and fail on missing weeks."""
    frames = [
        prepare_historical_surface(
            option_history,
            stock_history,
            rate_history,
            date,
            **surface_kwargs,
        )
        for date in dates
    ]
    if not frames:
        raise ValueError("At least one weekly surface is required.")
    return pd.concat(frames, ignore_index=True)


def attach_prevailing_mean(test_panel, training_values):
    """Attach the expanding mean forecast available immediately before each week."""
    panel = pd.DataFrame(test_panel).copy()
    training_values = np.asarray(training_values, dtype=float)
    training_values = training_values[np.isfinite(training_values)]
    if training_values.size == 0:
        raise ValueError("The historical-mean benchmark needs training observations.")
    total = float(training_values.sum())
    count = int(training_values.size)
    panel["mean_forecast"] = np.nan
    for date in sorted(panel["date"].unique()):
        mask = panel["date"].eq(date)
        panel.loc[mask, "mean_forecast"] = total / count
        realised = panel.loc[mask, "implied_vol"].to_numpy(dtype=float)
        realised = realised[np.isfinite(realised)]
        total += float(realised.sum())
        count += int(realised.size)
    return panel


def oos_r2_metrics(
    predictions,
    model_columns,
    benchmark_column="mean_forecast",
    benchmark_name="Prevailing mean",
):
    """Compute Campbell--Thompson OOS R2 against one benchmark and peers."""
    frame = pd.DataFrame(predictions).copy()
    required = ["implied_vol", benchmark_column, *model_columns.values()]
    frame = frame.dropna(subset=required)
    if frame.empty:
        raise ValueError("No common finite OOS predictions remain.")
    actual = frame["implied_vol"].to_numpy(dtype=float)
    benchmark_error = actual - frame[benchmark_column].to_numpy(dtype=float)
    benchmark_sse = float(np.sum(benchmark_error**2))
    metrics = []
    errors = {}
    for model, column in model_columns.items():
        error = actual - frame[column].to_numpy(dtype=float)
        errors[model] = error
        sse = float(np.sum(error**2))
        metrics.append({
            "model": model,
            "observations": int(len(frame)),
            "weeks": int(frame["date"].nunique()),
            "mae_bp": float(np.mean(np.abs(error)) * 10_000.0),
            "rmse_bp": float(np.sqrt(np.mean(error**2)) * 10_000.0),
            "benchmark": benchmark_name,
            "r2_oos_vs_benchmark": float(1.0 - sse / benchmark_sse),
            "sse_iv_decimal": sse,
            "benchmark_sse_iv_decimal": benchmark_sse,
        })
        if benchmark_column == "mean_forecast":
            metrics[-1]["r2_oos_vs_prevailing_mean"] = metrics[-1]["r2_oos_vs_benchmark"]
            metrics[-1]["mean_benchmark_sse_iv_decimal"] = benchmark_sse
    pairwise = []
    for model, comparator in permutations(model_columns, 2):
        sse_model = float(np.sum(errors[model] ** 2))
        sse_comparator = float(np.sum(errors[comparator] ** 2))
        pairwise.append({
            "model": model,
            "comparator": comparator,
            "r2_oos_pairwise": float(1.0 - sse_model / sse_comparator),
            "interpretation": "positive means model has lower OOS SSE",
        })
    return frame, pd.DataFrame(metrics), pd.DataFrame(pairwise)


def _newey_west_mean_test(loss_difference, max_lags=4):
    """Test whether a weekly loss differential has non-zero mean with HAC SE."""
    values = np.asarray(loss_difference, dtype=float)
    values = values[np.isfinite(values)]
    n = int(values.size)
    if n < 3:
        return {"weeks": n, "mean_loss_difference": np.nan, "hac_t": np.nan, "hac_pvalue_two_sided": np.nan}
    centered = values - values.mean()
    lag_count = min(int(max_lags), n - 1)
    gamma0 = float(np.dot(centered, centered) / n)
    long_run_variance = gamma0
    for lag in range(1, lag_count + 1):
        covariance = float(np.dot(centered[lag:], centered[:-lag]) / n)
        weight = 1.0 - lag / (lag_count + 1.0)
        long_run_variance += 2.0 * weight * covariance
    standard_error = np.sqrt(max(long_run_variance, 0.0) / n)
    statistic = float(values.mean() / standard_error) if standard_error > 0.0 else np.nan
    pvalue = float(2.0 * norm.sf(abs(statistic))) if np.isfinite(statistic) else np.nan
    return {
        "weeks": n,
        "mean_loss_difference": float(values.mean()),
        "hac_lags": lag_count,
        "hac_standard_error": float(standard_error),
        "hac_t": statistic,
        "hac_pvalue_two_sided": pvalue,
    }


def cumulative_loss_analysis(
    predictions,
    model_columns,
    benchmark_columns=None,
):
    """Create Welch--Goyal cumulative loss differences and HAC comparisons."""
    frame = pd.DataFrame(predictions).copy()
    frame["date"] = pd.to_datetime(frame["date"], utc=True)
    actual = frame["implied_vol"] * 100.0
    if benchmark_columns is None:
        benchmark_columns = {"Prevailing mean": "mean_forecast"}
    losses = pd.DataFrame({"date": frame["date"]})
    for label, column in benchmark_columns.items():
        losses[label] = (actual - frame[column] * 100.0) ** 2
    for model, column in model_columns.items():
        losses[model] = (actual - frame[column] * 100.0) ** 2
    weekly = losses.groupby("date", as_index=False).mean(numeric_only=True)

    cumulative = pd.DataFrame({"date": weekly["date"]})
    tests = []
    benchmarks = [*benchmark_columns.keys(), *model_columns.keys()]
    for model in model_columns:
        for comparator in benchmarks:
            if comparator == model:
                continue
            difference = weekly[comparator] - weekly[model]
            label = f"{model} vs {comparator}"
            cumulative[label] = difference.cumsum()
            test = _newey_west_mean_test(difference)
            test.update({
                "model": model,
                "comparator": comparator,
                "loss_difference": "comparator MSE minus model MSE (IV percentage-points squared)",
                "positive_mean_favors": model,
                "decision_5pct_two_sided": (
                    "different" if test["hac_pvalue_two_sided"] < 0.05 else "not distinguishable"
                ),
            })
            tests.append(test)
    return weekly, cumulative, pd.DataFrame(tests)


def plot_welch_goyal(cumulative, output_path):
    """Plot cumulative squared-error differences against mean and all peers."""
    frame = pd.DataFrame(cumulative).copy()
    frame["date"] = pd.to_datetime(frame["date"], utc=True)
    figure, axes = plt.subplots(2, 2, figsize=(12.5, 8.4), sharex=True)
    groups = [
        [f"{model} vs Prevailing mean" for model in MODEL_NAMES],
        [f"{model} vs Black-Scholes" for model in MODEL_NAMES if model != "Black-Scholes"],
        ["Bates vs Heston", "Full Bates-Hawkes vs Heston"],
        ["Full Bates-Hawkes vs Bates"],
    ]
    titles = [
        "Each model against the prevailing historical mean",
        "Added structure against Black--Scholes",
        "Jump models against Heston",
        "Self-excitation against independent jumps",
    ]
    for axis, labels, title in zip(axes.ravel(), groups, titles):
        for label in labels:
            if label in frame:
                axis.plot(frame["date"], frame[label], linewidth=2.0, label=label.split(" vs ")[0])
        axis.axhline(0.0, color="black", linewidth=0.9, linestyle="--")
        axis.set_title(title)
        axis.set_ylabel("Cumulative loss difference (IV pp$^2$)")
        axis.grid(True, alpha=0.25)
        axis.legend(fontsize=8)
    for axis in axes[-1]:
        axis.set_xlabel("Holdout week")
    figure.autofmt_xdate(rotation=25)
    figure.suptitle(
        "Welch--Goyal OOS diagnostics: an upward line favors the named model",
        y=1.01,
        fontsize=13,
    )
    figure.tight_layout()
    figure.savefig(output_path, dpi=220, bbox_inches="tight")
    plt.close(figure)
