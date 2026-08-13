"""Rolling one-step-ahead GLD option-surface validation.

Each model is recalibrated on date t, warm-started only from its own parameters
at t-1, and forecasts a fixed normalized IV grid for the next available trading
date.  The four chronological model chains are independent and can therefore
run in parallel without changing their sequential results.
"""

import hashlib
import json
from pathlib import Path
from time import perf_counter

import numpy as np
import pandas as pd
from scipy.optimize import minimize

from Bates import Bates
from BnS import BnS
from Hawkes import ExactHawkesCalibration
from calibration_core import OptionSurface
from calibration_workflow import price_surface
from historical_validation import latest_curve_without_lookahead
from lse_dataset import interpolate_zero_rates
from tools.rebuild_exact_hawkes_outputs import exact_prices
from tools.rebuild_lse_benchmarks import (
    HESTON_BOUNDS,
    black_scholes_fit,
    calibrate_heston,
    heston_objective,
    heston_prices,
)


MODEL_NAMES = (
    "Black-Scholes",
    "Heston",
    "Bates",
    "Full Bates-Hawkes",
)
MODEL_SLUGS = {
    "Black-Scholes": "black_scholes",
    "Heston": "heston",
    "Bates": "bates",
    "Full Bates-Hawkes": "full_bates_hawkes",
}
BH_BOUNDS = (
    (1e-4, 1.0),
    (0.1, 10.0),
    (1e-4, 1.0),
    (0.01, 8.0),
    (-0.99, 0.99),
    (1e-3, 5.0),
    (1e-3, 5.0),
    (0.02, 0.95),
    (0.1, 12.0),
    (-0.5, 0.5),
    (1e-3, 0.6),
)
CHECKPOINT_VERSION = "online-rolling-v2"


def _chain_signature(model, *paths):
    """Bind a resumable chain to its exact inputs and workflow version."""
    digest = hashlib.sha256()
    digest.update(CHECKPOINT_VERSION.encode("utf-8"))
    digest.update(model.encode("utf-8"))
    for path in paths:
        with Path(path).open("rb") as handle:
            for block in iter(lambda: handle.read(1024 * 1024), b""):
                digest.update(block)
    return digest.hexdigest()


def _feller_constraint(values):
    return 2.0 * values[1] * values[2] - values[3] ** 2


def _initial_calibration(model, surface, spot, seed):
    if model == "Black-Scholes":
        sigma, _ = black_scholes_fit(surface, spot)
        return np.asarray([sigma], dtype=float), True, "bounded scalar calibration"
    if model == "Heston":
        values, report = calibrate_heston(surface, spot)
        return values, bool(report["success"]), str(report["message"])
    if model == "Bates":
        report = Bates.calibrate_bates(
            surface,
            spot,
            maxiter=25,
            popsize=6,
            seed=seed,
            pricing="cos",
            cos_N=128,
            disp=False,
            return_report=True,
        )
        return report.x, bool(report.success), str(report.message)
    if model == "Full Bates-Hawkes":
        bates_report = Bates.calibrate_bates(
            surface,
            spot,
            maxiter=25,
            popsize=6,
            seed=seed,
            pricing="cos",
            cos_N=128,
            disp=False,
            return_report=True,
        )
        result = ExactHawkesCalibration.calibrate_heston(
            surface,
            spot,
            bates_seed=bates_report.x,
            maxiter=25,
            popsize=6,
            seed=seed,
            global_cos_N=96,
            local_cos_N=128,
            min_branching=0.02,
        )
        return np.asarray(result.x, dtype=float), bool(result.success), str(result.message)
    raise ValueError(f"Unknown model: {model}")


def _objective(model, values, surface, spot):
    if model == "Black-Scholes":
        sigma = float(values[0])
        errors = np.asarray([
            (BnS.bs_call_price(spot, row.K, row.T, row.rate, sigma) - row.price)
            / max(row.vega, 1e-4)
            for row in surface.itertuples(index=False)
        ])
        return float(np.mean(errors**2))
    if model == "Heston":
        return float(
            heston_objective(values, OptionSurface.from_frame(surface), spot, 128)
        )
    if model == "Bates":
        return float(Bates.bates_objective(values, surface, spot, cos_N=128))
    if model == "Full Bates-Hawkes":
        return float(
            ExactHawkesCalibration.objective_heston(
                values, surface, spot, cos_N=96
            )
        )
    raise ValueError(f"Unknown model: {model}")


def _rolling_calibration(model, surface, spot, previous):
    """Warm-start one daily calibration; retain t-1 only on invalid output."""
    if model == "Black-Scholes":
        values, _, message = _initial_calibration(model, surface, spot, 0)
        return values, True, True, message
    if model == "Heston":
        bounds = HESTON_BOUNDS
        objective = heston_objective
        args = (OptionSurface.from_frame(surface), spot, 128)
        options = {"ftol": 1e-7, "maxiter": 20, "disp": False}
    elif model == "Bates":
        bounds = Bates.BOUNDS
        objective = Bates.bates_objective
        args = (OptionSurface.from_frame(surface), spot, 0.0, "cos", 128)
        options = {"ftol": 1e-6, "maxiter": 20, "disp": False}
    elif model == "Full Bates-Hawkes":
        bounds = BH_BOUNDS
        objective = ExactHawkesCalibration.objective_heston
        args = (OptionSurface.from_frame(surface), spot, 0.0, None, 96)
        options = {"ftol": 1e-6, "maxiter": 12, "disp": False}
    else:
        raise ValueError(f"Unknown model: {model}")
    previous_loss = float(objective(previous, *args))
    result = minimize(
        objective,
        x0=np.asarray(previous, dtype=float),
        args=args,
        method="SLSQP",
        bounds=bounds,
        constraints=({"type": "ineq", "fun": _feller_constraint},),
        options=options,
    )
    candidate = np.asarray(result.x, dtype=float)
    candidate_loss = float(objective(candidate, *args))
    accepted = bool(
        np.all(np.isfinite(candidate))
        and np.isfinite(candidate_loss)
        and candidate_loss <= previous_loss + 1e-12
        and _feller_constraint(candidate) >= -1e-9
    )
    values = candidate if accepted else np.asarray(previous, dtype=float)
    return values, bool(result.success), accepted, str(result.message)


def _forecast_parameters(model, values, days_ahead):
    """Evolve only current states to t+1 under each fitted risk-neutral law."""
    values = np.asarray(values, dtype=float).copy()
    horizon = max(float(days_ahead), 1.0) / 365.25
    if model in {"Heston", "Bates", "Full Bates-Hawkes"}:
        values[0] = values[2] + (values[0] - values[2]) * np.exp(-values[1] * horizon)
    if model == "Full Bates-Hawkes":
        lambda0, lambda_bar, branching, beta = values[5:9]
        stationary_mean = lambda_bar / max(1.0 - branching, 1e-12)
        decay = max(beta * (1.0 - branching), 1e-12)
        values[5] = stationary_mean + (lambda0 - stationary_mean) * np.exp(-decay * horizon)
    return values


def _normalized_grid(nodes, rate_history, origin_date):
    grid = pd.DataFrame(nodes).copy()
    curve, curve_date = latest_curve_without_lookahead(rate_history, origin_date)
    grid["K"] = grid["moneyness"]
    grid["rate"] = interpolate_zero_rates(grid["T"], curve)
    grid["price"] = 0.0
    grid["vega"] = 1.0
    grid["curve_date"] = curve_date.strftime("%Y-%m-%d")
    return grid


def _forecast_iv(model, values, nodes, rate_history, origin_date, target_date):
    grid = _normalized_grid(nodes, rate_history, origin_date)
    days_ahead = (pd.Timestamp(target_date) - pd.Timestamp(origin_date)).days
    forecast_values = _forecast_parameters(model, values, days_ahead)
    if model == "Black-Scholes":
        prices = np.asarray([
            BnS.bs_call_price(1.0, row.K, row.T, row.rate, forecast_values[0])
            for row in grid.itertuples(index=False)
        ])
    elif model == "Heston":
        prices = heston_prices(grid, 1.0, forecast_values, terms=192)
    elif model == "Bates":
        prices = price_surface(
            grid,
            lambda strikes, maturity, rate: Bates.bates_prices_cos(
                1.0, strikes, maturity, *forecast_values, rate, N=192
            ),
        )
    elif model == "Full Bates-Hawkes":
        parameters = ExactHawkesCalibration.unpack_heston_params(forecast_values)
        prices = exact_prices(grid, 1.0, parameters)
    else:
        raise ValueError(f"Unknown model: {model}")
    implied_vol = np.asarray([
        BnS.implied_vol_call(price, 1.0, row.K, row.T, row.rate)
        for price, row in zip(prices, grid.itertuples(index=False))
    ])
    result = grid[["node_id", "T", "moneyness", "curve_date"]].copy()
    result["predicted_iv"] = implied_vol
    result["origin_date"] = pd.Timestamp(origin_date)
    result["target_date"] = pd.Timestamp(target_date)
    result["forecast_horizon_calendar_days"] = int(days_ahead)
    return result


def parameter_names(model):
    if model == "Black-Scholes":
        return ("sigma",)
    if model == "Heston":
        return ("v0", "kappa", "theta", "xi", "rho")
    if model == "Bates":
        return Bates.PARAMETER_NAMES
    if model == "Full Bates-Hawkes":
        return (
            "v0", "kappa", "theta", "xi", "rho", "lambda0", "lambda_bar",
            "branching_ratio", "beta", "mu_J", "sigma_J",
        )
    raise ValueError(f"Unknown model: {model}")


def run_model_chain(
    model,
    calibration_cache,
    grid_cache,
    rate_history_path,
    checkpoint_dir,
    seed=20260813,
):
    """Run one chronological model chain and checkpoint after every origin."""
    started = perf_counter()
    calibration = pd.read_parquet(calibration_cache)
    nodes = pd.read_parquet(grid_cache)
    rate_history = pd.read_csv(rate_history_path)
    calibration["date"] = pd.to_datetime(calibration["date"], utc=True)
    nodes["date"] = pd.to_datetime(nodes["date"], utc=True)
    dates = sorted(calibration["date"].unique())
    if len(dates) < 2:
        raise ValueError("Online validation requires at least two calibration dates.")
    node_definition = nodes[["node_id", "T", "moneyness"]].drop_duplicates("node_id")
    slug = MODEL_SLUGS[model]
    checkpoint_dir = Path(checkpoint_dir)
    checkpoint_dir.mkdir(parents=True, exist_ok=True)
    prediction_path = checkpoint_dir / f"online_{slug}_predictions.csv"
    state_path = checkpoint_dir / f"online_{slug}_states.csv"
    run_signature = _chain_signature(
        model, calibration_cache, grid_cache, rate_history_path
    )

    previous = None
    completed_origins = set()
    predictions = []
    states = []
    if prediction_path.exists() and state_path.exists():
        old_predictions = pd.read_csv(prediction_path)
        old_states = pd.read_csv(state_path)
        expected_first = pd.Timestamp(dates[0]).strftime("%Y-%m-%d")
        signature_matches = (
            "run_signature" in old_states
            and old_states["run_signature"].astype(str).eq(run_signature).all()
        )
        if (
            not old_states.empty
            and signature_matches
            and str(old_states["origin_date"].iloc[0])[:10] == expected_first
        ):
            predictions = old_predictions.to_dict(orient="records")
            states = old_states.to_dict(orient="records")
            completed_origins = set(pd.to_datetime(old_states["origin_date"], utc=True))
            latest = old_states.iloc[-1]
            previous = np.asarray([latest[name] for name in parameter_names(model)], dtype=float)

    for index, origin_date in enumerate(dates[:-1]):
        target_date = dates[index + 1]
        if pd.Timestamp(origin_date) in completed_origins:
            continue
        surface = calibration.loc[calibration["date"].eq(origin_date)].copy()
        spot = float(surface["spot"].iloc[0])
        fit_started = perf_counter()
        if previous is None:
            values, success, message = _initial_calibration(
                model, surface, spot, seed
            )
            accepted = True
            method = "global-plus-local initial calibration"
        else:
            values, success, accepted, message = _rolling_calibration(
                model, surface, spot, previous
            )
            method = "daily warm-start SLSQP"
        loss = _objective(model, values, surface, spot)
        forecast = _forecast_iv(
            model,
            values,
            node_definition,
            rate_history,
            origin_date,
            target_date,
        )
        forecast["model"] = model
        forecast["calibration_objective"] = loss
        forecast["calibration_accepted"] = bool(accepted)
        predictions.extend(forecast.to_dict(orient="records"))
        state = {
            "model": model,
            "origin_date": pd.Timestamp(origin_date).strftime("%Y-%m-%d"),
            "target_date": pd.Timestamp(target_date).strftime("%Y-%m-%d"),
            "spot_at_origin": spot,
            "objective": loss,
            "optimizer_success": bool(success),
            "accepted": bool(accepted),
            "method": method,
            "message": message,
            "elapsed_seconds": float(perf_counter() - fit_started),
            "run_signature": run_signature,
        }
        state.update(dict(zip(parameter_names(model), map(float, values))))
        states.append(state)
        pd.DataFrame(predictions).to_csv(prediction_path, index=False)
        pd.DataFrame(states).to_csv(state_path, index=False)
        previous = values

    return {
        "model": model,
        "prediction_path": str(prediction_path),
        "state_path": str(state_path),
        "origins": len(states),
        "fallbacks": int(sum(not bool(row["accepted"]) for row in states)),
        "elapsed_seconds": float(perf_counter() - started),
    }


def merge_online_chains(grid_cache, chain_results, evaluation_start):
    """Merge independent workers and build leakage-safe benchmark forecasts."""
    grids = pd.read_parquet(grid_cache)
    grids["date"] = pd.to_datetime(grids["date"], utc=True)
    evaluation_start = pd.Timestamp(evaluation_start)
    if evaluation_start.tzinfo is None:
        evaluation_start = evaluation_start.tz_localize("UTC")
    else:
        evaluation_start = evaluation_start.tz_convert("UTC")
    predictions = None
    for result in chain_results:
        frame = pd.read_csv(result["prediction_path"])
        frame["target_date"] = pd.to_datetime(frame["target_date"], utc=True)
        frame["origin_date"] = pd.to_datetime(frame["origin_date"], utc=True)
        column = f"pred_iv_{MODEL_SLUGS[result['model']]}"
        frame = frame[[
            "origin_date", "target_date", "node_id", "predicted_iv"
        ]].rename(columns={"predicted_iv": column})
        keys = ["origin_date", "target_date", "node_id"]
        predictions = frame if predictions is None else predictions.merge(
            frame, on=keys, how="inner", validate="one_to_one"
        )
    actual = grids[["date", "node_id", "implied_vol"]].rename(
        columns={"date": "target_date"}
    )
    origin = grids[["date", "node_id", "implied_vol"]].rename(
        columns={"date": "origin_date", "implied_vol": "random_walk_forecast"}
    )
    merged = predictions.merge(
        actual, on=["target_date", "node_id"], how="left", validate="many_to_one"
    ).merge(
        origin, on=["origin_date", "node_id"], how="left", validate="many_to_one"
    )

    training = grids.loc[grids["date"].lt(evaluation_start)].dropna(
        subset=["implied_vol"]
    )
    totals = training.groupby("node_id")["implied_vol"].sum().to_dict()
    counts = training.groupby("node_id")["implied_vol"].count().to_dict()
    merged["mean_forecast"] = np.nan
    for origin_date in sorted(merged["origin_date"].unique()):
        known = grids.loc[grids["date"].eq(origin_date)].dropna(subset=["implied_vol"])
        for row in known.itertuples(index=False):
            totals[row.node_id] = totals.get(row.node_id, 0.0) + float(row.implied_vol)
            counts[row.node_id] = counts.get(row.node_id, 0) + 1
        mask = merged["origin_date"].eq(origin_date)
        merged.loc[mask, "mean_forecast"] = [
            totals.get(node, np.nan) / counts.get(node, 0)
            if counts.get(node, 0) else np.nan
            for node in merged.loc[mask, "node_id"]
        ]
    return merged.sort_values(["target_date", "node_id"]).reset_index(drop=True)


def parameter_stability(chain_results):
    """Aggregate daily parameter paths without publishing licensed row data."""
    rows = []
    convergence = []
    for result in chain_results:
        states = pd.read_csv(result["state_path"])
        model = result["model"]
        for parameter in parameter_names(model):
            values = pd.to_numeric(states[parameter], errors="coerce")
            rows.append({
                "model": model,
                "parameter": parameter,
                "minimum": float(values.min()),
                "median": float(values.median()),
                "maximum": float(values.max()),
                "last": float(values.iloc[-1]),
            })
        convergence.append({
            "model": model,
            "origins": int(len(states)),
            "optimizer_successes": int(states["optimizer_success"].sum()),
            "accepted_updates": int(states["accepted"].sum()),
            "fallbacks": int((~states["accepted"].astype(bool)).sum()),
            "median_calibration_seconds": float(states["elapsed_seconds"].median()),
            "total_calibration_seconds": float(states["elapsed_seconds"].sum()),
        })
    return pd.DataFrame(rows), pd.DataFrame(convergence)
