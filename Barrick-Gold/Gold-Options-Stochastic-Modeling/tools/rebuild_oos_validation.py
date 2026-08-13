"""Rebuild the six-month, no-look-ahead historical model comparison."""

import json
from pathlib import Path
import sys

import matplotlib

matplotlib.use("Agg")
import numpy as np
import pandas as pd

ROOT = Path(__file__).resolve().parents[1]
if str(ROOT) not in sys.path:
    sys.path.insert(0, str(ROOT))

from Bates import Bates  # noqa: E402
from BatesHawkesExact import BatesHawkesExact  # noqa: E402
from BnS import BnS  # noqa: E402
from Hawkes import ExactHawkesCalibration  # noqa: E402
from calibration_workflow import option_diagnostics, price_surface  # noqa: E402
from historical_validation import (  # noqa: E402
    MODEL_NAMES,
    attach_prevailing_mean,
    build_weekly_panel,
    cumulative_loss_analysis,
    oos_r2_metrics,
    plot_welch_goyal,
    prepare_historical_surface,
    weekly_last_dates,
)
from tools.rebuild_exact_hawkes_outputs import exact_prices  # noqa: E402
from tools.rebuild_lse_benchmarks import (  # noqa: E402
    black_scholes_fit,
    calibrate_heston,
    error_metrics,
    heston_prices,
)


SEED = 20260813
MODEL_COLUMNS = {
    "Black-Scholes": "pred_iv_black_scholes",
    "Heston": "pred_iv_heston",
    "Bates": "pred_iv_bates",
    "Full Bates-Hawkes": "pred_iv_full_bates_hawkes",
}


def _date_range(option_history, stock_history):
    option_dates = pd.to_datetime(option_history["ts"], utc=True).dt.normalize()
    stock_dates = pd.to_datetime(stock_history["timestamp"], utc=True).dt.normalize()
    common_end = min(option_dates.max(), stock_dates.max())
    cutoff = common_end - pd.DateOffset(months=6)
    training_start = cutoff - pd.DateOffset(months=6)
    return training_start, cutoff, common_end


def _predict_prices_by_date(panel, pricer):
    values = np.full(len(panel), np.nan, dtype=float)
    for _, group in panel.groupby("date", sort=True):
        positions = panel.index.get_indexer(group.index)
        spot = float(group["spot"].iloc[0])
        values[positions] = np.asarray(pricer(group, spot), dtype=float)
    return values


def _prices_to_iv(panel, model_prices):
    return np.asarray([
        BnS.implied_vol_call(price, row.spot, row.K, row.T, row.rate)
        for price, row in zip(model_prices, panel.itertuples(index=False))
    ], dtype=float)


def _calibrate_models(surface, spot):
    print("  calibrating historical Black--Scholes...")
    bs_sigma, bs_diagnostics = black_scholes_fit(surface, spot)

    print("  calibrating historical Heston...")
    heston_params, heston_report = calibrate_heston(surface, spot)
    heston_diagnostics = option_diagnostics(
        surface, spot, heston_prices(surface, spot, heston_params, terms=256)
    )

    print("  calibrating historical Bates...")
    bates_report = Bates.calibrate_bates(
        surface,
        spot,
        maxiter=25,
        popsize=6,
        seed=SEED,
        pricing="cos",
        cos_N=128,
        disp=False,
        return_report=True,
    )
    bates_params = bates_report.x
    bates_diagnostics = option_diagnostics(
        surface,
        spot,
        price_surface(
            surface,
            lambda strikes, maturity, rate: Bates.bates_prices_cos(
                spot, strikes, maturity, *bates_params, rate, N=256
            ),
        ),
    )

    print("  calibrating historical full Bates--Hawkes...")
    hawkes_result = ExactHawkesCalibration.calibrate_heston(
        surface,
        spot,
        bates_seed=bates_params,
        maxiter=25,
        popsize=6,
        seed=SEED,
        global_cos_N=128,
        local_cos_N=192,
        min_branching=0.02,
    )
    hawkes_params = ExactHawkesCalibration.unpack_heston_params(hawkes_result.x)
    hawkes_params["feller_gap"] = (
        2.0 * hawkes_params["kappa"] * hawkes_params["theta"]
        - hawkes_params["xi"] ** 2
    )
    if not hawkes_result.success or hawkes_params["feller_gap"] < -1e-9:
        raise RuntimeError(
            f"Historical Bates--Hawkes calibration failed: {hawkes_result.message}"
        )
    hawkes_diagnostics = option_diagnostics(
        surface, spot, exact_prices(surface, spot, hawkes_params)
    )

    parameters = {
        "Black-Scholes": {
            "parameters": {"sigma": float(bs_sigma)},
            "success": True,
        },
        "Heston": heston_report,
        "Bates": bates_report.as_dict(),
        "Full Bates-Hawkes": {
            "parameters": hawkes_params,
            "objective": float(hawkes_result.fun),
            "success": bool(hawkes_result.success),
            "message": str(hawkes_result.message),
        },
    }
    calibration_diagnostics = {
        "Black-Scholes": bs_diagnostics,
        "Heston": heston_diagnostics,
        "Bates": bates_diagnostics,
        "Full Bates-Hawkes": hawkes_diagnostics,
    }
    calibration_metrics = pd.DataFrame([
        {"model": model, **error_metrics(diagnostics)}
        for model, diagnostics in calibration_diagnostics.items()
    ])
    fitted = {
        "bs_sigma": bs_sigma,
        "heston_params": heston_params,
        "bates_params": bates_params,
        "hawkes_params": hawkes_params,
    }
    return fitted, parameters, calibration_metrics


def _predict_models(panel, fitted):
    predictions = panel.copy().reset_index(drop=True)
    bs_prices = _predict_prices_by_date(
        predictions,
        lambda group, spot: np.asarray([
            BnS.bs_call_price(spot, row.K, row.T, row.rate, fitted["bs_sigma"])
            for row in group.itertuples(index=False)
        ]),
    )
    heston_model_prices = _predict_prices_by_date(
        predictions,
        lambda group, spot: heston_prices(
            group, spot, fitted["heston_params"], terms=256
        ),
    )
    bates_model_prices = _predict_prices_by_date(
        predictions,
        lambda group, spot: price_surface(
            group,
            lambda strikes, maturity, rate: Bates.bates_prices_cos(
                spot, strikes, maturity, *fitted["bates_params"], rate, N=256
            ),
        ),
    )
    hawkes_model_prices = _predict_prices_by_date(
        predictions,
        lambda group, spot: exact_prices(group, spot, fitted["hawkes_params"]),
    )
    for model_prices, column in (
        (bs_prices, MODEL_COLUMNS["Black-Scholes"]),
        (heston_model_prices, MODEL_COLUMNS["Heston"]),
        (bates_model_prices, MODEL_COLUMNS["Bates"]),
        (hawkes_model_prices, MODEL_COLUMNS["Full Bates-Hawkes"]),
    ):
        predictions[column] = _prices_to_iv(predictions, model_prices)
    return predictions


def main():
    local_dir = ROOT / "Data" / "lse_local"
    output_dir = ROOT / "Data"
    option_path = local_dir / "options_GLD_1d.parquet"
    rate_path = local_dir / "usd_treasury_history.csv"
    stock_path = local_dir / "gld_daily_history.csv"
    missing = [path for path in (option_path, rate_path, stock_path) if not path.exists()]
    if missing:
        raise FileNotFoundError(f"Missing local LSE historical inputs: {missing}")

    option_history = pd.read_parquet(option_path)
    rate_history = pd.read_csv(rate_path)
    stock_history = pd.read_csv(stock_path)
    training_start, cutoff, holdout_end = _date_range(option_history, stock_history)
    available_dates = pd.to_datetime(option_history["ts"], utc=True).dt.normalize().unique()
    training_dates = weekly_last_dates(available_dates, training_start, cutoff)
    test_dates = weekly_last_dates(
        available_dates, cutoff + pd.Timedelta(days=1), holdout_end
    )
    if len(training_dates) < 20 or len(test_dates) < 20:
        raise ValueError(
            f"Need at least 20 weekly surfaces per segment; got {len(training_dates)} training and {len(test_dates)} test."
        )
    calibration_date = max(date for date in available_dates if date <= cutoff)
    print(
        f"Historical design: training {training_dates[0].date()}--{calibration_date.date()}, "
        f"holdout {test_dates[0].date()}--{test_dates[-1].date()}."
    )
    print("  building weekly historical-mean training panel...")
    training_panel = build_weekly_panel(
        option_history, stock_history, rate_history, training_dates
    )
    calibration_surface = prepare_historical_surface(
        option_history, stock_history, rate_history, calibration_date
    )
    spot = float(calibration_surface["spot"].iloc[0])
    fitted, parameters, calibration_metrics = _calibrate_models(
        calibration_surface, spot
    )

    print("  building and pricing the untouched six-month holdout...")
    test_panel = build_weekly_panel(
        option_history, stock_history, rate_history, test_dates
    )
    test_panel = attach_prevailing_mean(
        test_panel, training_panel["implied_vol"].to_numpy(dtype=float)
    )
    predictions = _predict_models(test_panel, fitted)
    common, metrics, pairwise = oos_r2_metrics(predictions, MODEL_COLUMNS)
    _, cumulative, loss_tests = cumulative_loss_analysis(common, MODEL_COLUMNS)

    metadata = {
        "purpose": "retrospective model-behaviour diagnostic only",
        "valuation_separation": "current Barrick/GLD valuation continues to use current-date LSE calibration",
        "source": "London Strategic Edge GLD option 1d bars, GLD 1d candles, and US Treasury yields",
        "training_window_start": training_dates[0].strftime("%Y-%m-%d"),
        "calibration_cutoff": pd.Timestamp(calibration_date).strftime("%Y-%m-%d"),
        "holdout_start": test_dates[0].strftime("%Y-%m-%d"),
        "holdout_end": test_dates[-1].strftime("%Y-%m-%d"),
        "training_weeks": len(training_dates),
        "holdout_weeks": len(test_dates),
        "nodes_per_week": 64,
        "parameter_policy": "all model parameters frozen at the cutoff surface; no holdout refit",
        "mean_benchmark": "prevailing mean IV initialized on training weeks and updated only after each realized holdout week",
        "sampling": "liquid calls, volume >= 25, 0.85 <= K/S <= 1.20, 21--730 DTE, deterministic 8x8 Chebyshev nodes",
        "rate_policy": "latest complete LSE Treasury curve dated no later than each option date",
        "oos_target": "historical traded-close implied volatility",
        "r2_definition": "1 - SSE_model / SSE_benchmark on common OOS observations",
        "cumulative_loss_definition": "sum through week t of comparator MSE minus model MSE",
        "systematic_test": "two-sided Newey-West HAC test of the weekly loss-difference mean, four lags",
        "seed": SEED,
    }
    for column, value in (
        ("training_window_start", metadata["training_window_start"]),
        ("calibration_cutoff", metadata["calibration_cutoff"]),
        ("holdout_start", metadata["holdout_start"]),
        ("holdout_end", metadata["holdout_end"]),
    ):
        metrics[column] = value
    calibration_metrics.insert(1, "calibration_cutoff", metadata["calibration_cutoff"])

    metrics.to_csv(output_dir / "oos_validation_metrics.csv", index=False)
    pairwise.to_csv(output_dir / "oos_pairwise_r2.csv", index=False)
    loss_tests.to_csv(output_dir / "oos_loss_differential_tests.csv", index=False)
    cumulative.to_csv(output_dir / "oos_welch_goyal_cumulative.csv", index=False)
    calibration_metrics.to_csv(
        output_dir / "oos_historical_calibration_metrics.csv", index=False
    )
    plot_welch_goyal(cumulative, output_dir / "oos_welch_goyal_cumulative.png")
    (output_dir / "oos_historical_calibration_params.json").write_text(
        json.dumps(parameters, indent=2), encoding="utf-8"
    )
    (output_dir / "oos_validation_design.json").write_text(
        json.dumps(metadata, indent=2), encoding="utf-8"
    )
    (output_dir / "oos_validation_metrics.json").write_text(
        json.dumps({
            "design": metadata,
            "metrics": metrics.to_dict(orient="records"),
            "pairwise": pairwise.to_dict(orient="records"),
            "loss_tests": loss_tests.to_dict(orient="records"),
        }, indent=2), encoding="utf-8"
    )
    predictions.to_csv(local_dir / "oos_predictions_weekly.csv", index=False)
    print(json.dumps({
        "design": metadata,
        "metrics": metrics.to_dict(orient="records"),
        "pairwise": pairwise.to_dict(orient="records"),
    }, indent=2))
    return metadata, metrics, pairwise, loss_tests


if __name__ == "__main__":
    main()

