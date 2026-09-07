"""Conditional four-model gold-price experiment for Barrick valuation.

Only the gold-price layer changes across Black--Scholes/GBM, Heston,
Bates--Poisson and Full Bates--Hawkes. Current Barrick actuals plus the
remaining Team 4 production/cost forecast form a separate operating layer;
the unified DCF/WACC/equity bridge is common. The Team 8 GLD/Q calibration is
used as a conditional distribution-shape bridge, not as a validated
physical-gold forecast. Team 4 price-simulation artifacts are excluded.
"""

from __future__ import annotations

from dataclasses import dataclass
import hashlib
import importlib.util
import json
from pathlib import Path
import sys
from types import ModuleType
from typing import Any

import numpy as np

from .valuation import (
    ValuationInputError,
    ValuationInputs,
    ValuationResult,
    simulate_valuation_from_gold_paths,
)


MODEL_ORDER = (
    "black_scholes",
    "heston",
    "bates_poisson",
    "full_bates_hawkes",
)

MODEL_LABELS = {
    "black_scholes": "Black–Scholes / GBM",
    "heston": "Heston",
    "bates_poisson": "Bates–Poisson",
    "full_bates_hawkes": "Full Bates–Hawkes",
}

# ASCII labels keep CSV, console and LaTeX handoffs portable on Windows.
MODEL_LABELS.update(
    {
        "black_scholes": "Black-Scholes / GBM",
        "heston": "Heston",
        "bates_poisson": "Bates-Poisson",
        "full_bates_hawkes": "Full Bates-Hawkes",
    }
)


def _canonical_hash(value: Any) -> str:
    encoded = json.dumps(
        value, sort_keys=True, separators=(",", ":"), ensure_ascii=False
    ).encode("utf-8")
    return hashlib.sha256(encoded).hexdigest().upper()


def _array_hash(array: np.ndarray) -> str:
    values = np.ascontiguousarray(np.asarray(array, dtype="<f8"))
    return hashlib.sha256(values.tobytes()).hexdigest().upper()


def common_input_fingerprints(inputs: ValuationInputs) -> dict[str, str]:
    """Hash every common non-engine contract separately for audit tests."""

    return {
        "operating_production_koz": _canonical_hash(inputs.production_koz.tolist()),
        "operating_cost_usd_per_oz": _canonical_hash(inputs.cost_usd_per_oz.tolist()),
        "unified_dcf": _canonical_hash(
            {
                "tax_rate": inputs.tax_rate,
                "high_growth": inputs.high_growth,
                "stable_growth": inputs.stable_growth,
                "roic_high": inputs.roic_high,
                "roic_stable": inputs.roic_stable,
                "terminal_spread_floor": inputs.terminal_spread_floor,
            }
        ),
        "unified_wacc_parameters": _canonical_hash(
            {
                "wacc_0": inputs.wacc_0,
                "wacc_long_run": inputs.wacc_long_run,
                "wacc_reversion": inputs.wacc_reversion,
                "wacc_volatility": inputs.wacc_volatility,
            }
        ),
        "equity_bridge": _canonical_hash(
            {
                "debt_usd_mn": inputs.debt_usd_mn,
                "cash_usd_mn": inputs.cash_usd_mn,
                "minority_interest_usd_mn": inputs.minority_interest_usd_mn,
                "non_operating_assets_usd_mn": inputs.non_operating_assets_usd_mn,
                "other_claims_usd_mn": inputs.other_claims_usd_mn,
                "diluted_shares_mn": inputs.diluted_shares_mn,
            }
        ),
        "common_gold_scenario": _canonical_hash(
            {
                "start_usd_per_oz": inputs.gold_price_0_usd_per_oz,
                "drift_source": inputs.gold_drift_source,
                "nss_parameters": inputs.nss_parameters,
                "quarterly_drift": inputs.gold_drift_curve().tolist(),
            }
        ),
    }


def load_team8_path_module(source_dir: Path) -> ModuleType:
    """Load the frozen Team 8 path module while preserving local imports."""

    source_dir = source_dir.resolve()
    module_path = source_dir / "path_simulation.py"
    if not module_path.is_file():
        raise FileNotFoundError(f"Team 8 path module not found: {module_path}")
    for dependency_name in ("BatesHawkesExact", "Hawkes"):
        dependency = sys.modules.get(dependency_name)
        dependency_file = getattr(dependency, "__file__", None)
        if dependency_file is not None and Path(dependency_file).resolve().parent != source_dir:
            raise ImportError(
                f"refusing cached {dependency_name} from a different source tree"
            )
    source_key = hashlib.sha256(str(source_dir).encode("utf-8")).hexdigest()[:16]
    module_name = f"_barrick_frozen_team8_path_simulation_{source_key}"
    cached = sys.modules.get(module_name)
    if cached is not None:
        if Path(str(cached.__file__)).resolve() != module_path:
            raise ImportError("cached Team 8 path module points to another source tree")
        return cached
    spec = importlib.util.spec_from_file_location(module_name, module_path)
    if spec is None or spec.loader is None:
        raise ImportError(f"cannot load Team 8 path module: {module_path}")
    module = importlib.util.module_from_spec(spec)
    original_path = list(sys.path)
    try:
        sys.path.insert(0, str(source_dir))
        spec.loader.exec_module(module)
    finally:
        sys.path[:] = original_path
    sys.modules[module_name] = module
    return module


def _read_parameters(path: Path) -> dict[str, Any]:
    payload = json.loads(path.read_text(encoding="utf-8"))
    parameters = payload.get("parameters")
    if not isinstance(parameters, dict) or not parameters:
        raise ValuationInputError(f"parameter file has no parameters object: {path}")
    return parameters


def _resolve_inside_project(project_root: Path, relative: str, field: str) -> Path:
    path = (project_root / relative).resolve()
    try:
        path.relative_to(project_root.resolve())
    except ValueError as exc:
        raise ValuationInputError(f"{field} must remain inside project root") from exc
    if not path.is_file() and not path.is_dir():
        raise FileNotFoundError(f"{field} not found: {path}")
    return path


@dataclass(frozen=True)
class GoldModelRun:
    model_id: str
    label: str
    engine_seed: int
    parameter_path: Path
    quarterly_gold_paths: np.ndarray
    valuation: ValuationResult


@dataclass(frozen=True)
class MultiModelRun:
    inputs: ValuationInputs
    models: dict[str, GoldModelRun]
    wacc_shocks: np.ndarray
    common_input_hashes: dict[str, str]
    wacc_shocks_sha256: str
    path_grid: dict[str, int | float]
    bridge: dict[str, Any]
    calibration_snapshot: dict[str, Any]
    asynchronous_dates: dict[str, str]


def load_multimodel_inputs(
    project_root: Path, experiment_config: dict[str, Any]
) -> tuple[ValuationInputs, Path]:
    """Load the base contract and apply current simulation/NSS overrides."""

    relative = str(experiment_config.get("base_valuation_config", ""))
    base_path = _resolve_inside_project(
        project_root, relative, "base_valuation_config"
    )
    payload = json.loads(base_path.read_text(encoding="utf-8"))
    layer = experiment_config.get("gold_price_layer", {})
    nss_relative = str(layer.get("nss_curve_file", "")).strip()
    if nss_relative:
        nss_path = _resolve_inside_project(
            project_root, nss_relative, "gold_price_layer.nss_curve_file"
        )
        nss_payload = json.loads(nss_path.read_text(encoding="utf-8"))
        method = nss_payload.get("method", nss_payload.get("model"))
        if method not in {
            "Nelson-Siegel-Svensson nonlinear least squares",
            "Nelson-Siegel-Svensson",
        }:
            raise ValuationInputError("the selected rate artifact is not an approved NSS fit")
        if nss_payload.get("legacy_team4_nss_used", False) is not False:
            raise ValuationInputError("legacy Team 4 NSS must remain excluded")
        required = ("beta0", "beta1", "beta2", "beta3", "tau1", "tau2")
        if "parameters" in nss_payload:
            parameters = nss_payload["parameters"]
        else:
            parameters = {
                name: nss_payload[name] for name in required if name in nss_payload
            }
        if tuple(parameters) != required:
            raise ValuationInputError("current NSS parameters must use canonical order")
        payload["model"].pop("gold_drift_rates_annual", None)
        payload["model"]["gold_drift_nss"] = {
            name: float(parameters[name]) for name in required
        }
        payload["model"]["gold_drift_source"] = (
            "Team 8 US Treasury Nelson-Siegel-Svensson curve dated "
            f"{nss_payload.get('curve_date')}; integral-preserving forward rates"
        )
    simulation = experiment_config.get("simulation", {})
    payload["simulation"] = {
        "n_simulations": int(simulation.get("n_simulations", 0)),
        "seed": int(simulation.get("wacc_seed", -1)),
    }
    return ValuationInputs.from_dict(payload), base_path


def _validate_experiment_contract(
    inputs: ValuationInputs, experiment_config: dict[str, Any]
) -> tuple[int, int, dict[str, int]]:
    layer = experiment_config.get("gold_price_layer", {})
    if experiment_config.get("status") != inputs.status:
        raise ValuationInputError(
            "experiment status must match the base valuation status"
        )
    if layer.get("scope") != "GOLD_PRICE_LAYER_ONLY":
        raise ValuationInputError(
            "gold_price_layer.scope must be GOLD_PRICE_LAYER_ONLY"
        )
    bridge = layer.get("conditional_bridge", {})
    if bridge.get("label") != (
        "conditional transfer of GLD/Q distributional shape to gold USD/oz"
    ):
        raise ValuationInputError(
            "conditional bridge must use the required GLD/Q-to-gold label"
        )
    if bridge.get("validated_physical_forecast") is not False:
        raise ValuationInputError(
            "conditional bridge must explicitly reject a validated physical forecast"
        )
    if bridge.get("q_to_p_mapping") != "NOT_VALIDATED":
        raise ValuationInputError("Q-to-P mapping must remain NOT_VALIDATED")
    common_start = layer.get("common_scenario_start", {})
    if not np.isclose(
        float(common_start.get("price", np.nan)), inputs.gold_price_0_usd_per_oz
    ) or common_start.get("unit") != "USD/troy oz":
        raise ValuationInputError(
            "declared common gold start must match the base valuation contract"
        )
    schema_version = str(experiment_config.get("schema_version", ""))
    if schema_version in {"3.0", "4.0"}:
        if not str(common_start.get("source_authority", "")).strip():
            raise ValuationInputError("current gold anchor requires a source authority")
        if not str(common_start.get("source_date", "")).strip():
            raise ValuationInputError("current gold anchor requires a source date")
        if common_start.get("team4_price_artifact") is not False:
            raise ValuationInputError("Team 4 price artifacts must be explicitly excluded")
        drift = layer.get("common_scenario_drift", {})
        if drift.get("legacy_team4_nss_used") is not False:
            raise ValuationInputError("legacy Team 4 NSS must be explicitly excluded")
        current_curve_enabled = (
            drift.get("current_team8_nss_used") is True
            if schema_version == "4.0"
            else drift.get("current_lse_nss_used") is True
        )
        if not current_curve_enabled:
            raise ValuationInputError("the current Team 8 NSS curve must be explicitly enabled")
    elif common_start.get("source_team") != "Team 4" or common_start.get(
        "source_date"
    ) != "2026-04-02":
        raise ValuationInputError(
            "legacy common gold start must retain Team 4 source and 2026-04-02 date"
        )
    common_layers = experiment_config.get("common_non_gold_layers", {})
    if schema_version in {"3.0", "4.0"}:
        if common_layers.get("production_and_cost") != (
            "CURRENT_Q1_Q2_PLUS_TEAM4_FORECAST_Q3_ONWARD"
        ):
            raise ValuationInputError(
                "current production/cost contract must separate actuals from Team 4 forecasts"
            )
        if common_layers.get("dcf_wacc_equity_bridge") != (
            "COMMON_UNIFIED_DCF_METHODOLOGY_IDENTICAL_ACROSS_MODELS"
        ):
            raise ValuationInputError("the unified DCF contract must be common across models")
    else:
        if common_layers.get("production_and_cost") != (
            "IMPORTED_DETERMINISTIC_TEAM4_20_QUARTER_VECTORS"
        ):
            raise ValuationInputError(
                "production/cost must be deterministic imported Team 4 vectors"
            )
        if common_layers.get("dcf_wacc_equity_bridge") != (
            "COMMON_TEAM5_ADAPTATION_IDENTICAL_ACROSS_MODELS"
        ):
            raise ValuationInputError(
                "DCF/WACC/equity bridge must be the common Team 5 adaptation"
            )
    grid = layer.get("path_grid", {})
    fine_steps = int(grid.get("fine_steps", 0))
    if fine_steps <= 0 or fine_steps % inputs.n_quarters:
        raise ValuationInputError(
            "fine_steps must be a positive multiple of operating quarters"
        )
    steps_per_quarter = fine_steps // inputs.n_quarters
    if not np.isclose(
        float(grid.get("horizon_years", np.nan)),
        inputs.n_quarters * inputs.dt_years,
    ):
        raise ValuationInputError("path horizon must match the operating horizon")
    seeds = {
        key: int(value)
        for key, value in experiment_config.get("simulation", {})
        .get("engine_seeds", {})
        .items()
    }
    if tuple(seeds) != MODEL_ORDER or any(value < 0 for value in seeds.values()):
        raise ValuationInputError(
            "engine_seeds must list all four models in canonical order"
        )
    if len(set(seeds.values())) != 1:
        raise ValuationInputError(
            "all four gold engines must use the same aligned base seed"
        )
    return fine_steps, steps_per_quarter, seeds


def run_multimodel_valuation(
    project_root: Path, experiment_config: dict[str, Any]
) -> MultiModelRun:
    """Generate four conditional gold layers and value each on one common DCF."""

    project_root = project_root.resolve()
    inputs, _ = load_multimodel_inputs(project_root, experiment_config)
    fine_steps, steps_per_quarter, engine_seeds = _validate_experiment_contract(
        inputs, experiment_config
    )
    layer = experiment_config["gold_price_layer"]
    source_dir = _resolve_inside_project(
        project_root, str(layer.get("team8_source_dir", "")), "team8_source_dir"
    )
    team8 = load_team8_path_module(source_dir)
    calibration_manifest = layer.get("calibration_manifest")
    if calibration_manifest:
        snapshot_path = _resolve_inside_project(
            project_root,
            str(calibration_manifest),
            "gold_price_layer.calibration_manifest",
        )
    else:
        snapshot_path = source_dir / "Data" / "lse_publication_manifest.json"
    snapshot_manifest = json.loads(snapshot_path.read_text(encoding="utf-8"))
    calibration_snapshot = dict(layer.get("calibration_snapshot", {}))
    manifest_surface_date = snapshot_manifest.get("as_of_utc", snapshot_manifest.get("date"))
    curve_dates = snapshot_manifest.get("curve_dates", [])
    manifest_curve_date = snapshot_manifest.get(
        "risk_free_rate_curve_date",
        curve_dates[0] if len(curve_dates) == 1 else snapshot_manifest.get("date"),
    )
    if calibration_snapshot.get("surface_as_of_utc") != manifest_surface_date:
        raise ValuationInputError(
            "Team 8 surface date must match the selected calibration manifest"
        )
    if calibration_snapshot.get("treasury_curve_date") != manifest_curve_date:
        raise ValuationInputError(
            "Team 8 curve date must match the selected calibration manifest"
        )
    rate_models = snapshot_manifest.get(
        "rate_curve_models", [snapshot_manifest.get("risk_free_rate_model")]
    )
    if str(experiment_config.get("schema_version")) in {"3.0", "4.0"} and not any(
        model in {"NSS", "Nelson-Siegel-Svensson"} for model in rate_models
    ):
        raise ValuationInputError("Team 8 calibration must use the current NSS curve")
    asynchronous_dates = {
        key: str(value)
        for key, value in experiment_config.get("asynchronous_dates", {}).items()
    }
    anchor_key = "gold_anchor" if "gold_anchor" in asynchronous_dates else "team4_gold_start"
    required_dates = {
        anchor_key: str(layer["common_scenario_start"]["source_date"]),
        "team8_surface": str(manifest_surface_date),
        "team8_treasury_curve": str(manifest_curve_date),
        "b_close": inputs.observed_share_price_timestamp_utc,
        "valuation_as_of": inputs.valuation_as_of_utc,
    }
    if asynchronous_dates != required_dates:
        raise ValuationInputError(
            "asynchronous_dates must exactly serialize all five source dates"
        )
    parameter_files = layer.get("parameter_files", {})
    if tuple(parameter_files) != MODEL_ORDER:
        raise ValuationInputError(
            "parameter_files must list all four models in canonical order"
        )
    parameter_paths = {
        model_id: _resolve_inside_project(
            project_root,
            str(parameter_files[model_id]),
            f"parameter_files.{model_id}",
        )
        for model_id in MODEL_ORDER
    }
    parameters = {
        model_id: _read_parameters(parameter_paths[model_id])
        for model_id in MODEL_ORDER
    }

    dt = inputs.n_quarters * inputs.dt_years / fine_steps
    fine_drift = np.repeat(inputs.gold_drift_curve(), steps_per_quarter)
    if fine_drift.shape != (fine_steps,):
        raise AssertionError("fine drift expansion failed")
    n_paths = inputs.n_simulations
    start = inputs.gold_price_0_usd_per_oz

    generated: dict[str, np.ndarray] = {}
    generated["black_scholes"] = team8.simulate_gbm_paths(
        start,
        fine_drift,
        float(parameters["black_scholes"]["sigma"]),
        dt,
        n_paths,
        engine_seeds["black_scholes"],
    )
    heston = parameters["heston"]
    heston_parameters = (
        float(heston["v0"]),
        float(heston["kappa"]),
        float(heston["theta"]),
        float(heston.get("xi", heston.get("sigma"))),
        float(heston["rho"]),
    )
    generated["heston"] = team8.simulate_heston_paths(
        start,
        fine_drift,
        heston_parameters,
        dt,
        n_paths,
        engine_seeds["heston"],
    )[0]
    bates_parameters = tuple(
        float(parameters["bates_poisson"][name])
        for name in (
            "v0",
            "kappa",
            "theta",
            "sigma",
            "rho",
            "lambd",
            "mu_J",
            "sigma_J",
        )
    )
    generated["bates_poisson"] = team8.simulate_bates_paths(
        start,
        fine_drift,
        bates_parameters,
        dt,
        n_paths,
        engine_seeds["bates_poisson"],
    )[0]
    generated["full_bates_hawkes"] = team8.simulate_full_hawkes_paths(
        start,
        fine_drift,
        parameters["full_bates_hawkes"],
        dt,
        n_paths,
        engine_seeds["full_bates_hawkes"],
    )[0]

    quarterly_indices = np.arange(
        steps_per_quarter, fine_steps + 1, steps_per_quarter
    )
    wacc_seed = int(experiment_config["simulation"]["wacc_seed"])
    wacc_rng = np.random.default_rng(wacc_seed)
    wacc_shocks = wacc_rng.standard_normal((n_paths, inputs.n_years))
    models: dict[str, GoldModelRun] = {}
    for model_id in MODEL_ORDER:
        fine_paths = generated.pop(model_id)
        quarterly = np.ascontiguousarray(fine_paths[:, quarterly_indices])
        if quarterly.shape != (n_paths, inputs.n_quarters):
            raise AssertionError(f"quarterly resampling failed for {model_id}")
        valuation = simulate_valuation_from_gold_paths(
            inputs, quarterly, wacc_shocks
        )
        models[model_id] = GoldModelRun(
            model_id=model_id,
            label=MODEL_LABELS[model_id],
            engine_seed=engine_seeds[model_id],
            parameter_path=parameter_paths[model_id],
            quarterly_gold_paths=quarterly,
            valuation=valuation,
        )

    reference_wacc = models[MODEL_ORDER[0]].valuation.annual_wacc
    for model_id in MODEL_ORDER[1:]:
        if not np.array_equal(
            reference_wacc, models[model_id].valuation.annual_wacc
        ):
            raise AssertionError(f"WACC contamination detected for {model_id}")

    return MultiModelRun(
        inputs=inputs,
        models=models,
        wacc_shocks=wacc_shocks,
        common_input_hashes=common_input_fingerprints(inputs),
        wacc_shocks_sha256=_array_hash(wacc_shocks),
        path_grid={
            "fine_steps": fine_steps,
            "steps_per_quarter": steps_per_quarter,
            "quarters": inputs.n_quarters,
            "horizon_years": inputs.n_quarters * inputs.dt_years,
            "fine_dt_years": dt,
        },
        bridge=dict(layer["conditional_bridge"]),
        calibration_snapshot=calibration_snapshot,
        asynchronous_dates=asynchronous_dates,
    )
