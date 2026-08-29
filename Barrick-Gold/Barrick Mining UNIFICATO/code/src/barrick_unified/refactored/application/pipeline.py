"""Class-based compatibility pipeline for CODE-012.

The legacy module is used only as an input/compatibility boundary. All four
gold engines and all common operating/valuation layers are composed through
the new contracts.
"""

from __future__ import annotations

import json
from pathlib import Path
from typing import Any

import numpy as np

from barrick_unified.multimodel_valuation import (
    MODEL_ORDER,
    GoldModelRun,
    MultiModelRun,
    _read_parameters,
    _resolve_inside_project,
    _validate_experiment_contract,
    common_input_fingerprints,
    load_multimodel_inputs,
    load_team8_path_module,
)

from ..domain.policy import ConditionalBridgePolicy
from ..domain.time import QuarterGrid
from ..gold.base import GoldSimulationContext
from ..gold.registry import GoldModelRegistry, build_frozen_team8_registry
from ..operations.costs import FrozenTeam4CostModel
from ..operations.production import FrozenTeam4ProductionModel
from ..operations.projection import OperatingProjection
from ..simulation.engine import BarrickScenarioEngine
from ..simulation.randomness import RandomStreams


class RefactoredBarrickPipeline:
    def __init__(self, registry: GoldModelRegistry | None = None) -> None:
        self._registry = registry or build_frozen_team8_registry()

    def run(self, project_root: Path, experiment_config: dict[str, Any]) -> MultiModelRun:
        project_root = project_root.resolve()
        inputs, _ = load_multimodel_inputs(project_root, experiment_config)
        fine_steps, _, engine_seeds = _validate_experiment_contract(inputs, experiment_config)
        layer = experiment_config["gold_price_layer"]
        bridge = ConditionalBridgePolicy.from_dict(dict(layer["conditional_bridge"]))
        grid = QuarterGrid(inputs.n_quarters, inputs.dt_years, fine_steps)
        source_dir = _resolve_inside_project(
            project_root, str(layer["team8_source_dir"]), "team8_source_dir"
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
        snapshot = json.loads(snapshot_path.read_text(encoding="utf-8"))
        calibration = dict(layer["calibration_snapshot"])
        if calibration.get("surface_as_of_utc") != snapshot.get("as_of_utc"):
            raise ValueError("Team 8 surface date does not match the selected manifest")
        if calibration.get("treasury_curve_date") != snapshot.get("risk_free_rate_curve_date"):
            raise ValueError("Team 8 curve date does not match the selected manifest")
        if (
            str(experiment_config.get("schema_version")) == "3.0"
            and snapshot.get("risk_free_rate_model") != "Nelson-Siegel-Svensson"
        ):
            raise ValueError("Team 8 calibration must use the current LSE NSS curve")
        asynchronous_dates = {
            key: str(value) for key, value in experiment_config["asynchronous_dates"].items()
        }
        anchor_key = "gold_anchor" if "gold_anchor" in asynchronous_dates else "team4_gold_start"
        required_dates = {
            anchor_key: str(layer["common_scenario_start"]["source_date"]),
            "team8_surface": str(snapshot["as_of_utc"]),
            "team8_treasury_curve": str(snapshot["risk_free_rate_curve_date"]),
            "b_close": inputs.observed_share_price_timestamp_utc,
            "valuation_as_of": inputs.valuation_as_of_utc,
        }
        if asynchronous_dates != required_dates:
            raise ValueError("asynchronous date contract is incomplete")
        parameter_files = layer["parameter_files"]
        if tuple(parameter_files) != MODEL_ORDER:
            raise ValueError("parameter files must follow canonical model order")
        parameter_paths = {
            model_id: _resolve_inside_project(
                project_root, str(parameter_files[model_id]), f"parameter_files.{model_id}"
            )
            for model_id in MODEL_ORDER
        }
        parameters = {model_id: _read_parameters(path) for model_id, path in parameter_paths.items()}
        drift = np.repeat(inputs.gold_drift_curve(), grid.steps_per_quarter)
        streams = RandomStreams.build(
            n_paths=inputs.n_simulations,
            n_years=inputs.n_years,
            wacc_seed=int(experiment_config["simulation"]["wacc_seed"]),
        )
        operations = OperatingProjection(
            FrozenTeam4ProductionModel(inputs).forecast(),
            FrozenTeam4CostModel(inputs).forecast(),
        )
        scenario = BarrickScenarioEngine(inputs, operations)
        model_runs: dict[str, GoldModelRun] = {}
        for model_id in MODEL_ORDER:
            context = GoldSimulationContext(
                grid=grid,
                n_paths=inputs.n_simulations,
                seed=engine_seeds[model_id],
                start_usd_per_oz=inputs.gold_price_0_usd_per_oz,
                fine_drift=drift,
                parameters=parameters[model_id],
                parameter_path=parameter_paths[model_id],
                team8_module=team8,
            )
            gold = self._registry.get(model_id).simulate(context)
            valuation = scenario.value(gold, streams.wacc_shocks)
            model_runs[model_id] = GoldModelRun(
                model_id=model_id,
                label=gold.label,
                engine_seed=gold.seed,
                parameter_path=gold.parameter_path,
                quarterly_gold_paths=gold.quarterly_gold_paths,
                valuation=valuation,
            )
        reference_wacc = model_runs[MODEL_ORDER[0]].valuation.annual_wacc
        if any(
            not np.array_equal(reference_wacc, model_runs[mid].valuation.annual_wacc)
            for mid in MODEL_ORDER[1:]
        ):
            raise AssertionError("WACC contamination across gold engines")
        return MultiModelRun(
            inputs=inputs,
            models=model_runs,
            wacc_shocks=streams.wacc_shocks,
            common_input_hashes=common_input_fingerprints(inputs),
            wacc_shocks_sha256=streams.wacc_sha256,
            path_grid=grid.as_dict(),
            bridge=bridge.as_dict(),
            calibration_snapshot=calibration,
            asynchronous_dates=asynchronous_dates,
        )
