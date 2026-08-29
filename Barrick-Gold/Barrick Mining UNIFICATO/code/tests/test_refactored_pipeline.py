from __future__ import annotations

import copy
import json
from pathlib import Path
import shutil
from uuid import uuid4

import numpy as np
import pytest

from barrick_unified.multimodel_valuation import MODEL_ORDER, run_multimodel_valuation
from barrick_unified.refactored import RefactoredBarrickPipeline
from barrick_unified.refactored.domain.policy import ConditionalBridgePolicy
from barrick_unified.refactored.domain.time import QuarterGrid
from barrick_unified.refactored.domain.units import Unit, UnitArray
from barrick_unified.refactored.reporting import ThesisFigureService


ROOT = Path(__file__).resolve().parents[1]
CONFIG = ROOT / "config" / "multimodel_valuation_20260827_team4_separated.json"


@pytest.fixture(scope="module")
def small_config() -> dict:
    payload = json.loads(CONFIG.read_text(encoding="utf-8"))
    payload["simulation"]["n_simulations"] = 128
    payload["gold_price_layer"]["path_grid"]["fine_steps"] = 20
    payload["gold_price_layer"]["path_grid"]["resampling"] = "one fine step per quarter"
    return payload


def test_domain_boundaries_fail_closed() -> None:
    with pytest.raises(ValueError, match="multiple of four"):
        QuarterGrid(19, 0.25, 38)
    with pytest.raises(ValueError, match="strictly positive"):
        UnitArray(np.array([1.0, 0.0]), Unit.PRODUCTION_KOZ, "production").require_positive()
    with pytest.raises(ValueError, match="physical forecast"):
        ConditionalBridgePolicy.from_dict(
            {
                "label": "conditional transfer of GLD/Q distributional shape to gold USD/oz",
                "source_instrument": "GLD calls",
                "destination_variable": "gold USD/oz",
                "validated_physical_forecast": True,
                "q_to_p_mapping": "NOT_VALIDATED",
                "gld_to_gold_level_conversion": "NOT_PERFORMED",
                "interpretation": "not a target price or investment recommendation",
            }
        )


def test_refactored_pipeline_is_bitwise_equal_to_legacy(small_config: dict) -> None:
    legacy = run_multimodel_valuation(ROOT, copy.deepcopy(small_config))
    refactored = RefactoredBarrickPipeline().run(ROOT, copy.deepcopy(small_config))
    assert refactored.path_grid == legacy.path_grid
    assert refactored.bridge == legacy.bridge
    assert refactored.common_input_hashes == legacy.common_input_hashes
    assert refactored.wacc_shocks_sha256 == legacy.wacc_shocks_sha256
    assert np.array_equal(refactored.wacc_shocks, legacy.wacc_shocks)
    for model_id in MODEL_ORDER:
        left = legacy.models[model_id]
        right = refactored.models[model_id]
        assert left.label == right.label
        assert left.engine_seed == right.engine_seed
        assert left.parameter_path == right.parameter_path
        assert np.array_equal(left.quarterly_gold_paths, right.quarterly_gold_paths)
        for field in left.valuation.__dataclass_fields__:
            assert np.array_equal(
                getattr(left.valuation, field), getattr(right.valuation, field)
            ), (model_id, field)


def test_separated_operating_layers_do_not_depend_on_gold_model(small_config: dict) -> None:
    run = RefactoredBarrickPipeline().run(ROOT, copy.deepcopy(small_config))
    reference_wacc = run.models[MODEL_ORDER[0]].valuation.annual_wacc
    for model_id in MODEL_ORDER[1:]:
        assert np.array_equal(reference_wacc, run.models[model_id].valuation.annual_wacc)
    assert len({run.common_input_hashes["operating_production_koz"]}) == 1
    assert len({run.common_input_hashes["operating_cost_usd_per_oz"]}) == 1
    assert small_config["gold_price_layer"]["common_scenario_start"]["team4_price_artifact"] is False
    assert small_config["gold_price_layer"]["common_scenario_drift"]["team4_nss_used"] is False
    assert small_config["gold_price_layer"]["common_scenario_drift"]["legacy_team4_nss_used"] is False
    assert small_config["gold_price_layer"]["common_scenario_drift"]["current_lse_nss_used"] is True
    assert run.inputs.nss_parameters is not None
    assert "Nelson-Siegel-Svensson" in run.inputs.gold_drift_source


def test_thesis_figures_have_png_pdf_csv_json_and_manifest(
    small_config: dict,
) -> None:
    run = RefactoredBarrickPipeline().run(ROOT, copy.deepcopy(small_config))
    scratch = ROOT / "test-temp-parent" / f"figure-{uuid4().hex}"
    output = scratch / "figures"
    try:
        manifest = ThesisFigureService().build(
            run=run,
            output_dir=output,
            run_id="pytest-figures",
            input_paths=[CONFIG],
            code_paths=[ROOT / "run_refactored_thesis_figures.py"],
        )
        assert manifest["status"] == "FIGURE_OUTPUT_CANDIDATE"
        assert len(manifest["artifacts"]) == 24
        assert {path.suffix for path in output.iterdir()} == {
            ".png",
            ".pdf",
            ".csv",
            ".json",
        }
        for artifact in manifest["artifacts"]:
            assert (output.parent / artifact["path"]).is_file()

        assert not (output / "fig_ops_inputs.png").exists()
        assert not (output / "fig_ops_ebitda.png").exists()
        assert manifest["model_boundary"]["team4_price_demonstrator"] == "EXCLUDED"
    finally:
        shutil.rmtree(scratch, ignore_errors=True)
