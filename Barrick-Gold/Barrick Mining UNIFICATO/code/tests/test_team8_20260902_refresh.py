from __future__ import annotations

import copy
import json
import sys
from pathlib import Path

import numpy as np
import pytest

from barrick_unified.multimodel_valuation import MODEL_ORDER, run_multimodel_valuation
from barrick_unified.multimodel_valuation import load_multimodel_inputs
from barrick_unified.multimodel_reporting import write_multimodel_outputs, sha256


PROJECT_ROOT = Path(__file__).resolve().parents[1]
CONFIG_PATH = (
    PROJECT_ROOT / "config" / "multimodel_valuation_20260902_team8_refresh.json"
)


@pytest.fixture(autouse=True)
def isolated_snapshot_imports(monkeypatch):
    """Restore legacy imports after each test; keep production cache guards intact."""
    for name in ("BatesHawkesExact", "Hawkes"):
        monkeypatch.delitem(sys.modules, name, raising=False)
    yield
    for name in ("BatesHawkesExact", "Hawkes"):
        sys.modules.pop(name, None)


def _small_config() -> dict:
    payload = json.loads(CONFIG_PATH.read_text(encoding="utf-8"))
    payload["simulation"]["n_simulations"] = 128
    payload["gold_price_layer"]["path_grid"]["fine_steps"] = 20
    payload["gold_price_layer"]["path_grid"]["resampling"] = "test quarterly grid"
    return payload


def test_refresh_contract_serializes_current_team8_snapshot() -> None:
    payload = _small_config()
    snapshot = payload["gold_price_layer"]["calibration_snapshot"]
    assert payload["schema_version"] == "4.0"
    assert snapshot["surface_as_of_utc"] == "2026-09-02"
    assert snapshot["eligible_surface_points_pre_sampling"] == 605
    assert snapshot["dense_cc64_points"] == 64
    assert snapshot["in_sample_best_iv_rmse_model"] == "heston"
    assert snapshot["oos_best_date_equal_iv_rmse_model"] == "full_bates_hawkes"


def test_refresh_paths_are_finite_positive_distinct_and_deterministic() -> None:
    payload = _small_config()
    first = run_multimodel_valuation(PROJECT_ROOT, copy.deepcopy(payload))
    second = run_multimodel_valuation(PROJECT_ROOT, copy.deepcopy(payload))
    hashes: set[bytes] = set()
    for model_id in MODEL_ORDER:
        first_paths = first.models[model_id].quarterly_gold_paths
        second_paths = second.models[model_id].quarterly_gold_paths
        assert first_paths.shape == (128, 20)
        assert np.isfinite(first_paths).all()
        assert np.all(first_paths > 0.0)
        assert np.array_equal(first_paths, second_paths)
        hashes.add(first_paths.tobytes())
    assert len(hashes) == 4


def test_hawkes_snapshot_is_stationary_and_selected_on_oos() -> None:
    payload = json.loads(CONFIG_PATH.read_text(encoding="utf-8"))
    path = PROJECT_ROOT / payload["gold_price_layer"]["parameter_files"][
        "full_bates_hawkes"
    ]
    parameters = json.loads(path.read_text(encoding="utf-8"))["parameters"]
    assert 0.0 <= parameters["branching_ratio"] < 1.0
    np.testing.assert_allclose(
        parameters["alpha"] / parameters["beta"],
        parameters["branching_ratio"],
        rtol=0.0,
        atol=1e-12,
    )
    assert payload["selection_basis"]["primary_structural_scenario"] == (
        "full_bates_hawkes"
    )


def test_current_refresh_reporting_preserves_input_and_artifact_hashes(tmp_path) -> None:
    payload = _small_config()
    config_path = tmp_path / "small-current-config.json"
    config_path.write_text(json.dumps(payload), encoding="utf-8")
    run = run_multimodel_valuation(PROJECT_ROOT, payload)
    _, base_config = load_multimodel_inputs(PROJECT_ROOT, payload)
    run_id = "pytest-september-refresh"
    manifest = write_multimodel_outputs(
        run=run,
        experiment_config=payload,
        experiment_config_path=config_path,
        base_config_path=base_config,
        output_dir=tmp_path / "outputs" / "valuation" / run_id,
        figure_dir=tmp_path / "figures" / "valuation" / run_id,
        manifest_path=tmp_path / "data" / "manifests" / "valuation" / run_id / "run_manifest.json",
        project_root=PROJECT_ROOT,
        run_id=run_id,
    )
    assert manifest["schema_version"] == "4.0"
    assert (tmp_path / "outputs" / "valuation" / run_id / "valuation_summary_by_model.json").is_file()
    for entry in manifest["inputs"] + manifest["code"]:
        candidate = (PROJECT_ROOT / entry["path"]).resolve()
        assert sha256(candidate) == entry["sha256"]
        assert candidate.stat().st_size == entry["bytes"]
    for entry in manifest["artifacts"]:
        candidate = tmp_path / entry["path"]
        assert sha256(candidate) == entry["sha256"]
        assert candidate.stat().st_size == entry["bytes"]
