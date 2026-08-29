from __future__ import annotations

import copy
import hashlib
import json
import os
from pathlib import Path
import subprocess
import sys

import numpy as np
import pytest

from barrick_unified.multimodel_reporting import (
    multimodel_quantile_rows,
    multimodel_summary_rows,
    write_multimodel_outputs,
)
from barrick_unified.multimodel_valuation import (
    MODEL_ORDER,
    load_multimodel_inputs,
    load_team8_path_module,
    run_multimodel_valuation,
)
from barrick_unified.valuation import QUANTILE_LEVELS, ValuationInputError


PROJECT_ROOT = Path(__file__).resolve().parents[1]
CONFIG_PATH = PROJECT_ROOT / "config" / "multimodel_valuation_20260826.json"
V3_MANIFEST_PATH = (
    PROJECT_ROOT
    / "data"
    / "manifests"
    / "valuation"
    / "20260825T143500Z-provisional-v3"
    / "run_manifest.json"
)


def _sha256(path: Path) -> str:
    digest = hashlib.sha256()
    with path.open("rb") as handle:
        for chunk in iter(lambda: handle.read(1024 * 1024), b""):
            digest.update(chunk)
    return digest.hexdigest().upper()


@pytest.fixture(scope="module")
def small_config() -> dict:
    payload = json.loads(CONFIG_PATH.read_text(encoding="utf-8"))
    payload["simulation"]["n_simulations"] = 128
    payload["gold_price_layer"]["path_grid"]["fine_steps"] = 20
    payload["gold_price_layer"]["path_grid"]["resampling"] = (
        "test grid: every step is one Team 4 quarter"
    )
    return payload


@pytest.fixture(scope="module")
def small_run(small_config: dict):
    return run_multimodel_valuation(PROJECT_ROOT, copy.deepcopy(small_config))


def test_canonical_config_serializes_gold_only_boundary_and_dates() -> None:
    payload = json.loads(CONFIG_PATH.read_text(encoding="utf-8"))
    layer = payload["gold_price_layer"]
    assert layer["scope"] == "GOLD_PRICE_LAYER_ONLY"
    assert layer["conditional_bridge"]["label"] == (
        "conditional transfer of GLD/Q distributional shape to gold USD/oz"
    )
    assert layer["conditional_bridge"]["validated_physical_forecast"] is False
    assert layer["conditional_bridge"]["q_to_p_mapping"] == "NOT_VALIDATED"
    assert payload["common_non_gold_layers"]["production_and_cost"] == (
        "IMPORTED_DETERMINISTIC_TEAM4_20_QUARTER_VECTORS"
    )
    assert payload["asynchronous_dates"] == {
        "team4_gold_start": "2026-04-02",
        "team8_surface": "2026-08-12T20:15:07.826914+00:00",
        "team8_treasury_curve": "2026-07-24",
        "b_close": "2026-08-24T00:00:00Z",
        "valuation_as_of": "2026-08-25T00:00:00Z",
    }
    assert len(set(payload["simulation"]["engine_seeds"].values())) == 1


def test_scipy_is_declared_on_both_dependency_surfaces() -> None:
    pyproject = (PROJECT_ROOT / "pyproject.toml").read_text(encoding="utf-8")
    requirements = (PROJECT_ROOT / "requirements.txt").read_text(encoding="utf-8")
    assert '"scipy>=1.11,<2"' in pyproject
    assert "scipy>=1.11,<2" in requirements.splitlines()


def test_seeded_four_engine_run_is_bitwise_deterministic(small_config: dict) -> None:
    first = run_multimodel_valuation(PROJECT_ROOT, copy.deepcopy(small_config))
    second = run_multimodel_valuation(PROJECT_ROOT, copy.deepcopy(small_config))
    assert np.array_equal(first.wacc_shocks, second.wacc_shocks)
    assert first.wacc_shocks_sha256 == second.wacc_shocks_sha256
    for model_id in MODEL_ORDER:
        assert np.array_equal(
            first.models[model_id].quarterly_gold_paths,
            second.models[model_id].quarterly_gold_paths,
        )
        for field_name in first.models[model_id].valuation.__dataclass_fields__:
            assert np.array_equal(
                getattr(first.models[model_id].valuation, field_name),
                getattr(second.models[model_id].valuation, field_name),
            ), (model_id, field_name)


def test_paths_are_positive_sized_and_engines_are_distinct(small_run) -> None:
    hashes = set()
    for model_id in MODEL_ORDER:
        paths = small_run.models[model_id].quarterly_gold_paths
        assert paths.shape == (128, 20)
        assert np.isfinite(paths).all()
        assert np.all(paths > 0.0)
        hashes.add(hashlib.sha256(paths.tobytes()).hexdigest())
    assert len(hashes) == 4


def test_non_gold_layers_and_wacc_are_identical(small_run) -> None:
    assert set(small_run.common_input_hashes) == {
        "operating_production_koz",
        "operating_cost_usd_per_oz",
        "unified_dcf",
        "unified_wacc_parameters",
        "equity_bridge",
        "common_gold_scenario",
    }
    reference = small_run.models[MODEL_ORDER[0]].valuation.annual_wacc
    for model_id in MODEL_ORDER[1:]:
        assert np.array_equal(reference, small_run.models[model_id].valuation.annual_wacc)


def test_bridge_identities_quantiles_and_summaries(small_run) -> None:
    inputs = small_run.inputs
    adjustment = (
        -inputs.debt_usd_mn
        + inputs.cash_usd_mn
        - inputs.minority_interest_usd_mn
        + inputs.non_operating_assets_usd_mn
        - inputs.other_claims_usd_mn
    )
    for model_id in MODEL_ORDER:
        result = small_run.models[model_id].valuation
        np.testing.assert_allclose(
            result.equity_value_proxy_usd_mn,
            result.enterprise_value_proxy_usd_mn + adjustment,
            rtol=0.0,
            atol=1e-12,
        )
        np.testing.assert_allclose(
            result.value_per_share_proxy_usd,
            result.equity_value_proxy_usd_mn / inputs.diluted_shares_mn,
            rtol=0.0,
            atol=1e-12,
        )
    quantile_rows = multimodel_quantile_rows(small_run)
    for model_id in MODEL_ORDER:
        rows = [row for row in quantile_rows if row["model_id"] == model_id]
        assert tuple(row["quantile_pct"] for row in rows) == QUANTILE_LEVELS
        for field in (
            "enterprise_value_proxy_usd_mn",
            "equity_value_proxy_usd_mn",
            "value_per_share_proxy_usd",
        ):
            assert np.all(np.diff([float(row[field]) for row in rows]) >= 0.0)
    summaries = multimodel_summary_rows(small_run)
    assert tuple(row["model_id"] for row in summaries) == MODEL_ORDER
    for row in summaries:
        assert 0.0 <= row["observed_close_percentile"] <= 100.0
        assert 0.0 <= row["probability_model_value_exceeds_observed"] <= 1.0
        assert row["observed_close_percentile"] / 100.0 == pytest.approx(
            1.0 - row["probability_model_value_exceeds_observed"]
        )


@pytest.mark.parametrize(
    "mutation,match",
    [
        ("bad_label", "required GLD/Q-to-gold label"),
        ("physical_forecast", "reject a validated physical forecast"),
        ("q_to_p", "Q-to-P mapping"),
        ("different_seeds", "same aligned base seed"),
        ("bad_surface_date", "surface date"),
        ("bad_async_dates", "five source dates"),
        ("stochastic_operations", "deterministic imported Team 4 vectors"),
        ("wrong_gold_start", "common gold start"),
        ("wrong_status", "status must match"),
    ],
)
def test_ambiguous_or_incomplete_contract_fails_closed(
    small_config: dict, mutation: str, match: str
) -> None:
    payload = copy.deepcopy(small_config)
    if mutation == "bad_label":
        payload["gold_price_layer"]["conditional_bridge"]["label"] = "transfer"
    elif mutation == "physical_forecast":
        payload["gold_price_layer"]["conditional_bridge"]["validated_physical_forecast"] = True
    elif mutation == "q_to_p":
        payload["gold_price_layer"]["conditional_bridge"]["q_to_p_mapping"] = "IMPLICIT"
    elif mutation == "different_seeds":
        payload["simulation"]["engine_seeds"]["heston"] += 1
    elif mutation == "bad_surface_date":
        payload["gold_price_layer"]["calibration_snapshot"]["surface_as_of_utc"] = "2026-08-25T00:00:00Z"
    elif mutation == "bad_async_dates":
        payload["asynchronous_dates"]["team4_gold_start"] = "2026-08-25"
    elif mutation == "stochastic_operations":
        payload["common_non_gold_layers"]["production_and_cost"] = "RERUN"
    elif mutation == "wrong_gold_start":
        payload["gold_price_layer"]["common_scenario_start"]["price"] = 404.77
    elif mutation == "wrong_status":
        payload["status"] = "FINAL_FAIR_VALUE"
    else:  # pragma: no cover
        raise AssertionError(mutation)
    with pytest.raises(ValuationInputError, match=match):
        run_multimodel_valuation(PROJECT_ROOT, payload)


def test_team8_module_cache_rejects_another_source_tree(
    tmp_path: Path, monkeypatch: pytest.MonkeyPatch
) -> None:
    source = PROJECT_ROOT / "parity" / "sources" / "team-8"
    load_team8_path_module(source)
    fake_dependency = type("Fake", (), {"__file__": str(tmp_path / "Hawkes.py")})()
    monkeypatch.setitem(sys.modules, "Hawkes", fake_dependency)
    with pytest.raises(ImportError, match="different source tree"):
        load_team8_path_module(source)


def test_reporting_manifest_is_hash_complete(tmp_path: Path, small_config: dict, small_run) -> None:
    run_id = "pytest-multimodel"
    _, base_config = load_multimodel_inputs(PROJECT_ROOT, small_config)
    manifest_path = tmp_path / "data" / "manifests" / "valuation" / run_id / "run_manifest.json"
    manifest = write_multimodel_outputs(
        run=small_run,
        experiment_config=small_config,
        experiment_config_path=CONFIG_PATH,
        base_config_path=base_config,
        output_dir=tmp_path / "outputs" / "valuation" / run_id,
        figure_dir=tmp_path / "figures" / "valuation" / run_id,
        manifest_path=manifest_path,
        project_root=PROJECT_ROOT,
        run_id=run_id,
    )
    assert manifest["model_boundary"]["scope"] == "GOLD_PRICE_LAYER_ONLY"
    assert manifest["validated_physical_gold_forecast"] is False
    assert manifest["conditional_bridge"]["label"].startswith("conditional transfer")
    assert manifest["asynchronous_dates"] == small_run.asynchronous_dates
    assert len(manifest["artifacts"]) == 7
    assert len(manifest["inputs"]) >= 16
    assert len(manifest["code"]) == 4
    wacc_hashes = {
        values["annual_wacc_sha256"]
        for values in manifest["model_array_hashes"].values()
    }
    assert wacc_hashes == {small_run.wacc_shocks_sha256} or len(wacc_hashes) == 1
    for entry in manifest["inputs"] + manifest["code"]:
        candidate = (PROJECT_ROOT / entry["path"]).resolve()
        assert candidate.is_file()
        assert candidate.stat().st_size == entry["bytes"]
        assert _sha256(candidate) == entry["sha256"]
    for artifact in manifest["artifacts"]:
        candidate = tmp_path / artifact["path"]
        assert candidate.is_file()
        assert candidate.stat().st_size == artifact["bytes"]
        assert _sha256(candidate) == artifact["sha256"]


def test_cli_end_to_end_and_refuses_overwrite(tmp_path: Path, small_config: dict) -> None:
    config_path = tmp_path / "small_multimodel.json"
    config_path.write_text(json.dumps(small_config), encoding="utf-8")
    run_id = "pytest-multimodel-cli"
    environment = os.environ.copy()
    environment.pop("PYTHONPATH", None)
    command = [
        sys.executable,
        str(PROJECT_ROOT / "run_multimodel_valuation.py"),
        "--config",
        str(config_path),
        "--run-id",
        run_id,
        "--output-root",
        str(tmp_path),
    ]
    first = subprocess.run(
        command,
        cwd=PROJECT_ROOT,
        env=environment,
        capture_output=True,
        text=True,
        check=False,
        timeout=90,
    )
    assert first.returncode == 0, first.stderr
    assert f"Run: {run_id}" in first.stdout
    assert "Scope: GOLD_PRICE_LAYER_ONLY" in first.stdout
    manifest_path = tmp_path / "data" / "manifests" / "valuation" / run_id / "run_manifest.json"
    assert manifest_path.is_file()
    second = subprocess.run(
        command,
        cwd=PROJECT_ROOT,
        env=environment,
        capture_output=True,
        text=True,
        check=False,
        timeout=30,
    )
    assert second.returncode != 0
    assert "refusing to overwrite" in second.stderr


def test_accepted_v3_artifacts_remain_byte_identical() -> None:
    manifest = json.loads(V3_MANIFEST_PATH.read_text(encoding="utf-8"))
    assert manifest["run_id"] == "20260825T143500Z-provisional-v3"
    for artifact in manifest["artifacts"]:
        path = PROJECT_ROOT / artifact["path"]
        assert path.stat().st_size == artifact["bytes"]
        assert _sha256(path) == artifact["sha256"]
