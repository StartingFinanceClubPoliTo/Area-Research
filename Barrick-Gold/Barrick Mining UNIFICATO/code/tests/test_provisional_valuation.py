from __future__ import annotations

import copy
import hashlib
import json
import os
import subprocess
import sys
from pathlib import Path

import numpy as np
import pytest

from barrick_unified.valuation import (
    QUANTILE_LEVELS,
    ValuationInputError,
    ValuationInputs,
    quantile_rows,
    simulate_valuation,
)


PROJECT_ROOT = Path(__file__).resolve().parents[1]
CONFIG_PATH = PROJECT_ROOT / "config" / "provisional_valuation_20260825.json"


@pytest.fixture
def base_payload() -> dict:
    payload = json.loads(CONFIG_PATH.read_text(encoding="utf-8"))
    payload["simulation"]["n_simulations"] = 2_048
    return payload


def _inputs(payload: dict) -> ValuationInputs:
    return ValuationInputs.from_dict(copy.deepcopy(payload))


def _sha256(path: Path) -> str:
    digest = hashlib.sha256()
    with path.open("rb") as handle:
        for chunk in iter(lambda: handle.read(1024 * 1024), b""):
            digest.update(chunk)
    return digest.hexdigest().upper()


def test_seeded_replay_is_bitwise_deterministic(base_payload: dict) -> None:
    first = simulate_valuation(_inputs(base_payload))
    second = simulate_valuation(_inputs(base_payload))

    for field_name in first.__dataclass_fields__:
        first_array = getattr(first, field_name)
        second_array = getattr(second, field_name)
        assert np.array_equal(first_array, second_array), field_name


def test_quantiles_are_canonical_monotone_and_preserve_bridge_identities(
    base_payload: dict,
) -> None:
    inputs = _inputs(base_payload)
    result = simulate_valuation(inputs)
    bridge_adjustment = (
        -inputs.debt_usd_mn
        + inputs.cash_usd_mn
        - inputs.minority_interest_usd_mn
        + inputs.non_operating_assets_usd_mn
        - inputs.other_claims_usd_mn
    )

    np.testing.assert_allclose(
        result.equity_value_proxy_usd_mn,
        result.enterprise_value_proxy_usd_mn + bridge_adjustment,
        rtol=0.0,
        atol=1e-12,
    )
    np.testing.assert_allclose(
        result.value_per_share_proxy_usd,
        result.equity_value_proxy_usd_mn / inputs.diluted_shares_mn,
        rtol=0.0,
        atol=1e-12,
    )

    rows = quantile_rows(result)
    assert tuple(row["quantile_pct"] for row in rows) == QUANTILE_LEVELS
    value_fields = (
        "enterprise_value_proxy_usd_mn",
        "equity_value_proxy_usd_mn",
        "value_per_share_proxy_usd",
    )
    for field_name in value_fields:
        values = np.asarray([row[field_name] for row in rows], dtype=float)
        assert np.all(np.diff(values) >= 0.0), field_name

    for row in rows:
        assert row["equity_value_proxy_usd_mn"] == pytest.approx(
            row["enterprise_value_proxy_usd_mn"] + bridge_adjustment
        )
        assert row["value_per_share_proxy_usd"] == pytest.approx(
            row["equity_value_proxy_usd_mn"] / inputs.diluted_shares_mn
        )


@pytest.mark.parametrize(
    ("case", "message"),
    [
        ("too_few_paths", "at least 100"),
        ("future_market_observation", "after the valuation as-of"),
        ("zero_production", "production and unit costs"),
        ("negative_gold_volatility", "gold volatility"),
        ("unapproved_proxy", "allow_unresolved_corporate_proxies=true"),
        ("zero_nss_tau", "tau1"),
        ("nonfinite_nss_beta", "NSS parameters"),
        ("nonfinite_initial_wacc", "initial WACC"),
        ("nonfinite_long_run_wacc", "long-run WACC"),
        ("nonfinite_wacc_volatility", "wacc_volatility"),
        ("terminal_floor_above_wacc_cap", "terminal spread"),
    ],
)
def test_invalid_inputs_fail_at_contract_boundary(
    base_payload: dict, case: str, message: str
) -> None:
    payload = copy.deepcopy(base_payload)
    if case == "too_few_paths":
        payload["simulation"]["n_simulations"] = 99
    elif case == "future_market_observation":
        payload["market_price"]["timestamp_utc"] = "2026-08-26T00:00:00Z"
    elif case == "zero_production":
        payload["model"]["production_koz"][0] = 0.0
    elif case == "negative_gold_volatility":
        payload["model"]["gold_volatility_annual"][0] = -0.01
    elif case == "unapproved_proxy":
        payload["allow_unresolved_corporate_proxies"] = False
    elif case == "zero_nss_tau":
        payload["model"]["gold_drift_nss"]["tau1"] = 0.0
    elif case == "nonfinite_nss_beta":
        payload["model"]["gold_drift_nss"]["beta0"] = float("nan")
    elif case == "nonfinite_initial_wacc":
        payload["model"]["wacc_0"] = float("nan")
    elif case == "nonfinite_long_run_wacc":
        payload["model"]["wacc_long_run"] = float("nan")
    elif case == "nonfinite_wacc_volatility":
        payload["model"]["wacc_volatility"] = float("nan")
    elif case == "terminal_floor_above_wacc_cap":
        payload["model"]["stable_growth"] = 0.24
        payload["model"]["roic_stable"] = 0.50
        payload["model"]["terminal_spread_floor"] = 0.02
        payload["model"]["wacc_0"] = 0.30
        payload["model"]["wacc_long_run"] = 0.30
    else:  # pragma: no cover - protects the parametrized case registry
        raise AssertionError(f"unknown invalid-input case: {case}")

    with pytest.raises(ValuationInputError, match=message):
        ValuationInputs.from_dict(payload)


def test_directional_sensitivities_for_gold_cost_and_wacc(base_payload: dict) -> None:
    base = simulate_valuation(_inputs(base_payload))

    higher_gold_payload = copy.deepcopy(base_payload)
    higher_gold_payload["model"]["gold_price_0_usd_per_oz"] *= 1.05
    higher_gold = simulate_valuation(_inputs(higher_gold_payload))

    higher_cost_payload = copy.deepcopy(base_payload)
    higher_cost_payload["model"]["cost_usd_per_oz"] = [
        value * 1.05 for value in higher_cost_payload["model"]["cost_usd_per_oz"]
    ]
    higher_cost = simulate_valuation(_inputs(higher_cost_payload))

    higher_wacc_payload = copy.deepcopy(base_payload)
    higher_wacc_payload["model"]["wacc_0"] += 0.01
    higher_wacc_payload["model"]["wacc_long_run"] += 0.01
    higher_wacc = simulate_valuation(_inputs(higher_wacc_payload))

    assert np.median(higher_gold.value_per_share_proxy_usd) > np.median(
        base.value_per_share_proxy_usd
    )
    assert np.median(higher_cost.value_per_share_proxy_usd) < np.median(
        base.value_per_share_proxy_usd
    )
    assert np.median(higher_wacc.value_per_share_proxy_usd) < np.median(
        base.value_per_share_proxy_usd
    )


def test_cli_writes_complete_outputs_to_requested_root(tmp_path: Path) -> None:
    run_id = "pytest-provisional-valuation"
    environment = os.environ.copy()
    existing_pythonpath = environment.get("PYTHONPATH", "")
    environment["PYTHONPATH"] = os.pathsep.join(
        part
        for part in (str(PROJECT_ROOT / "src"), existing_pythonpath)
        if part
    )
    completed = subprocess.run(
        [
            sys.executable,
            str(PROJECT_ROOT / "run_provisional_valuation.py"),
            "--run-id",
            run_id,
            "--output-root",
            str(tmp_path),
        ],
        cwd=PROJECT_ROOT,
        env=environment,
        check=False,
        capture_output=True,
        text=True,
        timeout=60,
    )
    assert completed.returncode == 0, completed.stderr
    assert f"Run: {run_id}" in completed.stdout

    output_dir = tmp_path / "outputs" / "valuation" / run_id
    expected_outputs = (
        output_dir / "valuation_quantiles.csv",
        output_dir / "valuation_summary.json",
        output_dir / "valuation_summary.csv",
        output_dir / "valuation_bridges.csv",
        output_dir / "valuation_quantiles.tex",
        tmp_path
        / "figures"
        / "valuation"
        / run_id
        / "provisional_value_per_share_distribution.png",
        tmp_path
        / "data"
        / "manifests"
        / "valuation"
        / run_id
        / "run_manifest.json",
    )
    assert all(path.is_file() and path.stat().st_size > 0 for path in expected_outputs)

    manifest_path = expected_outputs[-1]
    manifest = json.loads(manifest_path.read_text(encoding="utf-8"))
    assert manifest["run_id"] == run_id
    assert manifest["status"] == "PROVISIONAL_RESEARCH_SENSITIVITY_NOT_TARGET"
    assert len(manifest["artifacts"]) == 6
    for artifact in manifest["artifacts"]:
        relative_path = Path(artifact["path"])
        assert not relative_path.is_absolute()
        assert ".." not in relative_path.parts
        artifact_path = tmp_path / relative_path
        assert artifact_path.stat().st_size == artifact["bytes"]
        assert _sha256(artifact_path) == artifact["sha256"]
