from __future__ import annotations

import json
from pathlib import Path

import numpy as np

from barrick_unified.valuation import ValuationInputs
from barrick_unified.refactored.valuation.wacc import WACCProcess


ROOT = Path(__file__).resolve().parents[1]
CONFIG = ROOT / "config" / "provisional_valuation_20260827_team4_separated.json"


def _inputs(n_simulations: int = 128) -> ValuationInputs:
    payload = json.loads(CONFIG.read_text(encoding="utf-8"))
    payload["simulation"]["n_simulations"] = n_simulations
    return ValuationInputs.from_dict(payload)


def test_growth_roic_and_reinvestment_schedule_matches_config() -> None:
    inputs = _inputs()
    growth = np.linspace(inputs.high_growth, inputs.stable_growth, inputs.n_years)
    roic = np.linspace(inputs.roic_high, inputs.roic_stable, inputs.n_years)
    reinvestment = growth / roic

    assert np.allclose(growth, [0.085, 0.07, 0.055, 0.04, 0.025])
    assert np.allclose(roic, [0.145, 0.135, 0.125, 0.115, 0.105])
    assert np.allclose(reinvestment, growth / roic)
    assert np.isclose(inputs.stable_growth / inputs.roic_stable, 0.23809523809523808)


def test_zero_shock_wacc_follows_actual_mean_reversion_and_floor() -> None:
    inputs = _inputs()
    rates = WACCProcess(inputs).simulate(
        np.zeros((inputs.n_simulations, inputs.n_years))
    ).annual_rates.values
    persistence = np.exp(-inputs.wacc_reversion)
    expected = []
    state = inputs.wacc_0
    floor = inputs.stable_growth + inputs.terminal_spread_floor
    for _ in range(inputs.n_years):
        state = inputs.wacc_long_run + (state - inputs.wacc_long_run) * persistence
        expected.append(np.clip(state, floor, 0.25))

    assert np.allclose(rates, np.tile(expected, (inputs.n_simulations, 1)))
    assert np.all(rates >= floor)
    assert np.all(np.diff(rates[0]) < 0.0)


def test_terminal_multiple_moves_in_expected_directions() -> None:
    inputs = _inputs()

    def multiple(growth: float, wacc: float) -> float:
        return (1.0 + growth) * (1.0 - growth / inputs.roic_stable) / (
            wacc - growth
        )

    base = multiple(inputs.stable_growth, inputs.wacc_long_run)
    assert multiple(inputs.stable_growth - 0.005, inputs.wacc_long_run) < base
    assert multiple(inputs.stable_growth, inputs.wacc_long_run + 0.005) < base
    assert inputs.wacc_long_run > inputs.stable_growth + inputs.terminal_spread_floor
