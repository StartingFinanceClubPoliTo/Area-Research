"""Team 5 FCFF and terminal-value identities."""

from __future__ import annotations

from dataclasses import dataclass

import numpy as np

from barrick_unified.valuation import ValuationInputs

from ..domain.units import Unit, UnitArray
from ..operations.margin import MarginProjection
from .wacc import WACCProjection


@dataclass(frozen=True)
class DCFProjection:
    annual_fcff_proxy_usd_mn: UnitArray
    annual_wacc: UnitArray
    pv_explicit_usd_mn: UnitArray
    pv_terminal_usd_mn: UnitArray
    enterprise_value_usd_mn: UnitArray


class Team5DCFValuator:
    def __init__(self, inputs: ValuationInputs) -> None:
        self._inputs = inputs

    def value(self, margin: MarginProjection, wacc: WACCProjection) -> DCFProjection:
        inputs = self._inputs
        annual_margin = margin.annual_usd_mn.values
        annual_wacc = wacc.annual_rates.values
        expected = (inputs.n_simulations, inputs.n_years)
        if annual_margin.shape != expected or annual_wacc.shape != expected:
            raise ValueError(f"DCF inputs must have shape {expected}")
        growth = np.linspace(inputs.high_growth, inputs.stable_growth, inputs.n_years, dtype=float)
        roic = np.linspace(inputs.roic_high, inputs.roic_stable, inputs.n_years, dtype=float)
        reinvestment_rate = growth / roic
        after_tax_margin = annual_margin * (1.0 - inputs.tax_rate)
        fcff = after_tax_margin - np.maximum(after_tax_margin, 0.0) * reinvestment_rate
        discount_factors = 1.0 / np.cumprod(1.0 + annual_wacc, axis=1)
        pv_explicit = np.sum(fcff * discount_factors, axis=1)
        terminal_after_tax = np.maximum(after_tax_margin[:, -1], 0.0) * (
            1.0 + inputs.stable_growth
        )
        terminal_reinvestment = inputs.stable_growth / inputs.roic_stable
        terminal_fcff = terminal_after_tax * (1.0 - terminal_reinvestment)
        terminal_value = terminal_fcff / (annual_wacc[:, -1] - inputs.stable_growth)
        pv_terminal = terminal_value * discount_factors[:, -1]
        enterprise = pv_explicit + pv_terminal
        arrays = (fcff, annual_wacc, pv_explicit, pv_terminal, enterprise)
        if not all(np.isfinite(array).all() for array in arrays):
            raise FloatingPointError("DCF produced non-finite values")
        return DCFProjection(
            UnitArray(fcff, Unit.USD_MN, "annual FCFF proxy"),
            UnitArray(annual_wacc, Unit.DECIMAL_RATE, "annual WACC"),
            UnitArray(pv_explicit, Unit.USD_MN, "PV explicit FCFF proxy"),
            UnitArray(pv_terminal, Unit.USD_MN, "PV terminal proxy"),
            UnitArray(enterprise, Unit.USD_MN, "enterprise value proxy"),
        )
