"""Explicit Team 5 enterprise-to-equity bridge."""

from __future__ import annotations

from dataclasses import dataclass

import numpy as np

from barrick_unified.valuation import ValuationInputs

from ..domain.units import Unit, UnitArray


@dataclass(frozen=True)
class EquityProjection:
    equity_value_usd_mn: UnitArray
    value_per_share_usd: UnitArray
    data_status: str


class EquityBridge:
    def __init__(self, inputs: ValuationInputs) -> None:
        self._inputs = inputs

    def apply(self, enterprise_value_usd_mn: np.ndarray) -> EquityProjection:
        inputs = self._inputs
        enterprise = np.asarray(enterprise_value_usd_mn, dtype=float)
        if enterprise.shape != (inputs.n_simulations,) or not np.isfinite(enterprise).all():
            raise ValueError("enterprise value must be one finite value per path")
        adjustment = (
            -inputs.debt_usd_mn
            + inputs.cash_usd_mn
            - inputs.minority_interest_usd_mn
            + inputs.non_operating_assets_usd_mn
            - inputs.other_claims_usd_mn
        )
        equity = enterprise + adjustment
        per_share = equity / inputs.diluted_shares_mn
        status = (
            "UNRESOLVED_CORPORATE_PROXIES_CONDITIONAL"
            if inputs.unresolved_corporate_inputs
            else "RECONCILED"
        )
        return EquityProjection(
            UnitArray(equity, Unit.USD_MN, "equity value proxy"),
            UnitArray(per_share, Unit.USD_PER_SHARE, "value per share proxy"),
            status,
        )
