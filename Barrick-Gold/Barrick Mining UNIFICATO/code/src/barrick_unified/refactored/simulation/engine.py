"""Composition root for one gold model and common Team 4/5 layers."""

from __future__ import annotations

from barrick_unified.valuation import ValuationInputs, ValuationResult

from ..gold.base import GoldPathResult
from ..operations.margin import ComponentMarginEngine
from ..operations.projection import OperatingProjection
from ..valuation.dcf import Team5DCFValuator
from ..valuation.equity_bridge import EquityBridge
from ..valuation.wacc import WACCProcess


class BarrickScenarioEngine:
    def __init__(self, inputs: ValuationInputs, operations: OperatingProjection) -> None:
        self._inputs = inputs
        self._operations = operations
        self._margin = ComponentMarginEngine()
        self._wacc = WACCProcess(inputs)
        self._dcf = Team5DCFValuator(inputs)
        self._bridge = EquityBridge(inputs)

    def value(self, gold: GoldPathResult, wacc_shocks) -> ValuationResult:
        margin = self._margin.calculate(
            gold_paths_usd_per_oz=gold.quarterly_gold_paths,
            operations=self._operations,
        )
        wacc = self._wacc.simulate(wacc_shocks)
        dcf = self._dcf.value(margin, wacc)
        equity = self._bridge.apply(dcf.enterprise_value_usd_mn.values)
        return ValuationResult(
            gold_price_usd_per_oz=gold.quarterly_gold_paths,
            annual_component_margin_usd_mn=margin.annual_usd_mn.values,
            annual_fcff_proxy_usd_mn=dcf.annual_fcff_proxy_usd_mn.values,
            annual_wacc=dcf.annual_wacc.values,
            pv_explicit_fcff_proxy_usd_mn=dcf.pv_explicit_usd_mn.values,
            pv_terminal_proxy_usd_mn=dcf.pv_terminal_usd_mn.values,
            enterprise_value_proxy_usd_mn=dcf.enterprise_value_usd_mn.values,
            equity_value_proxy_usd_mn=equity.equity_value_usd_mn.values,
            value_per_share_proxy_usd=equity.value_per_share_usd.values,
        )
