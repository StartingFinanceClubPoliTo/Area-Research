"""Explicitly provisional stochastic Barrick valuation engine.

The engine keeps the operating projection, DCF assumptions and gold-price
engines as separate contracts.  Historical configurations may still use an
NSS drift, while current configurations can provide a directly observed and
versioned quarterly rate curve.  External gold paths never depend on the
Team 4 demonstrator price process.
"""

from __future__ import annotations

from dataclasses import dataclass
from datetime import datetime, timezone
from typing import Any

import numpy as np


QUANTILE_LEVELS = (1, 5, 10, 25, 50, 75, 90, 95, 99)


class ValuationInputError(ValueError):
    """Raised when the provisional valuation contract is not explicit."""


def _utc(value: str, field: str) -> datetime:
    try:
        parsed = datetime.fromisoformat(value.replace("Z", "+00:00"))
    except (TypeError, ValueError) as exc:
        raise ValuationInputError(f"{field} must be an ISO-8601 timestamp") from exc
    if parsed.tzinfo is None:
        raise ValuationInputError(f"{field} must include a timezone")
    return parsed.astimezone(timezone.utc)


def _float_array(value: Any, field: str, length: int) -> np.ndarray:
    try:
        array = np.asarray(value, dtype=float)
    except (TypeError, ValueError) as exc:
        raise ValuationInputError(f"{field} must be numeric") from exc
    if array.shape != (length,):
        raise ValuationInputError(f"{field} must contain exactly {length} values")
    if not np.isfinite(array).all():
        raise ValuationInputError(f"{field} contains non-finite values")
    return array


@dataclass(frozen=True)
class ValuationInputs:
    """Validated, unit-explicit inputs for one provisional research run."""

    schema_version: str
    status: str
    valuation_as_of_utc: str
    reporting_currency: str
    reporting_scale: str
    n_simulations: int
    seed: int
    dt_years: float
    observed_share_price_usd: float
    observed_share_price_timestamp_utc: str
    observed_share_price_source: str
    gold_price_0_usd_per_oz: float
    production_koz: np.ndarray
    cost_usd_per_oz: np.ndarray
    gold_volatility_annual: np.ndarray
    gold_drift_rates_annual: np.ndarray | None
    nss_parameters: dict[str, float] | None
    gold_drift_source: str
    tax_rate: float
    high_growth: float
    stable_growth: float
    roic_high: float
    roic_stable: float
    wacc_0: float
    wacc_long_run: float
    wacc_reversion: float
    wacc_volatility: float
    terminal_spread_floor: float
    debt_usd_mn: float
    cash_usd_mn: float
    minority_interest_usd_mn: float
    non_operating_assets_usd_mn: float
    other_claims_usd_mn: float
    diluted_shares_mn: float
    allow_unresolved_corporate_proxies: bool
    unresolved_corporate_inputs: tuple[str, ...]
    assumptions: tuple[dict[str, Any], ...]
    reference_files: tuple[dict[str, Any], ...]

    @property
    def n_quarters(self) -> int:
        return int(self.production_koz.size)

    @property
    def n_years(self) -> int:
        return self.n_quarters // 4

    @classmethod
    def from_dict(cls, payload: dict[str, Any]) -> "ValuationInputs":
        model = payload.get("model", {})
        market = payload.get("market_price", {})
        bridge = payload.get("equity_bridge", {})
        simulation = payload.get("simulation", {})
        quarters = int(model.get("quarters", 0))
        if quarters < 4 or quarters % 4:
            raise ValuationInputError("model.quarters must be a positive multiple of four")

        direct_drift_raw = model.get("gold_drift_rates_annual")
        direct_drift = None
        nss: dict[str, float] | None = None
        if direct_drift_raw is not None:
            direct_drift = _float_array(
                direct_drift_raw,
                "model.gold_drift_rates_annual",
                quarters,
            )
        else:
            nss_raw = model.get("gold_drift_nss", {})
            required_nss = ("beta0", "beta1", "beta2", "beta3", "tau1", "tau2")
            missing_nss = [name for name in required_nss if name not in nss_raw]
            if missing_nss:
                raise ValuationInputError(
                    "model must provide gold_drift_rates_annual or a complete "
                    f"gold_drift_nss; missing NSS fields: {missing_nss}"
                )
            nss = {name: float(nss_raw[name]) for name in required_nss}

        result = cls(
            schema_version=str(payload.get("schema_version", "")),
            status=str(payload.get("status", "")),
            valuation_as_of_utc=str(payload.get("valuation_as_of_utc", "")),
            reporting_currency=str(payload.get("reporting_currency", "")),
            reporting_scale=str(payload.get("reporting_scale", "")),
            n_simulations=int(simulation.get("n_simulations", 0)),
            seed=int(simulation.get("seed", -1)),
            dt_years=float(model.get("dt_years", 0.0)),
            observed_share_price_usd=float(market.get("price_usd", np.nan)),
            observed_share_price_timestamp_utc=str(market.get("timestamp_utc", "")),
            observed_share_price_source=str(market.get("source", "")),
            gold_price_0_usd_per_oz=float(model.get("gold_price_0_usd_per_oz", np.nan)),
            production_koz=_float_array(model.get("production_koz"), "model.production_koz", quarters),
            cost_usd_per_oz=_float_array(model.get("cost_usd_per_oz"), "model.cost_usd_per_oz", quarters),
            gold_volatility_annual=_float_array(
                model.get("gold_volatility_annual"), "model.gold_volatility_annual", quarters
            ),
            gold_drift_rates_annual=direct_drift,
            nss_parameters=nss,
            gold_drift_source=str(model.get("gold_drift_source", "legacy NSS configuration")),
            tax_rate=float(model.get("tax_rate", np.nan)),
            high_growth=float(model.get("high_growth", np.nan)),
            stable_growth=float(model.get("stable_growth", np.nan)),
            roic_high=float(model.get("roic_high", np.nan)),
            roic_stable=float(model.get("roic_stable", np.nan)),
            wacc_0=float(model.get("wacc_0", np.nan)),
            wacc_long_run=float(model.get("wacc_long_run", np.nan)),
            wacc_reversion=float(model.get("wacc_reversion", np.nan)),
            wacc_volatility=float(model.get("wacc_volatility", np.nan)),
            terminal_spread_floor=float(model.get("terminal_spread_floor", np.nan)),
            debt_usd_mn=float(bridge.get("debt_usd_mn", np.nan)),
            cash_usd_mn=float(bridge.get("cash_usd_mn", np.nan)),
            minority_interest_usd_mn=float(bridge.get("minority_interest_usd_mn", np.nan)),
            non_operating_assets_usd_mn=float(bridge.get("non_operating_assets_usd_mn", np.nan)),
            other_claims_usd_mn=float(bridge.get("other_claims_usd_mn", np.nan)),
            diluted_shares_mn=float(bridge.get("diluted_shares_mn", np.nan)),
            allow_unresolved_corporate_proxies=bool(
                payload.get("allow_unresolved_corporate_proxies", False)
            ),
            unresolved_corporate_inputs=tuple(payload.get("unresolved_corporate_inputs", ())),
            assumptions=tuple(payload.get("assumptions", ())),
            reference_files=tuple(payload.get("reference_files", ())),
        )
        result.validate()
        return result

    def validate(self) -> None:
        if self.schema_version != "1.0":
            raise ValuationInputError("schema_version must be 1.0")
        if self.status != "PROVISIONAL_RESEARCH_SENSITIVITY_NOT_TARGET":
            raise ValuationInputError("status must explicitly mark a provisional non-target run")
        as_of = _utc(self.valuation_as_of_utc, "valuation_as_of_utc")
        observed_at = _utc(
            self.observed_share_price_timestamp_utc,
            "market_price.timestamp_utc",
        )
        if observed_at > as_of:
            raise ValuationInputError("observed share price is after the valuation as-of")
        if self.reporting_currency != "USD" or self.reporting_scale != "millions":
            raise ValuationInputError("this adapter requires USD and millions")
        if self.gold_drift_rates_annual is not None:
            if not np.isfinite(self.gold_drift_rates_annual).all():
                raise ValuationInputError("direct gold drift rates must be finite")
            if np.any(np.abs(self.gold_drift_rates_annual) > 0.50):
                raise ValuationInputError("direct gold drift rates exceed the 50% validation bound")
            if not self.gold_drift_source.strip():
                raise ValuationInputError("model.gold_drift_source is required for direct rates")
        elif self.nss_parameters is not None:
            nss_values = np.asarray(tuple(self.nss_parameters.values()), dtype=float)
            if not np.isfinite(nss_values).all():
                raise ValuationInputError("NSS parameters must be finite")
            if self.nss_parameters["tau1"] <= 0.0 or self.nss_parameters["tau2"] <= 0.0:
                raise ValuationInputError("NSS tau1 and tau2 must be strictly positive")
        else:  # pragma: no cover - constructor is internal and from_dict fails closed
            raise ValuationInputError("a gold drift curve is required")
        if self.n_simulations < 100:
            raise ValuationInputError("simulation.n_simulations must be at least 100")
        if self.seed < 0:
            raise ValuationInputError("simulation.seed must be non-negative")
        if not np.isclose(self.dt_years, 0.25):
            raise ValuationInputError("operating arrays require quarterly dt_years=0.25")
        if not self.observed_share_price_source.strip():
            raise ValuationInputError("market_price.source is required")
        positive_scalars = {
            "observed share price": self.observed_share_price_usd,
            "gold price": self.gold_price_0_usd_per_oz,
            "diluted shares": self.diluted_shares_mn,
            "ROIC high": self.roic_high,
            "ROIC stable": self.roic_stable,
            "initial WACC": self.wacc_0,
            "long-run WACC": self.wacc_long_run,
            "WACC reversion": self.wacc_reversion,
            "terminal spread floor": self.terminal_spread_floor,
        }
        for name, value in positive_scalars.items():
            if not np.isfinite(value) or value <= 0.0:
                raise ValuationInputError(f"{name} must be finite and positive")
        if np.any(self.production_koz <= 0.0) or np.any(self.cost_usd_per_oz <= 0.0):
            raise ValuationInputError("production and unit costs must be strictly positive")
        if np.any(self.gold_volatility_annual < 0.0):
            raise ValuationInputError("gold volatility cannot be negative")
        if not 0.0 <= self.tax_rate < 1.0:
            raise ValuationInputError("tax_rate must be in [0, 1)")
        if not 0.0 <= self.stable_growth < self.roic_stable:
            raise ValuationInputError("stable_growth must be non-negative and below stable ROIC")
        if not 0.0 <= self.high_growth < self.roic_high:
            raise ValuationInputError("high_growth must be non-negative and below high ROIC")
        if self.wacc_long_run <= self.stable_growth + self.terminal_spread_floor:
            raise ValuationInputError("long-run WACC must exceed growth plus terminal spread floor")
        if self.wacc_0 <= self.stable_growth + self.terminal_spread_floor:
            raise ValuationInputError("initial WACC must exceed growth plus terminal spread floor")
        if self.stable_growth + self.terminal_spread_floor >= 0.25:
            raise ValuationInputError(
                "stable growth plus terminal spread floor must remain below the 25% WACC cap"
            )
        if not np.isfinite(self.wacc_volatility) or self.wacc_volatility < 0.0:
            raise ValuationInputError("wacc_volatility must be finite and non-negative")
        bridge_values = (
            self.debt_usd_mn,
            self.cash_usd_mn,
            self.minority_interest_usd_mn,
            self.non_operating_assets_usd_mn,
            self.other_claims_usd_mn,
        )
        if not np.isfinite(bridge_values).all() or np.any(np.asarray(bridge_values) < 0.0):
            raise ValuationInputError("equity-bridge components must be finite and non-negative")
        if self.unresolved_corporate_inputs and not self.allow_unresolved_corporate_proxies:
            raise ValuationInputError(
                "unresolved corporate proxies require allow_unresolved_corporate_proxies=true"
            )
        if not self.assumptions:
            raise ValuationInputError("an explicit assumption registry is required")

    def gold_drift_curve(self) -> np.ndarray:
        if self.gold_drift_rates_annual is not None:
            return self.gold_drift_rates_annual.copy()
        if self.nss_parameters is None:  # pragma: no cover - validated above
            raise ValuationInputError("a gold drift curve is required")
        p = self.nss_parameters
        times = np.arange(1, self.n_quarters + 1, dtype=float) * self.dt_years
        x1 = times / p["tau1"]
        x2 = times / p["tau2"]
        level1 = (1.0 - np.exp(-x1)) / x1
        level2 = (1.0 - np.exp(-x2)) / x2
        zero_rates = (
            p["beta0"]
            + p["beta1"] * level1
            + p["beta2"] * (level1 - np.exp(-x1))
            + p["beta3"] * (level2 - np.exp(-x2))
        )
        edges = np.arange(self.n_quarters + 1, dtype=float) * self.dt_years
        positive_edges = edges.copy()
        positive_edges[0] = min(self.dt_years / 1000.0, 1e-4)
        x1_edges = positive_edges / p["tau1"]
        x2_edges = positive_edges / p["tau2"]
        l1_edges = (1.0 - np.exp(-x1_edges)) / x1_edges
        l2_edges = (1.0 - np.exp(-x2_edges)) / x2_edges
        edge_zero_rates = (
            p["beta0"]
            + p["beta1"] * l1_edges
            + p["beta2"] * (l1_edges - np.exp(-x1_edges))
            + p["beta3"] * (l2_edges - np.exp(-x2_edges))
        )
        integrated = edges * edge_zero_rates
        integrated[0] = 0.0
        forward_rates = np.diff(integrated) / self.dt_years
        if not np.allclose(
            np.sum(forward_rates * self.dt_years),
            zero_rates[-1] * times[-1],
            atol=1e-12,
            rtol=0.0,
        ):
            raise ValuationInputError("NSS forward-rate integral is inconsistent")
        return forward_rates


@dataclass(frozen=True)
class ValuationResult:
    gold_price_usd_per_oz: np.ndarray
    annual_component_margin_usd_mn: np.ndarray
    annual_fcff_proxy_usd_mn: np.ndarray
    annual_wacc: np.ndarray
    pv_explicit_fcff_proxy_usd_mn: np.ndarray
    pv_terminal_proxy_usd_mn: np.ndarray
    enterprise_value_proxy_usd_mn: np.ndarray
    equity_value_proxy_usd_mn: np.ndarray
    value_per_share_proxy_usd: np.ndarray


def _trend(start: float, end: float, periods: int) -> np.ndarray:
    return np.linspace(start, end, periods, dtype=float)


def simulate_valuation(inputs: ValuationInputs) -> ValuationResult:
    """Run one vectorized, deterministic-seed provisional valuation."""

    rng = np.random.default_rng(inputs.seed)
    gold_shocks = rng.standard_normal((inputs.n_simulations, inputs.n_quarters))
    rates = inputs.gold_drift_curve()[None, :]
    vols = inputs.gold_volatility_annual[None, :]
    increments = (
        (rates - 0.5 * vols**2) * inputs.dt_years
        + vols * np.sqrt(inputs.dt_years) * gold_shocks
    )
    gold = inputs.gold_price_0_usd_per_oz * np.exp(np.cumsum(increments, axis=1))

    # Preserve the legacy RNG order: WACC draws follow the temporary GBM gold
    # draws from the same generator.  New multi-model runs pass an independently
    # seeded common WACC matrix to ``simulate_valuation_from_gold_paths``.
    wacc_shocks = rng.standard_normal((inputs.n_simulations, inputs.n_years))
    return simulate_valuation_from_gold_paths(inputs, gold, wacc_shocks)


def simulate_valuation_from_gold_paths(
    inputs: ValuationInputs,
    gold_price_usd_per_oz: np.ndarray,
    wacc_shocks: np.ndarray,
    terminal_policy: str = "legacy_positive",
) -> ValuationResult:
    """Value Barrick from external quarterly gold paths.

    ``gold_price_usd_per_oz`` contains one price for each operating
    quarter (the initial time-zero price is not included). ``wacc_shocks`` is
    supplied explicitly so one independently generated matrix can be reused
    byte-for-byte across alternative gold engines.
    """

    gold = np.asarray(gold_price_usd_per_oz, dtype=float)
    expected_gold_shape = (inputs.n_simulations, inputs.n_quarters)
    if gold.shape != expected_gold_shape:
        raise ValuationInputError(
            "external gold paths must have shape "
            f"{expected_gold_shape}, received {gold.shape}"
        )
    if not np.isfinite(gold).all() or np.any(gold <= 0.0):
        raise ValuationInputError(
            "external gold paths must be finite and strictly positive"
        )
    wacc_shocks = np.asarray(wacc_shocks, dtype=float)
    expected_wacc_shape = (inputs.n_simulations, inputs.n_years)
    if wacc_shocks.shape != expected_wacc_shape:
        raise ValuationInputError(
            "WACC shocks must have shape "
            f"{expected_wacc_shape}, received {wacc_shocks.shape}"
        )
    if not np.isfinite(wacc_shocks).all():
        raise ValuationInputError("WACC shocks must be finite")

    quarterly_margin = (
        (gold - inputs.cost_usd_per_oz[None, :])
        * inputs.production_koz[None, :]
        / 1000.0
    )
    annual_margin = quarterly_margin.reshape(
        inputs.n_simulations, inputs.n_years, 4
    ).sum(axis=2)

    growth = _trend(inputs.high_growth, inputs.stable_growth, inputs.n_years)
    roic = _trend(inputs.roic_high, inputs.roic_stable, inputs.n_years)
    reinvestment_rate = growth / roic
    after_tax_margin = annual_margin * (1.0 - inputs.tax_rate)
    fcff_proxy = after_tax_margin - np.maximum(after_tax_margin, 0.0) * reinvestment_rate

    wacc = np.empty((inputs.n_simulations, inputs.n_years + 1), dtype=float)
    wacc[:, 0] = inputs.wacc_0
    mean_reversion = np.exp(-inputs.wacc_reversion)
    conditional_std = inputs.wacc_volatility * np.sqrt(
        (1.0 - np.exp(-2.0 * inputs.wacc_reversion))
        / (2.0 * inputs.wacc_reversion)
    )
    for year in range(inputs.n_years):
        wacc[:, year + 1] = (
            inputs.wacc_long_run
            + (wacc[:, year] - inputs.wacc_long_run) * mean_reversion
            + conditional_std * wacc_shocks[:, year]
        )
    minimum_wacc = inputs.stable_growth + inputs.terminal_spread_floor
    annual_wacc = np.clip(wacc[:, 1:], minimum_wacc, 0.25)
    discount_factors = 1.0 / np.cumprod(1.0 + annual_wacc, axis=1)
    pv_explicit = np.sum(fcff_proxy * discount_factors, axis=1)

    if terminal_policy not in {"legacy_positive", "signed", "none"}:
        raise ValuationInputError("unknown terminal_policy")
    final_margin = after_tax_margin[:, -1]
    if terminal_policy == "legacy_positive":
        final_margin = np.maximum(final_margin, 0.0)
    elif terminal_policy == "none":
        final_margin = np.zeros_like(final_margin)
    terminal_after_tax = final_margin * (1.0 + inputs.stable_growth)
    terminal_reinvestment = inputs.stable_growth / inputs.roic_stable
    terminal_fcff = terminal_after_tax * (1.0 - terminal_reinvestment)
    terminal_value = terminal_fcff / (annual_wacc[:, -1] - inputs.stable_growth)
    pv_terminal = terminal_value * discount_factors[:, -1]
    enterprise_value = pv_explicit + pv_terminal

    bridge_adjustment = (
        -inputs.debt_usd_mn
        + inputs.cash_usd_mn
        - inputs.minority_interest_usd_mn
        + inputs.non_operating_assets_usd_mn
        - inputs.other_claims_usd_mn
    )
    equity_value = enterprise_value + bridge_adjustment
    value_per_share = equity_value / inputs.diluted_shares_mn

    arrays = (
        gold,
        annual_margin,
        fcff_proxy,
        annual_wacc,
        pv_explicit,
        pv_terminal,
        enterprise_value,
        equity_value,
        value_per_share,
    )
    if not all(np.isfinite(array).all() for array in arrays):
        raise FloatingPointError("valuation produced non-finite values")

    return ValuationResult(
        gold_price_usd_per_oz=gold,
        annual_component_margin_usd_mn=annual_margin,
        annual_fcff_proxy_usd_mn=fcff_proxy,
        annual_wacc=annual_wacc,
        pv_explicit_fcff_proxy_usd_mn=pv_explicit,
        pv_terminal_proxy_usd_mn=pv_terminal,
        enterprise_value_proxy_usd_mn=enterprise_value,
        equity_value_proxy_usd_mn=equity_value,
        value_per_share_proxy_usd=value_per_share,
    )


def quantile_rows(result: ValuationResult) -> list[dict[str, float | int]]:
    """Return the publication table at the canonical percentile levels."""

    rows: list[dict[str, float | int]] = []
    for level in QUANTILE_LEVELS:
        rows.append(
            {
                "quantile_pct": level,
                "enterprise_value_proxy_usd_mn": float(
                    np.percentile(result.enterprise_value_proxy_usd_mn, level)
                ),
                "equity_value_proxy_usd_mn": float(
                    np.percentile(result.equity_value_proxy_usd_mn, level)
                ),
                "value_per_share_proxy_usd": float(
                    np.percentile(result.value_per_share_proxy_usd, level)
                ),
            }
        )
    return rows


def probability_value_exceeds_market(
    result: ValuationResult, observed_price_usd: float
) -> float:
    return float(np.mean(result.value_per_share_proxy_usd > observed_price_usd))
