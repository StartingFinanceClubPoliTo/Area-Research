"""Fail-closed interpretation contracts."""

from __future__ import annotations

from dataclasses import dataclass
from typing import Any


EXPECTED_BRIDGE_LABEL = (
    "conditional transfer of GLD/Q distributional shape to gold USD/oz"
)


@dataclass(frozen=True)
class ConditionalBridgePolicy:
    label: str
    source_instrument: str
    destination_variable: str
    validated_physical_forecast: bool
    q_to_p_mapping: str
    gld_to_gold_level_conversion: str
    interpretation: str

    @classmethod
    def from_dict(cls, value: dict[str, Any]) -> "ConditionalBridgePolicy":
        policy = cls(
            label=str(value.get("label", "")),
            source_instrument=str(value.get("source_instrument", "")),
            destination_variable=str(value.get("destination_variable", "")),
            validated_physical_forecast=bool(
                value.get("validated_physical_forecast", True)
            ),
            q_to_p_mapping=str(value.get("q_to_p_mapping", "")),
            gld_to_gold_level_conversion=str(
                value.get("gld_to_gold_level_conversion", "")
            ),
            interpretation=str(value.get("interpretation", "")),
        )
        policy.validate()
        return policy

    def validate(self) -> None:
        if self.label != EXPECTED_BRIDGE_LABEL:
            raise ValueError("conditional bridge label is not the accepted contract")
        if self.validated_physical_forecast:
            raise ValueError("conditional bridge cannot be a physical forecast")
        if self.q_to_p_mapping != "NOT_VALIDATED":
            raise ValueError("Q-to-P mapping must remain NOT_VALIDATED")
        if not self.gld_to_gold_level_conversion.startswith("NOT_PERFORMED"):
            raise ValueError("GLD-to-gold level conversion must remain NOT_PERFORMED")
        if not self.source_instrument or not self.destination_variable:
            raise ValueError("conditional bridge endpoints are required")
        lowered = self.interpretation.lower()
        for forbidden in ("target price", "investment recommendation"):
            if forbidden not in lowered:
                raise ValueError(
                    "conditional bridge interpretation must preserve non-target caveats"
                )

    def as_dict(self) -> dict[str, Any]:
        return {
            "label": self.label,
            "source_instrument": self.source_instrument,
            "destination_variable": self.destination_variable,
            "validated_physical_forecast": self.validated_physical_forecast,
            "q_to_p_mapping": self.q_to_p_mapping,
            "gld_to_gold_level_conversion": self.gld_to_gold_level_conversion,
            "interpretation": self.interpretation,
        }
