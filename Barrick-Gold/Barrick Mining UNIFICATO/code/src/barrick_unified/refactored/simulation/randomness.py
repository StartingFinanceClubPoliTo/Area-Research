"""Independent, auditable random streams."""

from __future__ import annotations

from dataclasses import dataclass

import numpy as np

from ..domain.provenance import array_sha256


@dataclass(frozen=True)
class RandomStreams:
    wacc_shocks: np.ndarray
    wacc_seed: int

    @classmethod
    def build(cls, *, n_paths: int, n_years: int, wacc_seed: int) -> "RandomStreams":
        if n_paths < 1 or n_years < 1 or wacc_seed < 0:
            raise ValueError("random stream dimensions and seed must be valid")
        shocks = np.random.default_rng(wacc_seed).standard_normal((n_paths, n_years))
        return cls(shocks, wacc_seed)

    @property
    def wacc_sha256(self) -> str:
        return array_sha256(self.wacc_shocks)
