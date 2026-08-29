"""Aggregate-only audit of the current GLD LSE option surface.

The transformation is delegated to the hash-pinned Team 8 implementation.
This module records the exact approved source hash before importing it,
preventing silent changes to either surface filtering or the updated LSE NSS
curve while G1.5 historical parity remains open.
"""

from __future__ import annotations

import hashlib
import importlib
import sys
from dataclasses import dataclass
from pathlib import Path
from typing import Any

import numpy as np
import pandas as pd


EXPECTED_LSE_DATASET_SHA256 = "F6C1352C43032B8BE6F14B1F2130929E5594B266FE5D1571CA682805C3C1459A"


def _sha256(path: Path) -> str:
    digest = hashlib.sha256()
    with path.open("rb") as handle:
        for chunk in iter(lambda: handle.read(1024 * 1024), b""):
            digest.update(chunk)
    return digest.hexdigest().upper()


def load_frozen_team8_module(team8_root: Path) -> Any:
    source = team8_root / "lse_dataset.py"
    actual = _sha256(source)
    if actual != EXPECTED_LSE_DATASET_SHA256:
        raise RuntimeError(
            f"Approved Team 8 lse_dataset.py hash mismatch: {actual}."
        )
    root_text = str(team8_root.resolve())
    sys.path.insert(0, root_text)
    try:
        existing = sys.modules.get("lse_dataset")
        if existing is not None:
            existing_path = Path(existing.__file__).resolve()
            if existing_path != source.resolve():
                del sys.modules["lse_dataset"]
        return importlib.import_module("lse_dataset")
    finally:
        if sys.path and sys.path[0] == root_text:
            sys.path.pop(0)


@dataclass(frozen=True)
class OptionSurfaceAudit:
    as_of_utc: str
    raw_call_rows: int
    normalized_call_rows: int
    eligible_rows: int
    sampled_nodes: int
    sampled_maturities: int
    sampled_strikes: int
    treasury_curve_date: str
    treasury_tenors: int
    team8_source_sha256: str = EXPECTED_LSE_DATASET_SHA256

    def as_dict(self) -> dict[str, Any]:
        return dict(self.__dict__)


def audit_current_surface(
    option_rows: list[dict[str, Any]],
    yield_rows: list[dict[str, Any]],
    team8_root: Path,
) -> tuple[OptionSurfaceAudit, pd.DataFrame, float]:
    """Return aggregate audit plus an in-memory sampled surface for plotting."""

    module = load_frozen_team8_module(team8_root)
    chain = module.normalise_lse_chain(option_rows)
    curve = module.normalise_lse_yield_curve(yield_rows)
    eligible, sample, spot = module.build_calibration_sample(chain, curve)
    audit = OptionSurfaceAudit(
        as_of_utc=str(chain.attrs["as_of_utc"]),
        raw_call_rows=int(len(option_rows)),
        normalized_call_rows=int(len(chain)),
        eligible_rows=int(len(eligible)),
        sampled_nodes=int(len(sample)),
        sampled_maturities=int(sample["T"].nunique()),
        sampled_strikes=int(sample["K"].nunique()),
        treasury_curve_date=str(curve["date"].iloc[0]),
        treasury_tenors=int(len(curve)),
    )
    if not np.isfinite(spot) or spot <= 0.0:
        raise ValueError("Team 8 returned an invalid GLD snapshot spot.")
    return audit, sample, float(spot)
