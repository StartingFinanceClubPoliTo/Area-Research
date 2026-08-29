"""Central SHA-256 and manifest helpers used by runs and figures."""

from __future__ import annotations

from dataclasses import asdict, dataclass
import hashlib
import json
from pathlib import Path
from typing import Any

import numpy as np


def file_sha256(path: Path) -> str:
    digest = hashlib.sha256()
    with path.open("rb") as handle:
        for chunk in iter(lambda: handle.read(1024 * 1024), b""):
            digest.update(chunk)
    return digest.hexdigest().upper()


def array_sha256(values: np.ndarray) -> str:
    array = np.ascontiguousarray(np.asarray(values, dtype="<f8"))
    return hashlib.sha256(array.tobytes()).hexdigest().upper()


@dataclass(frozen=True)
class ArtifactRecord:
    role: str
    path: str
    bytes: int
    sha256: str

    @classmethod
    def from_path(
        cls, role: str, path: Path, *, relative_to: Path
    ) -> "ArtifactRecord":
        resolved = path.resolve()
        return cls(
            role=role,
            path=resolved.relative_to(relative_to.resolve()).as_posix(),
            bytes=resolved.stat().st_size,
            sha256=file_sha256(resolved),
        )


class RunManifestBuilder:
    """Accumulate and atomically serialize one public run manifest."""

    def __init__(self, *, run_id: str, schema_version: str = "3.0") -> None:
        self._payload: dict[str, Any] = {
            "schema_version": schema_version,
            "run_id": run_id,
            "artifacts": [],
        }

    def set(self, key: str, value: Any) -> "RunManifestBuilder":
        if key in {"schema_version", "run_id", "artifacts"}:
            raise KeyError(f"reserved manifest key: {key}")
        self._payload[key] = value
        return self

    def add_artifact(self, record: ArtifactRecord) -> "RunManifestBuilder":
        self._payload["artifacts"].append(asdict(record))
        return self

    def build(self) -> dict[str, Any]:
        return json.loads(json.dumps(self._payload, sort_keys=True))

    def write(self, path: Path) -> dict[str, Any]:
        payload = self.build()
        path.parent.mkdir(parents=True, exist_ok=False)
        path.write_text(
            json.dumps(payload, indent=2, ensure_ascii=False) + "\n",
            encoding="utf-8",
        )
        return payload
