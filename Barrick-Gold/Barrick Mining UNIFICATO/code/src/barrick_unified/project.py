"""Handoff-friendly orchestration for the unified Barrick research project.

Numerical logic remains in the focused domain modules.  This layer owns only
project discovery, structural validation and dispatch to the established
entry points, mirroring the thin ``main.py`` pattern used by Team 8.
"""

from __future__ import annotations

from dataclasses import dataclass
import csv
import json
import os
from pathlib import Path
import subprocess
import sys
from typing import Any, Sequence


@dataclass(frozen=True)
class ProjectLayout:
    """Canonical, named paths used by people and automation."""

    root: Path

    @property
    def source(self) -> Path:
        return self.root / "src" / "barrick_unified"

    @property
    def data(self) -> Path:
        return self.root / "data"

    @property
    def figures(self) -> Path:
        return self.root / "figures"

    @property
    def tables(self) -> Path:
        return self.root / "tables"

    @property
    def outputs(self) -> Path:
        return self.root / "outputs"

    @property
    def tests(self) -> Path:
        return self.root / "tests"

    @property
    def table_index(self) -> Path:
        return self.tables / "publication_table_index.csv"

    def required_directories(self) -> dict[str, Path]:
        return {
            "source": self.source,
            "data": self.data,
            "figures": self.figures,
            "tables": self.tables,
            "outputs": self.outputs,
            "tests": self.tests,
        }


class UnifiedWorkflow:
    """Small application facade for status checks and existing runners."""

    ENTRY_POINTS = {
        "all-figures": "run_all_thesis_figures.py",
        "empirical-figures": "run_empirical_thesis_figures.py",
        "valuation": "run_multimodel_valuation.py",
        "provisional-valuation": "run_provisional_valuation.py",
        "refactored-figures": "run_refactored_thesis_figures.py",
        "snapshot": "run_research_snapshot.py",
    }

    def __init__(self, root: Path | None = None) -> None:
        discovered = root or Path(__file__).resolve().parents[2]
        self.layout = ProjectLayout(discovered.resolve())
        if not (self.layout.root / "pyproject.toml").is_file():
            raise ValueError(f"Not a Barrick unified project root: {self.layout.root}")

    def status(self) -> dict[str, Any]:
        """Return a deterministic, read-only handoff summary."""

        directories = self.layout.required_directories()
        missing = [name for name, path in directories.items() if not path.is_dir()]
        class_modules = sorted(
            path.relative_to(self.layout.root).as_posix()
            for path in (self.layout.source / "refactored").rglob("*.py")
            if path.name != "__init__.py"
        )
        notebooks = sorted(path.name for path in self.layout.root.glob("*.ipynb"))
        indexed_tables = 0
        broken_table_links: list[str] = []
        if self.layout.table_index.is_file():
            with self.layout.table_index.open(encoding="utf-8", newline="") as stream:
                rows = list(csv.DictReader(stream))
            indexed_tables = len(rows)
            broken_table_links = [
                row["authoritative_path"]
                for row in rows
                if not (self.layout.root / row["authoritative_path"]).is_file()
            ]
        artifact_patterns = ("*.pyc", "*.aux", "*.log", "*.out", "*.toc")
        operational_artifacts = sorted(
            path.relative_to(self.layout.root).as_posix()
            for pattern in artifact_patterns
            for path in self.layout.root.rglob(pattern)
            if "outputs" not in path.parts
        )
        ready = not missing and not broken_table_links and not operational_artifacts
        return {
            "status": "READY" if ready else "ATTENTION_REQUIRED",
            "root": str(self.layout.root),
            "directories": {name: path.is_dir() for name, path in directories.items()},
            "class_modules": class_modules,
            "curated_notebooks": notebooks,
            "indexed_publication_tables": indexed_tables,
            "broken_table_links": broken_table_links,
            "operational_artifacts": operational_artifacts,
        }

    def run_tests(self, extra_args: Sequence[str] = ()) -> int:
        """Run the offline suite without leaving Python or pytest caches."""

        environment = os.environ.copy()
        environment["PYTHONDONTWRITEBYTECODE"] = "1"
        command = [
            sys.executable,
            "-m",
            "pytest",
            "tests",
            "-p",
            "no:cacheprovider",
            *extra_args,
        ]
        return subprocess.run(
            command,
            cwd=self.layout.root,
            env=environment,
            check=False,
        ).returncode

    def run_entry_point(self, name: str, arguments: Sequence[str] = ()) -> int:
        """Dispatch to a named, existing runner without duplicating its logic."""

        try:
            script = self.ENTRY_POINTS[name]
        except KeyError as exc:
            allowed = ", ".join(sorted(self.ENTRY_POINTS))
            raise ValueError(f"Unknown entry point {name!r}; choose one of: {allowed}") from exc
        environment = os.environ.copy()
        environment["PYTHONDONTWRITEBYTECODE"] = "1"
        return subprocess.run(
            [sys.executable, script, *arguments],
            cwd=self.layout.root,
            env=environment,
            check=False,
        ).returncode

    def print_status(self) -> None:
        print(json.dumps(self.status(), indent=2, ensure_ascii=False))
