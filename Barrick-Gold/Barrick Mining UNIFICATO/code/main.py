"""Single handoff entry point for the unified Barrick research project."""

from __future__ import annotations

from pathlib import Path
import sys


ROOT = Path(__file__).resolve().parent
SRC = ROOT / "src"
if str(SRC) not in sys.path:
    sys.path.insert(0, str(SRC))

from barrick_unified.cli import main  # noqa: E402
from barrick_unified.project import ProjectLayout, UnifiedWorkflow  # noqa: E402


__all__ = ["ProjectLayout", "UnifiedWorkflow", "main"]


if __name__ == "__main__":
    raise SystemExit(main())
