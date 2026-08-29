from __future__ import annotations

import csv
import json
from pathlib import Path

from barrick_unified.project import UnifiedWorkflow


ROOT = Path(__file__).resolve().parents[1]


def test_handoff_status_is_ready() -> None:
    status = UnifiedWorkflow(ROOT).status()
    assert status["status"] == "READY"
    assert all(status["directories"].values())
    assert status["curated_notebooks"] == ["Main.ipynb"]
    assert status["indexed_publication_tables"] == 7
    assert status["broken_table_links"] == []
    assert status["operational_artifacts"] == []


def test_curated_notebook_is_thin_and_valid() -> None:
    notebook = json.loads((ROOT / "Main.ipynb").read_text(encoding="utf-8"))
    assert notebook["nbformat"] == 4
    assert len(notebook["cells"]) == 4
    code = "\n".join(
        "".join(cell.get("source", []))
        for cell in notebook["cells"]
        if cell["cell_type"] == "code"
    )
    assert "from main import UnifiedWorkflow" in code
    assert "workflow.status()" in code


def test_publication_table_index_points_to_real_outputs() -> None:
    index_path = ROOT / "tables" / "publication_table_index.csv"
    with index_path.open(encoding="utf-8", newline="") as stream:
        rows = list(csv.DictReader(stream))
    assert len(rows) == 7
    assert len({row["table_id"] for row in rows}) == len(rows)
    for row in rows:
        assert (ROOT / row["authoritative_path"]).is_file(), row
