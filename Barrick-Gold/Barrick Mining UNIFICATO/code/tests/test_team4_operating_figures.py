from __future__ import annotations

import json
from pathlib import Path

from barrick_unified.team4_operating_figures import FIGURE_NAMES, build_team4_operating_figures


ROOT = Path(__file__).resolve().parents[1]


def test_team4_current_figures_are_autonomous_and_separated(tmp_path: Path) -> None:
    output = tmp_path / "team4_operations"
    manifest = build_team4_operating_figures(
        mine_csv=ROOT / "data/processed/team4/barrick_mine_operating_q1_q2_2026.csv",
        totals_csv=ROOT / "data/processed/team4/barrick_total_gold_q1_q2_2026.csv",
        source_manifest=ROOT / "data/manifests/team4/barrick_operating_q1_q2_2026_manifest.json",
        valuation_config=ROOT / "config/provisional_valuation_20260827_team4_separated.json",
        output_dir=output,
        run_id="pytest-team4",
    )
    assert manifest["separation_policy"]["team4_price_simulation_used"] is False
    assert manifest["separation_policy"]["team4_illustrative_valuation_used"] is False
    assert len(manifest["artifacts"]) == 8
    for name in FIGURE_NAMES:
        assert (output / f"{name}.png").is_file()
        assert (output / f"{name}.pdf").is_file()
    saved = json.loads((output / "figure_manifest.json").read_text(encoding="utf-8"))
    assert saved["upstream_method_code"]["commit"] == "f0c77feff23944819f8bd63abe7b6244b348cc77"
