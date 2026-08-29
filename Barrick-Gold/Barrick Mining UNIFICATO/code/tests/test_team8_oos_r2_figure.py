from __future__ import annotations

from pathlib import Path

import matplotlib.image as mpimg

from tools.build_team8_oos_r2_figure import (
    DEFAULT_DESIGN,
    DEFAULT_METRICS,
    build_figure,
    load_validated_metrics,
)


def test_frozen_online_r2_aggregates_pass_identity_and_design_checks() -> None:
    rows, design = load_validated_metrics(DEFAULT_METRICS, DEFAULT_DESIGN)
    values = {
        (str(row["model"]), str(row["benchmark"])): float(row["r2_oos"])
        for row in rows
    }

    assert len(rows) == 8
    assert design["target_dates"] == 124
    assert design["common_node_date_observations"] == 3052
    assert values[("Black-Scholes", "Prevailing mean")] == 0.9133483883512143
    assert values[("Full Bates-Hawkes", "Random walk")] == -0.0781765109267134


def test_publication_figure_is_a_nonempty_png(tmp_path: Path) -> None:
    output = tmp_path / "online_oos_r2_by_benchmark.png"
    built = build_figure(DEFAULT_METRICS, DEFAULT_DESIGN, output)

    assert built == output
    assert output.read_bytes().startswith(b"\x89PNG\r\n\x1a\n")
    assert output.stat().st_size > 50_000
    image = mpimg.imread(output)
    assert image.shape[0] >= 1000
    assert image.shape[1] >= 1800
