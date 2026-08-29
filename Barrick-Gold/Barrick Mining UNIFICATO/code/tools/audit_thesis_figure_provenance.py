"""Audit every analytical image used by the unified thesis.

The audit proves that the Overleaf asset is byte-identical to a generated
output inside this branch and that the declared generator and input manifests
exist.  Editorial assets (SF logo and cover PDF) are reported separately and
are not treated as scientific figures.
"""

from __future__ import annotations

import argparse
import csv
import hashlib
import json
import re
from collections import Counter
from datetime import datetime, timezone
from pathlib import Path


ROOT = Path(__file__).resolve().parents[1]
UNIFIED = ROOT.parent
OVERLEAF = UNIFIED / "Overleaf"
IMAGE_RE = re.compile(r"\\includegraphics(?:\[[^]]*\])?\{([^}]+)\}")


def sha256(path: Path) -> str:
    return hashlib.sha256(path.read_bytes()).hexdigest().upper()


def rel(path: Path) -> str:
    try:
        return str(path.relative_to(ROOT)).replace("\\", "/")
    except ValueError:
        return "../" + str(path.relative_to(UNIFIED)).replace("\\", "/")


def referenced_images() -> tuple[dict[str, list[str]], list[str]]:
    references: dict[str, list[str]] = {}
    for tex in sorted(OVERLEAF.rglob("*.tex")):
        for image in IMAGE_RE.findall(tex.read_text(encoding="utf-8")):
            references.setdefault(image, []).append(str(tex.relative_to(OVERLEAF)).replace("\\", "/"))
    editorial = [name for name in references if name == "img/Logo_SFClub_Polito.png"]
    analytical = {name: files for name, files in references.items() if name not in editorial}
    return analytical, editorial


def catalog(figure_run_id: str, market_run_id: str) -> dict[str, dict[str, object]]:
    run = ROOT / "outputs" / "thesis" / figure_run_id
    market = ROOT / "figures" / "current" / market_run_id
    refactored = run / "figures"
    mapping: dict[str, dict[str, object]] = {}

    for output in sorted((run / "team7_empirical").glob("*.png")):
        mapping[f"img/team7_empirical/{output.name}"] = {
            "classification": "CURRENT",
            "output": output,
            "generators": [ROOT / "run_empirical_thesis_figures.py", ROOT / "src/barrick_unified/empirical_figures.py"],
            "inputs": [ROOT / "data/manifests/20260827T173500Z-codefig-v11-lse/empirical_figure_inputs_manifest.json"],
        }
    team8_manifest = json.loads((run / "team8_empirical/figure_manifest.json").read_text(encoding="utf-8"))
    team8_classes = {row["name"]: row["classification"] for row in team8_manifest["figures"]}
    for output in sorted((run / "team8_empirical").glob("*.png")):
        mapping[f"img/team8_empirical/{output.name}"] = {
            "classification": team8_classes[output.name],
            "output": output,
            "generators": [ROOT / "tools/build_team8_thesis_figures.py"],
            "inputs": [ROOT / f"data/manifests/{market_run_id}/gld_option_surface_manifest.json"],
        }
    team4_dir = run / "team4_operations"
    for name, classification in {
        "team4_mine_production_qoq.png": "CURRENT",
        "team4_mine_cost_qoq.png": "CURRENT",
        "team4_process_components_qoq.png": "CURRENT",
        "team4_operating_contract.png": "MIXED_CURRENT_FORECAST",
    }.items():
        mapping[f"img/team4_current/{name}"] = {
            "classification": classification,
            "output": team4_dir / name,
            "generators": [
                ROOT / "tools/build_team4_operating_figures.py",
                ROOT / "src/barrick_unified/team4_operating_figures.py",
            ],
            "inputs": [
                ROOT / "data/manifests/team4/barrick_operating_q1_q2_2026_manifest.json",
                team4_dir / "figure_manifest.json",
            ],
        }
    mapping.update(
        {
            "img/lse/lse_market_context_20260826.png": {
                "classification": "CURRENT",
                "output": market / "lse_market_context.png",
                "generators": [ROOT / "run_research_snapshot.py", ROOT / "src/barrick_unified/research_snapshot.py"],
                "inputs": [ROOT / f"data/manifests/{market_run_id}/run_manifest.json"],
            },
            "img/lse/gld_option_surface_20260826.png": {
                "classification": "CURRENT",
                "output": market / "gld_lse_option_surface_audit.png",
                "generators": [ROOT / "run_research_snapshot.py", ROOT / "src/barrick_unified/team8_option_audit.py"],
                "inputs": [ROOT / f"data/manifests/{market_run_id}/gld_option_surface_manifest.json"],
            },
            "img/diagnostics/online_oos_r2_by_benchmark.png": {
                "classification": "FROZEN_LEGACY",
                "output": ROOT / "outputs/figures/team-8/online_oos_r2_by_benchmark.png",
                "generators": [ROOT / "tools/build_team8_oos_r2_figure.py"],
                "inputs": [ROOT / "parity/sources/team-8/Data/online_validation_metrics.csv", ROOT / "parity/sources/team-8/Data/online_validation_design.json"],
            },
            "img/valuation/gold_path_bands_four_models.png": {
                "classification": "PROVISIONAL",
                "output": refactored / "fig_gold_bands.png",
                "generators": [ROOT / "run_refactored_thesis_figures.py", ROOT / "src/barrick_unified/refactored/reporting/figures.py"],
                "inputs": [run / "figure_manifest.json"],
            },
            "img/valuation/dcf_driver_schedule.png": {
                "classification": "PROVISIONAL",
                "output": refactored / "fig_dcf_drivers.png",
                "generators": [ROOT / "run_refactored_thesis_figures.py", ROOT / "src/barrick_unified/refactored/reporting/figures.py"],
                "inputs": [run / "figure_manifest.json"],
            },
            "img/valuation/wacc_assumption_paths.png": {
                "classification": "PROVISIONAL",
                "output": refactored / "fig_wacc_assumptions.png",
                "generators": [ROOT / "run_refactored_thesis_figures.py", ROOT / "src/barrick_unified/refactored/reporting/figures.py"],
                "inputs": [run / "figure_manifest.json"],
            },
            "img/valuation/terminal_value_mechanics.png": {
                "classification": "PROVISIONAL",
                "output": refactored / "fig_terminal_value.png",
                "generators": [ROOT / "run_refactored_thesis_figures.py", ROOT / "src/barrick_unified/refactored/reporting/figures.py"],
                "inputs": [run / "figure_manifest.json"],
            },
            "img/valuation/multimodel_value_distribution_comparison.png": {
                "classification": "PROVISIONAL",
                "output": refactored / "fig_val_multi.png",
                "generators": [ROOT / "run_refactored_thesis_figures.py", ROOT / "src/barrick_unified/refactored/reporting/figures.py"],
                "inputs": [run / "figure_manifest.json"],
            },
            "img/valuation/full_bates_hawkes_value_distribution.png": {
                "classification": "PROVISIONAL",
                "output": refactored / "fig_val_primary.png",
                "generators": [ROOT / "run_refactored_thesis_figures.py", ROOT / "src/barrick_unified/refactored/reporting/figures.py"],
                "inputs": [run / "figure_manifest.json"],
            },
        }
    )
    return mapping


def main() -> None:
    parser = argparse.ArgumentParser()
    parser.add_argument("--figure-run-id", required=True)
    parser.add_argument("--market-run-id", required=True)
    parser.add_argument("--output-dir", type=Path, required=True)
    parser.add_argument("--authoritative", type=Path)
    args = parser.parse_args()
    analytical, editorial = referenced_images()
    declared = catalog(args.figure_run_id, args.market_run_id)
    missing_catalog = sorted(set(analytical) - set(declared))
    extra_catalog = sorted(set(declared) - set(analytical))
    rows: list[dict[str, object]] = []
    for image_name, tex_files in sorted(analytical.items()):
        item = declared.get(image_name, {})
        article_path = OVERLEAF / image_name.removeprefix("img/")
        # Overleaf-relative image names already contain the img/ prefix.
        article_path = OVERLEAF / image_name
        output = Path(item["output"]) if item else Path()
        generators = [Path(path) for path in item.get("generators", [])]
        inputs = [Path(path) for path in item.get("inputs", [])]
        article_exists = article_path.is_file()
        output_exists = output.is_file()
        hash_match = article_exists and output_exists and sha256(article_path) == sha256(output)
        generator_ok = bool(generators) and all(path.is_file() for path in generators)
        inputs_ok = bool(inputs) and all(path.is_file() for path in inputs)
        rows.append(
            {
                "image": image_name,
                "classification": item.get("classification", "UNMAPPED"),
                "article_files": "; ".join(tex_files),
                "generated_output": rel(output) if item else "",
                "article_sha256": sha256(article_path) if article_exists else "",
                "output_sha256": sha256(output) if output_exists else "",
                "hash_match": hash_match,
                "generators": "; ".join(rel(path) for path in generators),
                "generator_files_exist": generator_ok,
                "inputs": "; ".join(rel(path) for path in inputs),
                "input_files_exist": inputs_ok,
                "status": "PASS" if hash_match and generator_ok and inputs_ok else "FAIL",
            }
        )
    counts = Counter(str(row["classification"]) for row in rows)
    failures = [row["image"] for row in rows if row["status"] != "PASS"]
    manifest = {
        "schema_version": "1.0",
        "generated_at_utc": datetime.now(timezone.utc).isoformat(),
        "status": "PASS" if not failures and not missing_catalog and not extra_catalog else "FAIL",
        "figure_run_id": args.figure_run_id,
        "market_run_id": args.market_run_id,
        "analytical_figure_count": len(rows),
        "classification_counts": dict(sorted(counts.items())),
        "editorial_assets_excluded": editorial + ["Copertina_research.pdf"],
        "missing_catalog_entries": missing_catalog,
        "unused_catalog_entries": extra_catalog,
        "failed_figures": failures,
        "figures": rows,
    }
    output_dir = args.output_dir.resolve()
    output_dir.mkdir(parents=True, exist_ok=True)
    json_path = output_dir / "thesis_figure_provenance.json"
    csv_path = output_dir / "thesis_figure_provenance.csv"
    json_path.write_text(json.dumps(manifest, indent=2), encoding="utf-8")
    with csv_path.open("w", newline="", encoding="utf-8") as handle:
        writer = csv.DictWriter(handle, fieldnames=list(rows[0]))
        writer.writeheader()
        writer.writerows(rows)
    if args.authoritative:
        authoritative = args.authoritative.resolve()
        authoritative.parent.mkdir(parents=True, exist_ok=True)
        authoritative.write_text(json.dumps(manifest, indent=2), encoding="utf-8")
    print(json.dumps({key: manifest[key] for key in ("status", "analytical_figure_count", "classification_counts", "failed_figures")}, indent=2))
    if manifest["status"] != "PASS":
        raise SystemExit(1)


if __name__ == "__main__":
    main()
