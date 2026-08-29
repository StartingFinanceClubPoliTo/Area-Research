"""Build a deterministic comparison report for the frozen Team 7 offline run.

This tool is intentionally read-only with respect to the original Team 7 tree.
It compares the legacy distributed output with a run produced from the frozen
source/input copies and writes one UTF-8 JSON evidence file inside UNIFICATO.
"""

from __future__ import annotations

import argparse
import hashlib
import json
from pathlib import Path
from typing import Any

import numpy as np
import pandas as pd
from PIL import Image


def sha256(path: Path) -> str:
    digest = hashlib.sha256()
    with path.open("rb") as handle:
        for chunk in iter(lambda: handle.read(1024 * 1024), b""):
            digest.update(chunk)
    return digest.hexdigest().upper()


def csv_comparison(original: Path, actual: Path) -> dict[str, Any]:
    try:
        left = pd.read_csv(original)
        right = pd.read_csv(actual)
    except Exception as exc:  # pragma: no cover - evidence path
        return {"read_status": "ERROR", "error": f"{type(exc).__name__}: {exc}"}

    result: dict[str, Any] = {
        "read_status": "PASS",
        "original_shape": list(left.shape),
        "actual_shape": list(right.shape),
        "columns_equal": list(left.columns) == list(right.columns),
        "columns": list(left.columns),
    }
    if left.shape != right.shape or list(left.columns) != list(right.columns):
        result["numeric_comparison"] = "NOT_COMPARABLE"
        return result

    nonnumeric_equal = True
    numeric_columns: list[str] = []
    per_column: dict[str, Any] = {}
    overall_max = 0.0
    overall_max_column: str | None = None
    all_close_1e12 = True

    for column in left.columns:
        left_num = pd.to_numeric(left[column], errors="coerce")
        right_num = pd.to_numeric(right[column], errors="coerce")
        numeric_mask = left_num.notna() | right_num.notna()
        if numeric_mask.any():
            numeric_columns.append(str(column))
            left_values = left_num.to_numpy(dtype=float)
            right_values = right_num.to_numpy(dtype=float)
            same_nan = np.array_equal(np.isnan(left_values), np.isnan(right_values))
            finite_mask = np.isfinite(left_values) & np.isfinite(right_values)
            if finite_mask.any():
                differences = np.abs(left_values[finite_mask] - right_values[finite_mask])
                maximum = float(np.max(differences))
                mean = float(np.mean(differences))
            else:
                maximum = 0.0
                mean = 0.0
            close = bool(same_nan and np.allclose(left_values, right_values, rtol=1e-12, atol=1e-12, equal_nan=True))
            all_close_1e12 = all_close_1e12 and close
            if maximum > overall_max:
                overall_max = maximum
                overall_max_column = str(column)
            per_column[str(column)] = {
                "max_absolute_difference": maximum,
                "mean_absolute_difference": mean,
                "allclose_rtol_atol_1e-12": close,
            }
        else:
            equal = left[column].fillna("<NA>").astype(str).equals(
                right[column].fillna("<NA>").astype(str)
            )
            nonnumeric_equal = nonnumeric_equal and equal

    result.update(
        {
            "numeric_columns": numeric_columns,
            "nonnumeric_columns_equal": nonnumeric_equal,
            "numeric_allclose_rtol_atol_1e-12": all_close_1e12,
            "overall_max_absolute_difference": overall_max,
            "overall_max_difference_column": overall_max_column,
            "per_numeric_column": per_column,
        }
    )
    return result


def png_comparison(original: Path, actual: Path) -> dict[str, Any]:
    with Image.open(original) as left_image, Image.open(actual) as right_image:
        left = np.asarray(left_image.convert("RGBA"), dtype=np.int16)
        right = np.asarray(right_image.convert("RGBA"), dtype=np.int16)
        result: dict[str, Any] = {
            "original_dimensions": [left_image.width, left_image.height],
            "actual_dimensions": [right_image.width, right_image.height],
            "original_mode": left_image.mode,
            "actual_mode": right_image.mode,
            "decoded_shape_equal": left.shape == right.shape,
        }
        if left.shape == right.shape:
            difference = np.abs(left - right)
            result.update(
                {
                    "decoded_pixels_equal": bool(np.array_equal(left, right)),
                    "mean_absolute_channel_difference": float(np.mean(difference)),
                    "max_absolute_channel_difference": int(np.max(difference)),
                    "changed_channel_fraction": float(np.count_nonzero(difference) / difference.size),
                }
            )
        else:
            result["decoded_pixels_equal"] = False
            result["pixel_difference"] = "NOT_COMPARABLE_DIFFERENT_SHAPE"
        return result


def text_comparison(original: Path, actual: Path) -> dict[str, Any]:
    left = original.read_text(encoding="utf-8", errors="replace")
    right = actual.read_text(encoding="utf-8", errors="replace")
    left_lines = left.splitlines()
    right_lines = right.splitlines()
    return {
        "text_equal": left == right,
        "line_count_original": len(left_lines),
        "line_count_actual": len(right_lines),
        "differing_line_positions": sum(
            a != b for a, b in zip(left_lines, right_lines)
        )
        + abs(len(left_lines) - len(right_lines)),
    }


def main() -> None:
    parser = argparse.ArgumentParser()
    parser.add_argument("--original-root", type=Path, required=True)
    parser.add_argument("--run-root", type=Path, required=True)
    parser.add_argument("--output", type=Path, required=True)
    args = parser.parse_args()

    original_root = args.original_root.resolve()
    actual_root = (args.run_root / "work" / "output").resolve()
    original_files = {
        path.relative_to(original_root).as_posix(): path
        for path in original_root.rglob("*")
        if path.is_file()
    }
    actual_files = {
        path.relative_to(actual_root).as_posix(): path
        for path in actual_root.rglob("*")
        if path.is_file()
    }

    artifacts: list[dict[str, Any]] = []
    for relative in sorted(set(original_files) | set(actual_files)):
        original = original_files.get(relative)
        actual = actual_files.get(relative)
        item: dict[str, Any] = {
            "path": relative,
            "generated_artifact": relative != "README.md",
            "original_present": original is not None,
            "actual_present": actual is not None,
        }
        if original is not None:
            item["original_bytes"] = original.stat().st_size
            item["original_sha256"] = sha256(original)
        if actual is not None:
            item["actual_bytes"] = actual.stat().st_size
            item["actual_sha256"] = sha256(actual)

        if original is None:
            item["status"] = "UNEXPECTED_ACTUAL"
        elif actual is None:
            item["status"] = "MISSING_ACTUAL"
        elif item["original_sha256"] == item["actual_sha256"]:
            item["status"] = "BYTE_IDENTICAL"
        else:
            item["status"] = "DIFFERENT"
            suffix = original.suffix.lower()
            if suffix == ".csv":
                item["comparison"] = csv_comparison(original, actual)
            elif suffix == ".png":
                item["comparison"] = png_comparison(original, actual)
            elif suffix in {".tex", ".md", ".txt"}:
                item["comparison"] = text_comparison(original, actual)
        artifacts.append(item)

    generated = [item for item in artifacts if item["generated_artifact"]]
    statuses = sorted({item["status"] for item in artifacts})
    summary = {
        "original_file_count": len(original_files),
        "actual_file_count": len(actual_files),
        "expected_generated_artifact_count": sum(
            relative != "README.md" for relative in original_files
        ),
        "actual_generated_artifact_count": sum(
            relative != "README.md" for relative in actual_files
        ),
        "all_generated_artifacts_present": all(
            item["actual_present"] for item in generated if item["original_present"]
        ),
        "status_counts_all_files": {
            status: sum(item["status"] == status for item in artifacts)
            for status in statuses
        },
        "status_counts_generated_artifacts": {
            status: sum(item["status"] == status for item in generated)
            for status in statuses
        },
        "byte_identical_by_extension": {},
    }
    for suffix in sorted({Path(item["path"]).suffix.lower() for item in generated}):
        subset = [item for item in generated if Path(item["path"]).suffix.lower() == suffix]
        summary["byte_identical_by_extension"][suffix] = {
            "byte_identical": sum(item["status"] == "BYTE_IDENTICAL" for item in subset),
            "total": len(subset),
        }

    report = {
        "schema_version": "1.0",
        "team": 7,
        "run_id": args.run_root.name,
        "original_root": original_root.as_posix(),
        "actual_root": actual_root.as_posix(),
        "summary": summary,
        "artifacts": artifacts,
    }
    args.output.parent.mkdir(parents=True, exist_ok=True)
    args.output.write_text(
        json.dumps(report, indent=2, ensure_ascii=False) + "\n", encoding="utf-8"
    )


if __name__ == "__main__":
    main()
