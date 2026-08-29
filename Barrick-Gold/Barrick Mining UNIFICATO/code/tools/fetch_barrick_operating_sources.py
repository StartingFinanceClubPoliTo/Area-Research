"""Download and hash-check Barrick Q1/Q2 2026 mine-statistics PDFs.

The public PDFs are verification inputs only. They are saved below the
Git-ignored raw-data tree and are never required to render publication figures,
which use the committed, source-manifested derived tables.
"""

from __future__ import annotations

import argparse
import hashlib
import json
from pathlib import Path
from urllib.request import Request, urlopen


ROOT = Path(__file__).resolve().parents[1]
MANIFEST = ROOT / "data/manifests/team4/barrick_operating_q1_q2_2026_manifest.json"


def main() -> None:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument(
        "--output-dir",
        type=Path,
        default=ROOT / "data/raw/barrick_public/q1_q2_2026",
    )
    args = parser.parse_args()
    payload = json.loads(MANIFEST.read_text(encoding="utf-8"))
    output_dir = args.output_dir.resolve()
    output_dir.mkdir(parents=True, exist_ok=True)
    for source in payload["sources"]:
        quarter = source["quarter"]
        target = output_dir / f"Barrick_{quarter}_Mine_Stats.pdf"
        request = Request(source["url"], headers={"User-Agent": "SF-Research-Provenance/1.0"})
        with urlopen(request, timeout=60) as response:  # nosec B310 - URLs are fixed in the reviewed manifest
            data = response.read()
        actual = hashlib.sha256(data).hexdigest().upper()
        expected = str(source["sha256"]).upper()
        if actual != expected:
            raise ValueError(f"SHA-256 mismatch for {quarter}: {actual} != {expected}")
        target.write_bytes(data)
        print(f"{quarter}: {target} ({actual})")


if __name__ == "__main__":
    main()
