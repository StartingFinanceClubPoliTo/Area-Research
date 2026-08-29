"""Build a redistribution-safe current NSS manifest from a private LSE snapshot."""

from __future__ import annotations

import argparse
import hashlib
import json
from pathlib import Path
import sys


ROOT = Path(__file__).resolve().parents[1]
TEAM8 = ROOT / "parity" / "sources" / "team-8"
sys.path.insert(0, str(TEAM8))

from lse_dataset import NSS_TREASURY_TENORS, normalise_lse_yield_curve  # noqa: E402
from nss_curve import PARAMETER_NAMES  # noqa: E402


def sha256(path: Path) -> str:
    return hashlib.sha256(path.read_bytes()).hexdigest().upper()


def main() -> None:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--raw-input", type=Path, required=True)
    parser.add_argument("--output", type=Path, required=True)
    args = parser.parse_args()
    raw_path = args.raw_input.resolve()
    payload = json.loads(raw_path.read_text(encoding="utf-8"))
    curve = normalise_lse_yield_curve(payload["rows"]["usd_treasury_yields"])
    parameters = {name: float(curve[name].iloc[0]) for name in PARAMETER_NAMES}
    symbols = curve["symbol"].astype(str).tolist()
    manifest = {
        "schema_version": "1.0",
        "method": "Nelson-Siegel-Svensson nonlinear least squares",
        "source": "London Strategic Edge /ref/bond_yields",
        "source_run_id": str(payload["run_id"]),
        "source_raw_sha256": sha256(raw_path),
        "curve_date": str(curve["date"].iloc[0]),
        "requested_symbols": list(NSS_TREASURY_TENORS),
        "available_symbols": symbols,
        "missing_symbols_on_curve_date": sorted(set(NSS_TREASURY_TENORS) - set(symbols)),
        "observations": int(len(curve)),
        "rate_definition": "continuous proxy log(1 + Treasury par yield)",
        "parameters": parameters,
        "rmse_bp": float((curve["nss_residual_bp"].pow(2).mean()) ** 0.5),
        "max_abs_error_bp": float(curve["nss_residual_bp"].abs().max()),
        "curve_points": [
            {
                "symbol": str(row.symbol),
                "maturity_years": float(row.maturity_years),
                "par_yield_pct": float(row.par_yield_pct),
                "observed_continuous_rate": float(row.observed_continuous_rate),
                "fitted_continuous_rate": float(row.continuous_rate),
                "residual_bp": float(row.nss_residual_bp),
            }
            for row in curve.itertuples(index=False)
        ],
        "use": "common risk-free zero curve for Team 8 calibration and unified simulations",
        "forward_rate_policy": "integral-preserving deterministic forwards",
        "legacy_team4_nss_used": False,
        "redistribution": "aggregate curve observations and fitted parameters only; row history remains local-only",
    }
    output = args.output.resolve()
    output.parent.mkdir(parents=True, exist_ok=True)
    output.write_text(json.dumps(manifest, indent=2), encoding="utf-8")
    print(json.dumps({
        "output": str(output),
        "curve_date": manifest["curve_date"],
        "observations": manifest["observations"],
        "missing": manifest["missing_symbols_on_curve_date"],
        "rmse_bp": manifest["rmse_bp"],
    }, indent=2))


if __name__ == "__main__":
    main()
