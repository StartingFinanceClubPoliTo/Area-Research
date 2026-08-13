"""Default end-to-end LSE workflow for the Team 8 research project.

The key gate is deliberately evaluated before any network request or file
write. If ``LSE_API_KEY`` cannot be found on the current computer, the function
returns ``False`` and leaves every project artifact unchanged.
"""

import argparse
import os
from datetime import datetime, timedelta, timezone
from pathlib import Path


ROOT = Path(__file__).resolve().parent


def find_local_lse_key():
    """Read the process or Windows user environment without printing the key."""
    key = os.environ.get("LSE_API_KEY", "").strip()
    if key:
        return key
    if os.name == "nt":
        try:
            import winreg

            with winreg.OpenKey(winreg.HKEY_CURRENT_USER, "Environment") as handle:
                value, _ = winreg.QueryValueEx(handle, "LSE_API_KEY")
            key = str(value).strip()
        except (FileNotFoundError, OSError):
            key = ""
    return key


def _authorise_run(key, confirm):
    if not key:
        if confirm:
            answer = input(
                "LSE_API_KEY was not detected on this computer. "
                "Have you already configured it? [y/N] "
            ).strip().lower()
            if answer in {"y", "yes"}:
                key = find_local_lse_key()
        if not key:
            print(
                "No local LSE key is available. The pipeline stopped before "
                "any download or file modification. Configure LSE_API_KEY and rerun."
            )
            return ""
    if confirm:
        answer = input(
            "A local LSE key is available. Run the complete project workflow? [y/N] "
        ).strip().lower()
        if answer not in {"y", "yes"}:
            print("Run cancelled before any download or file modification.")
            return ""
    return key


def run_project(confirm=True, data_only=False):
    """Fetch LSE inputs, then rebuild current and historical diagnostics."""
    key = _authorise_run(find_local_lse_key(), confirm)
    if not key:
        return False
    os.environ["LSE_API_KEY"] = key

    from lse_dataset import (
        fetch_lse_calls,
        fetch_lse_history,
        fetch_lse_historical_yields,
        fetch_lse_option_history,
        fetch_lse_yields,
        write_local_dataset,
        write_local_historical_inputs,
    )

    print("[1/6] Fetching current and historical GLD/LSE inputs...")
    option_rows = fetch_lse_calls(max_dte=1000, limit=5000)
    yield_rows = fetch_lse_yields(lookback_days=120)
    history_rows = fetch_lse_history(start="2021-01-01")
    today = datetime.now(timezone.utc).date()
    historical_start = (today - timedelta(days=370)).isoformat()
    historical_rate_start = (today - timedelta(days=400)).isoformat()
    historical_end = today.isoformat()
    historical_option_path = fetch_lse_option_history(
        historical_start,
        historical_end,
        ROOT / "Data" / "lse_local",
    )
    historical_yield_rows = fetch_lse_historical_yields(
        historical_rate_start, historical_end
    )
    audit_path, audit = write_local_dataset(
        option_rows, yield_rows, history_rows, ROOT / "Data" / "lse_local"
    )
    _, historical_audit_path, historical_audit = write_local_historical_inputs(
        historical_yield_rows, ROOT / "Data" / "lse_local"
    )
    print(
        f"Local-only dataset ready: {audit['n_rows_chebyshev']} options, "
        f"{audit['n_treasury_tenors']} Treasury tenors, "
        f"{audit['history_observations']} daily returns; historical options at "
        f"{historical_option_path} and {historical_audit['complete_curve_dates']} "
        f"complete historical curves ({audit_path}; {historical_audit_path})."
    )
    if data_only:
        return True

    from tools.rebuild_lse_benchmarks import main as rebuild_baselines
    from tools.rebuild_exact_hawkes_outputs import main as rebuild_full_hawkes
    from tools.rebuild_online_validation import main as rebuild_online_validation
    from tools.rebuild_oos_validation import main as rebuild_oos_validation
    from tools.rebuild_path_outputs import main as rebuild_paths

    print("[2/6] Calibrating current Black-Scholes, Heston, and Bates...")
    rebuild_baselines()
    print("[3/6] Calibrating the current full affine Bates-Hawkes model...")
    rebuild_full_hawkes()
    print("[4/6] Running the primary rolling t-to-t+1 online validation in parallel...")
    rebuild_online_validation()
    print("[5/6] Running the frozen-parameter six-month stress test...")
    rebuild_oos_validation()
    print("[6/6] Rebuilding Monte Carlo paths and the 0--100 percentile table...")
    rebuild_paths()
    print("Complete LSE workflow finished successfully.")
    return True


def parse_args():
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument(
        "--yes", action="store_true",
        help="run without the interactive confirmation (the key is still mandatory)",
    )
    parser.add_argument(
        "--data-only", action="store_true",
        help="refresh local LSE inputs without recalibrating models",
    )
    return parser.parse_args()


if __name__ == "__main__":
    arguments = parse_args()
    run_project(confirm=not arguments.yes, data_only=arguments.data_only)
