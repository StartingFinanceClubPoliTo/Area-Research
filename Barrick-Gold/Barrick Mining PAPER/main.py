"""Run the audited Barrick operating-proxy experiment."""
from pathlib import Path
import subprocess
import sys
if __name__ == "__main__":
    raise SystemExit(subprocess.call([sys.executable, str(Path(__file__).resolve().parent / "code" / "main.py"), *sys.argv[1:]]))
