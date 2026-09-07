"""Replay audited Paper outputs; original dated fits are preserved."""
from pathlib import Path
import argparse,subprocess,sys,hashlib,json
R=Path(__file__).resolve().parent
p=argparse.ArgumentParser();p.add_argument('--recalibrate',action='store_true');a=p.parse_args()
for row in json.loads((R/'DATA_MANIFEST.json').read_text())['files']:
    assert hashlib.sha256((R/row['path']).read_bytes()).hexdigest()==row['sha256'],row['path']
scripts=['audit.py']
if a.recalibrate:scripts+=['refine_hawkes.py']
scripts+=['build_audited_results.py','test_implementation.py','numerical_sensitivity.py']
for script in scripts:subprocess.run([sys.executable,str(R/script)],check=True,cwd=R)
print('Audited operating-proxy outputs complete. This is not a corporate fair-value model.')
