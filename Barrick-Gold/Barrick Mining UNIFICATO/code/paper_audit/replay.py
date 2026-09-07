from pathlib import Path
import sys,json
R=Path(__file__).resolve().parent;sys.path.insert(0,str(R/'src'))
from barrick_unified.multimodel_valuation import run_multimodel_valuation
c=json.loads((R/'config/multimodel_valuation_20260902_team8_refresh.json').read_text())
r=run_multimodel_valuation(R,c)
print([(m,x.valuation.value_per_share_proxy_usd.mean()) for m,x in r.models.items()])
