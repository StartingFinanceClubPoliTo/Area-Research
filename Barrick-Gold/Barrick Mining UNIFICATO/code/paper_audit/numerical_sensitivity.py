from pathlib import Path
import sys,json,copy
import numpy as np,pandas as pd
R=Path(__file__).resolve().parent;sys.path.insert(0,str(R/'src'))
from barrick_unified.multimodel_valuation import run_multimodel_valuation
from barrick_unified.valuation import simulate_valuation_from_gold_paths
c=json.loads((R/'config/paper_audited_20260907.json').read_text());rows=[]
for count,steps,offset in [(8192,260,0),(16384,260,0),(8192,520,0),(8192,260,1)]:
 d=copy.deepcopy(c);d['simulation']['n_simulations']=count;d['gold_price_layer']['path_grid']['fine_steps']=steps
 for k in d['simulation']['engine_seeds']:d['simulation']['engine_seeds'][k]+=offset
 d['simulation']['wacc_seed']+=offset;r=run_multimodel_valuation(R,d)
 for k,m in r.models.items():
  v=simulate_valuation_from_gold_paths(r.inputs,m.quarterly_gold_paths,r.wacc_shocks,'signed');x=v.enterprise_value_proxy_usd_mn/1000
  rows.append({'model':k,'paths':count,'steps':steps,'seed_offset':offset,'mean_bn':np.mean(x),'mc_mean_se_bn':np.std(x,ddof=1)/np.sqrt(len(x)),'median_bn':np.median(x),'p90_bn':np.quantile(x,.9)})
pd.DataFrame(rows).to_csv(R/'audit_outputs/numerical_sensitivity.csv',index=False);print(pd.DataFrame(rows).to_string(index=False))
