from pathlib import Path
import json,sys
import numpy as np,pandas as pd
from scipy.special import ndtr
R=Path(__file__).resolve().parent;sys.path.insert(0,str(R/'src'))
from barrick_unified.multimodel_valuation import run_multimodel_valuation
from barrick_unified.valuation import simulate_valuation_from_gold_paths,ValuationInputError
c=json.loads((R/'config/multimodel_valuation_20260902_team8_refresh.json').read_text());r=run_multimodel_valuation(R,c);out=[]
for k,m in r.models.items():
 v=m.valuation; gold=m.quarterly_gold_paths
 signed=simulate_valuation_from_gold_paths(r.inputs,gold,r.wacc_shocks,terminal_policy='signed')
 finite=simulate_valuation_from_gold_paths(r.inputs,gold,r.wacc_shocks,terminal_policy='none')
 assert np.allclose(finite.pv_terminal_proxy_usd_mn,0)
 assert np.allclose(finite.enterprise_value_proxy_usd_mn,v.pv_explicit_fcff_proxy_usd_mn)
 assert np.all(signed.enterprise_value_proxy_usd_mn<=v.enterprise_value_proxy_usd_mn+1e-8)
 assert np.allclose(v.annual_component_margin_usd_mn,(((gold-r.inputs.cost_usd_per_oz)*r.inputs.production_koz/1000).reshape(-1,5,4).sum(2)))
 out.append({'model':k,'legacy_median_bn':np.median(v.enterprise_value_proxy_usd_mn)/1000,'signed_median_bn':np.median(signed.enterprise_value_proxy_usd_mn)/1000,'five_year_median_bn':np.median(finite.enterprise_value_proxy_usd_mn)/1000,'negative_final_margin_paths':float(np.mean(v.annual_component_margin_usd_mn[:,-1]<0)),'terminal_positive_floor_mean_uplift_bn':np.mean(v.enterprise_value_proxy_usd_mn-signed.enterprise_value_proxy_usd_mn)/1000,'clipped_wacc_fraction':float(np.mean((v.annual_wacc<=.035)|(v.annual_wacc>=.25)))})
pd.DataFrame(out).to_csv(R/'audit_outputs/terminal_audit.csv',index=False)
try:simulate_valuation_from_gold_paths(r.inputs,gold,r.wacc_shocks,terminal_policy='invalid')
except ValuationInputError:pass
else:raise AssertionError('invalid policy accepted')
# Independent CRR early exercise check for positive-rate non-distributing GLD calls.
f=pd.read_csv(R/'audit_inputs/outputs/calibrations/CC/2026-09-02/calibration_surface.csv');diff=[]
for row in f.itertuples():
 n=800;dt=row.T/n;u=np.exp(row.implied_vol*np.sqrt(dt));d=1/u;p=(np.exp(row.rate*dt)-d)/(u-d);assert 0<=p<=1
 st=row.spot*d**np.arange(n,-1,-1)*u**np.arange(n+1);a=np.maximum(st-row.K,0);e=a.copy();disc=np.exp(-row.rate*dt)
 for j in range(n-1,-1,-1):
  a=disc*((1-p)*a[:-1]+p*a[1:]);e=disc*((1-p)*e[:-1]+p*e[1:]);st=row.spot*d**np.arange(j,-1,-1)*u**np.arange(j+1);a=np.maximum(a,st-row.K)
 diff.append(float(a[0]-e[0]))
assert max(diff)<1e-8
(R/'audit_outputs/implementation_tests.json').write_text(json.dumps({'quarterly_to_annual':'pass','terminal_policy_tests':'pass','american_check':{'contracts':len(diff),'tree_steps':800,'q':0,'max_american_minus_european':max(diff),'scope':'CRR at observed IV, non-distributing calls and positive rates; not a calibrated stochastic-volatility American tree'},'rate_min':f.rate.min()},indent=2));print(out);print('implementation tests pass')
