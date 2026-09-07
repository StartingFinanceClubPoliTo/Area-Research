from pathlib import Path
import sys,json
import numpy as np,pandas as pd
import matplotlib
matplotlib.use('Agg')
import matplotlib.pyplot as plt
R=Path(__file__).resolve().parent;sys.path.insert(0,str(R/'src'))
from barrick_unified.multimodel_valuation import run_multimodel_valuation
from barrick_unified.valuation import simulate_valuation_from_gold_paths
from PIL import Image
def save_figure(fig, path):
    dimensions=json.loads((R/'figure_dimensions.json').read_text())
    if path.name in dimensions:
        w,h=dimensions[path.name]
        fig.set_size_inches(7,7*h/w)
        fig.tight_layout()
    fig.savefig(path,dpi=240)

O=R/'audit_outputs';F=O/'figures';labels=['Black-Scholes','Heston','Bates-Poisson','Bates-Hawkes'];summary=pd.read_csv(O/'calibration_summary_audited.csv').to_dict('records')

c=json.loads((R/'config/multimodel_valuation_20260902_team8_refresh.json').read_text());c['experiment']='AUDITED_OPERATING_PROXY_SENSITIVITY';c['gold_price_layer']['parameter_files']={key:str(Path('data/processed/team8/calibration_audited_20260907')/Path(v).name).replace('\\','/') for key,v in c['gold_price_layer']['parameter_files'].items()};(R/'config/paper_audited_20260907.json').write_text(json.dumps(c,indent=2))
run=run_multimodel_valuation(R,c);rows=[];samples={};tc=[]
for key,model in run.models.items():
 legacy=model.valuation;signed=simulate_valuation_from_gold_paths(run.inputs,model.quarterly_gold_paths,run.wacc_shocks,'signed');finite=simulate_valuation_from_gold_paths(run.inputs,model.quarterly_gold_paths,run.wacc_shocks,'none');x=signed.enterprise_value_proxy_usd_mn/1000;samples[key]=x
 rows.append({'model':key,'p10_bn':np.quantile(x,.1),'median_bn':np.median(x),'p90_bn':np.quantile(x,.9),'mean_bn':np.mean(x),'negative_fraction':np.mean(x<0),'five_year_median_bn':np.median(finite.enterprise_value_proxy_usd_mn)/1000,'terminal_uplift_mean_bn':np.mean(legacy.enterprise_value_proxy_usd_mn-signed.enterprise_value_proxy_usd_mn)/1000})
 for policy,v in [('Signed perpetuity',signed),('Five years only',finite),('Legacy positive',legacy)]:tc.append({'model':key,'policy':policy,'median_bn':np.median(v.enterprise_value_proxy_usd_mn)/1000})
pd.DataFrame(rows).to_csv(O/'proxy_summary.csv',index=False);pd.DataFrame(tc).to_csv(O/'terminal_policy_comparison.csv',index=False)
np.savez_compressed(O/'proxy_samples.npz',**samples)
keys=list(samples);colors=['#7287b1','#d59655','#55968b','#825da9']
fig,ax=plt.subplots(figsize=(6,3));x=samples[keys[-1]];ax.hist(x,bins=85,color=colors[-1],alpha=.65,density=True);ax.axvline(0,color='black',lw=.7);ax.set(xlabel='Unconstrained operating proxy (USD billion)',ylabel='Density',title='Audited Bates-Hawkes | signed terminal');ax.set_xlim(np.quantile(x,.005),np.quantile(x,.995));fig.tight_layout();save_figure(fig,F/'fig_val_primary.png');plt.close(fig)
fig,ax=plt.subplots(figsize=(6,3))
for key,label,col in zip(keys,labels,colors):ax.hist(samples[key],bins=np.linspace(-80,300,100),histtype='step',density=True,label=label,color=col)
ax.set(xlabel='Unconstrained operating proxy (USD billion)',ylabel='Density',title='Common operating inputs | no equity or market comparison');ax.legend(fontsize=7);fig.tight_layout();save_figure(fig,F/'fig_val_multi.png');plt.close(fig)
fig,ax=plt.subplots(figsize=(6,3));xx=np.arange(4)
for i,policy in enumerate(['Five years only','Signed perpetuity','Legacy positive']):ax.bar(xx+(i-1)*.24,[r['median_bn'] for r in tc if r['policy']==policy],width=.24,label=policy)
ax.set_xticks(xx,labels,rotation=12);ax.set(ylabel='Median proxy (USD billion)',title='Terminal closure is a scenario assumption');ax.legend(fontsize=7);fig.tight_layout();save_figure(fig,F/'fig_terminal_share.png');plt.close(fig)
o=pd.read_csv(O/'oos_audit.csv');mean=o[o.benchmark=='mean'];fig,ax=plt.subplots(figsize=(6,2.9));ax.bar(labels,mean.mean_daily_rmse_bp,color=colors);ax.set(ylabel='Mean daily IV RMSE (bp)',title='Historical conditional repricing | original fitted parameters');ax.tick_params(axis='x',rotation=10);fig.tight_layout();save_figure(fig,F/'oos_rmse.png');plt.close(fig)
fig,ax=plt.subplots(figsize=(6,2.8))
for i,bench in enumerate(['mean','persistence']):ax.bar(xx+(i-.5)*.33,100*o[o.benchmark==bench].r2,width=.33,label=bench)
ax.axhline(0,color='black',lw=.6);ax.set_xticks(xx,labels,rotation=10);ax.set(ylabel='Equal-date R-squared (%)',title='Historical paired supports | ratio of daily MSE sums');ax.legend(fontsize=7);fig.tight_layout();save_figure(fig,F/'oos_r2.png');plt.close(fig)
print(json.dumps({'calibration':summary,'proxy':rows},indent=2),flush=True)
