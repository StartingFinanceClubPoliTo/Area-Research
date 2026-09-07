from pathlib import Path
import sys,json,hashlib
import numpy as np,pandas as pd
from scipy.optimize import minimize,brentq
from scipy.stats import norm
R=Path(__file__).resolve().parent;sys.path.insert(0,str(R/'pricing'));sys.path.insert(0,str(R/'src'))
from Heston import Heston
from Bates import Bates
from BatesHawkesExact import BatesHawkesExact
from Hawkes import ExactHawkesCalibration
D=R/'audit_inputs/outputs/calibrations/CC/2026-09-02';O=R/'audit_outputs';O.mkdir(exist_ok=True)
f=pd.read_csv(D/'calibration_surface.csv');spot=float(f.spot.iloc[0]);p=json.loads((D/'heston.json').read_text())['parameters']; hp=np.array([p[k] for k in ['v0','kappa','theta','sigma','rho']]);bp=np.r_[hp,0.,0.,.1]
# Verify characteristic functions and losses on the actual 64 contracts.
u=np.array([0.,.1,1.,5.,10.]);err=[]
for T in f.T if False else sorted(f['T'].unique()):
 a=Heston.heston_charfunc(u,spot,*hp,T,.04,0.)
 b=Bates.bates_charfunc(u,spot,*bp,T,.04,0.)
 c=Bates.bates_charfunc(u,spot,*np.r_[hp,.2,-.03,.1],T,.04,0.)
 d=BatesHawkesExact.hawkes_charfunc(u,spot,*hp,.2,.2,0.,2.,-.03,.1,T,.04,0.)
 err.append([float(np.max(abs(a-b))),float(np.max(abs(c-d)))])
assert np.max(err)<1e-10
lossH=Heston.heston_objective(hp,f,spot,cos_N=256)
lossB=Bates.bates_objective(bp,f,spot,cos_N=256)
assert abs(lossH-lossB)<1e-12
out={'source_sha256':hashlib.sha256((D/'calibration_surface.csv').read_bytes()).hexdigest(),'boundary_cf_max_errors':np.max(err,axis=0).tolist(),'heston_loss':lossH,'bates_zero_jump_loss':lossB}
# Safeguarded constrained local refinement; the nested candidate remains eligible.
constraints=({'type':'ineq','fun':lambda x:2*x[1]*x[2]-x[3]**2},)
orig=json.loads((D/'bates.json').read_text()); op=np.array([orig['parameters'][k] for k in Bates.PARAMETER_NAMES])
candidates=[bp,op];results=[]
for start in candidates:
 z=minimize(Bates.bates_objective,start,args=(f,spot,0.,'cos',256),method='SLSQP',bounds=Bates.BOUNDS,constraints=constraints,options={'ftol':1e-12,'maxiter':150});results.append(z.x)
best=min(candidates+results,key=lambda x:Bates.bates_objective(x,f,spot,cos_N=256));out['refined_bates_parameters']=best.tolist();out['refined_bates_loss']=Bates.bates_objective(best,f,spot,cos_N=256)
assert out['refined_bates_loss']<=lossH+1e-12
(O/'calibration_audit.json').write_text(json.dumps(out,indent=2));print(json.dumps(out),flush=True)
# Recompute all OOS aggregates from daily errors and paired date support.
dm=pd.read_csv(R/'audit_inputs/outputs/oos/rolling/date_metrics.csv');rows=[];rng=np.random.default_rng(20260907)
for model,g in dm.groupby('model',sort=False):
 for label,mc,bc,nc in [('mean','mse_iv','mean_benchmark_mse_iv','common_n'),('persistence','model_mse_on_persistence_support_iv','persistence_mse_iv','persistence_n')]:
  g=g[g[nc]>0]; a=g[mc].to_numpy();b=g[bc].to_numpy()
  rows.append({'model':model,'benchmark':label,'dates':len(g),'observations':int(g[nc].sum()),'sum_daily_model_mse':a.sum(),'sum_daily_benchmark_mse':b.sum(),'mean_daily_rmse_bp':np.sqrt(a).mean()*1e4,'root_mean_daily_mse_bp':np.sqrt(a.mean())*1e4,'benchmark_root_mean_daily_mse_bp':np.sqrt(b.mean())*1e4,'r2':1-a.sum()/b.sum()})
pd.DataFrame(rows).to_csv(O/'oos_audit.csv',index=False)
# Date-paired moving-block bootstrap (length 5), exploratory with only 30 dates.
pivot=dm.pivot(index='target_date',columns='model',values='mse_iv').sort_index();delta=pivot['hawkes'].to_numpy()-pivot['bates'].to_numpy();n=len(delta);starts=rng.integers(0,n,size=(10000,int(np.ceil(n/5))));ix=(starts[:,:,None]+np.arange(5))%n;means=delta[ix.reshape(10000,-1)[:,:n]].mean(axis=1)
(O/'oos_uncertainty.json').write_text(json.dumps({'contrast':'Hawkes minus Bates daily MSE','dates':n,'block_length':5,'replications':10000,'seed':20260907,'mean_difference':delta.mean(),'percentile_95_interval':np.quantile(means,[.025,.975]).tolist(),'interpretation':'exploratory paired circular block bootstrap; not proof of superiority'},indent=2))
print('OOS audit complete',flush=True)
