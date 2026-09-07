from pathlib import Path
import sys,json
import numpy as np,pandas as pd
from scipy.optimize import minimize
R=Path(__file__).resolve().parent;sys.path.insert(0,str(R/'pricing'))
from Hawkes import ExactHawkesCalibration as H
D=R/'audit_inputs/outputs/calibrations/CC/2026-09-02';O=R/'audit_outputs'
a=json.loads((O/'calibration_audit.json').read_text());b=np.array(a['refined_bates_parameters']);bp=np.r_[b[:5],b[5],b[5],0.,2.,b[6:]]
f=pd.read_csv(D/'calibration_surface.csv');S=float(f.spot.iloc[0]);p=json.loads((D/'full_bates_hawkes.json').read_text())['parameters'];old=np.array([p[k] for k in ['v0','kappa','theta','xi','rho','lambda0','lambda_bar','branching_ratio','beta','mu_J','sigma_J']])
bounds=[(1e-4,1),(.1,10),(1e-4,1),(.01,8),(-.99,.99),(1e-6,5),(1e-6,5),(0,.95),(.1,12),(-.5,.5),(1e-4,.6)]
cs=({'type':'ineq','fun':lambda x:2*x[1]*x[2]-x[3]**2},)
candidates=[old,bp]
for start in [bp,old]:
 z=minimize(H.objective_heston,start,args=(f,S,0.,None,256),method='SLSQP',bounds=bounds,constraints=cs,options={'ftol':1e-11,'maxiter':100});candidates.append(z.x)
best=min(candidates,key=lambda x:H.objective_heston(x,f,S,cos_N=256));loss=H.objective_heston(best,f,S,cos_N=256)
assert loss<=a['refined_bates_loss']+1e-10
out={'parameters':H.unpack_heston_params(best),'loss':loss,'nested_bates_loss':H.objective_heston(bp,f,S,cos_N=256),'bounds':bounds,'candidate_preservation':True,'scope':'current snapshot only; historical OOS untouched'}
(O/'hawkes_refinement.json').write_text(json.dumps(out,indent=2));print(json.dumps(out),flush=True)
