from pathlib import Path
import sys
import numpy as np
from scipy.optimize import OptimizeResult
from time import perf_counter
sys.path.insert(0,str(Path(__file__).resolve().parents[1]/"pricing"))
from calibration_core import CalibrationReport
from Bates import Bates
from Heston import Heston
from BatesHawkesExact import BatesHawkesExact

def test_better_global_candidate_is_not_discarded():
    local=OptimizeResult(x=[2.],fun=4.,success=True,message="local")
    glob=OptimizeResult(x=[1.],fun=1.,success=True,message="global")
    report=CalibrationReport.from_optimizer("model",("x",),local,glob,perf_counter())
    assert report.objective==1. and report.x[0]==1.

def test_zero_jumps_equal_heston():
    u=np.array([0.,.1,1.,8.,-1j]);hp=[.06,4.,.065,.7,-.03]
    a=Heston.heston_charfunc(u,400.,*hp,1.,.04)
    b=Bates.bates_charfunc(u,400.,*hp,0.,-.02,.1,1.,.04)
    np.testing.assert_allclose(a,b,atol=1e-12,rtol=1e-12)

def test_hawkes_zero_excitation_equal_bates():
    u=np.array([0.,.1,1.,8.,-1j]);hp=[.06,4.,.065,.7,-.03]
    b=Bates.bates_charfunc(u,400.,*hp,.2,-.02,.1,1.,.04)
    h=BatesHawkesExact.hawkes_charfunc(u,400.,*hp,.2,.2,0.,2.,-.02,.1,1.,.04)
    np.testing.assert_allclose(b,h,atol=1e-12,rtol=1e-12)

def test_hawkes_zero_intensity_is_valid_boundary():
    hp=[.06,4.,.065,.7,-.03];u=np.array([0.,.2,1.])
    h=BatesHawkesExact.hawkes_charfunc(u,400.,*hp,0.,0.,0.,2.,-.02,.1,1.,.04)
    a=Heston.heston_charfunc(u,400.,*hp,1.,.04)
    np.testing.assert_allclose(a,h,atol=1e-12,rtol=1e-12)


def test_finite_global_survives_nonfinite_local():
    for bad in (np.nan, np.inf, -np.inf):
        local = OptimizeResult(x=[2.], fun=bad, success=False, message="invalid local")
        glob = OptimizeResult(x=[1.], fun=1., success=True, message="global")
        report = CalibrationReport.from_optimizer("model", ("x",), local, glob, perf_counter())
        assert report.objective == 1. and report.x[0] == 1.
