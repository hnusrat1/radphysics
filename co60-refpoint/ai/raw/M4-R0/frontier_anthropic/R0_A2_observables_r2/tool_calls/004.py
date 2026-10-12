
code = r'''
import numpy as np
from scipy.interpolate import interp1d
mec2=0.51099895
re=2.8179403e-13  # cm
NA=6.02214076e23
xc=np.loadtxt('/tmp/data/xcom_water.txt'); es=np.loadtxt('/tmp/data/estar_water.txt')
Ex=xc[:,0]
def lli(col):
    f=interp1d(np.log(Ex),np.log(np.maximum(col,1e-30)),kind='linear',fill_value='extrapolate')
    return lambda E: np.exp(f(np.log(E)))
mu_coh=lli(xc[:,1]); mu_inc=lli(xc[:,2]); mu_pe=lli(xc[:,3]); mu_pn=lli(xc[:,4]); mu_pe_=lli(xc[:,5])
def mu_pair(E): return mu_pn(E)+mu_pe_(E)
def mu_tot_fn(E): return mu_coh(E)+mu_inc(E)+mu_pe(E)+mu_pair(E)
def kn_eps_sample(E,rng):
    k=E/mec2
    n=len(E); out=np.empty(n); todo=np.arange(n)
    emin=1/(1+2*k)
    while todo.size:
        kk=k[todo]; em=emin[todo]
        e=em+(1-em)*rng.random(todo.size)
        cost=1-(1/e-1)/kk; sin2=1-cost**2
        f=e+1/e-sin2
        fmax=em+1/em
        acc=rng.random(todo.size)*fmax<f
        out[todo[acc]]=e[acc]
        todo=todo[~acc]
    return out
'''
open('/tmp/work_base.py','w').write(code)
import sys; sys.path.insert(0,'/tmp')
import work_base as wb, numpy as np
E0=np.array([0.5,1.1732,1.3325])
print(wb.mu_coh(E0)+wb.mu_inc(E0)+wb.mu_pe(E0)+wb.mu_pair(E0), wb.mu_tot_fn(E0))
rng=np.random.default_rng(1)
for E in [0.1,0.5,1.1732,1.3325]:
    e=wb.kn_eps_sample(np.full(200000,E),rng)
    print(E, 'mean T/E', 1-e.mean())
