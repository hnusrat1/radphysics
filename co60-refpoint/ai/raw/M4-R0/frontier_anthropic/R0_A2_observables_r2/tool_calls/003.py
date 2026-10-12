
import numpy as np, time
from scipy.interpolate import interp1d
mec2=0.51099895
# log-log interpolation tables of XCOM components
Ex=xc[:,0]
def lli(col):
    f=interp1d(np.log(Ex),np.log(np.maximum(col,1e-30)),kind='linear',fill_value='extrapolate')
    return lambda E: np.exp(f(np.log(E)))
mu_coh=lli(xc[:,1]); mu_inc=lli(xc[:,2]); mu_pe=lli(xc[:,3]); mu_pn=lli(xc[:,4]); mu_pe_=lli(xc[:,5])
mu_pair=lambda E: mu_pn(E)+mu_pe_(E)
mu_tot_fn=lli(xc[:,6])
# check total = sum
E0=np.array([0.5,1.1732,1.3325])
print(mu_coh(E0)+mu_inc(E0)+mu_pe(E0)+mu_pair(E0), mu_tot_fn(E0))
# Compton: mean energy transfer fraction from KN (per-electron shape)
def kn_eps_sample(E,rng):
    # sample eps=E'/E from KN; vectorized rejection in eps
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
# KN mean energy fraction <T/E> and check
rng=np.random.default_rng(1)
for E in [0.1,0.5,1.1732,1.3325]:
    e=kn_eps_sample(np.full(200000,E),rng)
    print(E, 1-e.mean(), 'check uniform')
