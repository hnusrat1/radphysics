
code = r'''
import numpy as np
re=2.8179403e-13; mc2=0.51099895; ne=3.3428e23
def sigKN(E):
    k=E/mc2
    t1=(1+k)/k**2*(2*(1+k)/(1+2*k)-np.log(1+2*k)/k)
    t2=np.log(1+2*k)/(2*k)
    t3=(1+3*k)/(1+2*k)**2
    return 2*np.pi*re**2*(t1+t2-t3)
def muKN(E): return ne*sigKN(E)
def mu_pe(E): return 0.00345*(0.05/E)**3
def mu_coh(E): return 0.0019*(0.1/E)**2
def mu_pair(E):
    x=np.clip(E-1.022,0,None)/0.31
    return 0.0004*x**2
def sample_compton(E,rng):
    # E: array
    E=np.asarray(E,float); n=E.size
    k=E/mc2
    out_eps=np.empty(n); out_mu=np.empty(n)
    todo=np.arange(n)
    while todo.size>0:
        m=todo.size
        kk=k[todo]
        mu=rng.uniform(-1,1,m)
        eps=1/(1+kk*(1-mu))
        f=eps**2*(eps+1/eps-1+mu**2)
        acc=rng.uniform(0,2,m)<f
        idx=todo[acc]
        out_eps[idx]=eps[acc]; out_mu[idx]=mu[acc]
        todo=todo[~acc]
    return out_eps,out_mu
'''
open('mc.py','w').write(code)
import importlib, sys
sys.path.insert(0,'.')
import mc, numpy as np
rng=np.random.default_rng(1)
E=np.full(400000,1.25)
e,m=mc.sample_compton(E,rng)
print("mean T/E",np.mean(1-e), "muKN*meanT/E / muKN check")
# analytic energy transfer fraction: sigma_tr/sigma (Compton); compare
print(np.mean(1-e))
