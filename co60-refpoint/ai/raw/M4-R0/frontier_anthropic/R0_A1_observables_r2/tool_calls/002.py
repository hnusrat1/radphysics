
import numpy as np, time
rng=np.random.default_rng(1)
def mu_pe(E): return 0.00345*(0.05/E)**3
def mu_coh(E): return 0.0019*(0.1/E)**2
def mu_pair(E):
    x=np.clip(E-1.022,0,None)/0.31
    return 0.0004*x**2
def mu_tot_parts(E):
    return dict(C=muKN(E), PE=mu_pe(E), PP=mu_pair(E), R=mu_coh(E))
# Compton energy transfer fraction mean (analytic via sampling later); sampling functions
def sample_compton(E,n):
    k=E/mc2
    out_eps=np.empty(n); out_mu=np.empty(n)
    todo=np.arange(n)
    while todo.size>0:
        m=todo.size
        mu=rng.uniform(-1,1,m)
        eps=1/(1+k*(1-mu))
        f=eps**2*(eps+1/eps-1+mu**2)
        acc=rng.uniform(0,2,m)<f
        idx=todo[acc]
        out_eps[idx]=eps[acc]; out_mu[idx]=mu[acc]
        todo=todo[~acc]
    return out_eps,out_mu
# test mean energy transfer fraction vs analytic
E=1.25; e,m=sample_compton(np.full(200000,E),200000) if False else sample_compton(E,200000)
print("T/E mean",np.mean(1-e), "sigma_tr/sigma approx")
