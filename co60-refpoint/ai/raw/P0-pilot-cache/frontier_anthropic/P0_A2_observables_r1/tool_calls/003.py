
import numpy as np, time
rng=np.random.default_rng(1)
# ---- cross sections (per cm, rho=1)
def lg(col, E):
    E=np.atleast_1d(E).astype(float)
    x=np.log(xc[:,0]); 
    y=np.log(np.maximum(col,1e-30))
    out=np.exp(np.interp(np.log(np.maximum(E,1e-4)),x,y))
    return out
mu_coh_t=xc[:,1]; mu_inc_t=xc[:,2]; mu_pe_t=xc[:,3]; mu_pr_t=xc[:,4]+xc[:,5]
def mu(E,which):
    E=np.atleast_1d(np.asarray(E,float))
    col={'coh':mu_coh_t,'inc':mu_inc_t,'pe':mu_pe_t,'pr':mu_pr_t}[which]
    out=np.zeros_like(E)
    m=(E>=0.001)&(E<=20)
    if which=='pr':
        m2=m&(E>1.022)
        out[m2]=np.exp(np.interp(np.log(E[m2]),np.log(xc[:,0]),np.log(np.maximum(col,1e-30))))
        # pair ~ linear interp for E near threshold handled by table
        return out
    out[m]=np.exp(np.interp(np.log(E[m]),np.log(xc[:,0]),np.log(np.maximum(col,1e-30))))
    return out
def mut(E): return mu(E,'coh')+mu(E,'inc')+mu(E,'pe')+mu(E,'pr')
print(mu(2.0,'coh'),mu(2.0,'inc'),mu(2.0,'pe'),mu(2.0,'pr'),mut(2.0))
print(mut(0.5),mu(0.5,'inc'), mu(1.0,'pr'))
