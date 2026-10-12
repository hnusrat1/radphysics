
import numpy as np, time
A_uv=0.01
Om=4*np.arcsin(np.sin(np.arctan(0.05))**2)
F0=Om*105.0**2
def mu_parts(E):
    E=np.asarray(E,float)
    tot=mu_tab(E)
    sC=np.array([kn_sigma(e) for e in np.atleast_1d(E)])*ne_w
    sF=np.array([kn_tr_frac(e)[1] for e in np.atleast_1d(E)])*ne_w
    coh=np.minimum(0.0055*(0.1/np.atleast_1d(E))**2, 0.5*(tot-sC))
    resid=np.maximum(tot-sC-coh,0)
    pe=np.where(np.atleast_1d(E)<1.022,resid,0.0)
    pair=np.where(np.atleast_1d(E)>=1.022,resid,0.0)
    return dict(tot=tot, C=sC, trC=sF, coh=coh, pe=pe, pair=pair)
Etest=np.array([0.02,0.1,0.3,0.5,1.0,1.1732,1.3325])
for k,v in mu_parts(Etest).items(): print(k, np.round(v,5))
