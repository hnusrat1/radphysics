
import numpy as np, time
rng=np.random.default_rng(1)
Lines=np.array([1.1732,1.3325])
def mu_pe(E): return 0.0022*(0.1/E)**3
def mu_coh(E): return 0.0030*(0.1/E)**2
def mu_pair(E):
    x=np.clip((E-1.022)/0.308,0,None); return 0.0002*x**2
def mu_c(E): return ne*sigKN(E)
def mu_tot(E): return mu_c(E)+mu_coh(E)+mu_pe(E)+mu_pair(E)
def mu_tr_c(E):
    # energy transfer coefficient: integral of (E-E') f
    c=np.linspace(-1,1,401); w=np.ones_like(c)
    from numpy.polynomial.legendre import leggauss
    x,wt=leggauss(120)
    out=[]
    for e in np.atleast_1d(E):
        Ep=1/(1+e/mc2*(1-x))
        out.append(ne*2*np.pi*np.sum(wt*(e-e*Ep)*fKN(e,x)))
    return np.array(out).reshape(np.shape(E)) if np.ndim(E)>0 else out[0]
def mu_en_c(E): return mu_tr_c(E)*0.9965  # approx radiative loss in electrons
print(mu_tot(Lines), mu_tr_c(Lines), mu_pair(Lines), mu_coh(Lines))
Omega=4*np.arcsin(25/10025.); print(Omega)
