
import numpy as np, time
from scipy.interpolate import interp1d
me=0.51099895; re=2.8179403e-13
ne_w=3.3428e23
# NIST-like water total mu/rho (cm2/g), incl. coherent
Etab=np.array([0.01,0.015,0.02,0.03,0.04,0.05,0.06,0.08,0.1,0.15,0.2,0.3,0.4,0.5,0.6,0.8,1.0,1.25,1.5,2.0])
mutab=np.array([4.944,1.374,0.8096,0.3756,0.2683,0.2269,0.2059,0.1837,0.1707,0.1505,0.1370,0.1186,0.1061,0.0969,0.0893,0.0765,0.0707,0.0632,0.0572,0.0494])
def loglog(x,xt,yt):
    return np.exp(np.interp(np.log(x),np.log(xt),np.log(yt)))
def mu_tab(E): return loglog(E,Etab,mutab)
def kn_sigma(E):
    k=E/me
    t1=(1+k)/k**2*(2*(1+k)/(1+2*k)-np.log(1+2*k)/k)
    t2=np.log(1+2*k)/(2*k)
    t3=(1+3*k)/(1+2*k)**2
    return 2*np.pi*re**2*(t1+t2-t3)
def kn_tr_frac(E):
    # sigma_frac = int dsigma*(1-eps) dOmega  (energy-transfer fraction weighted)
    k=E/me
    c=np.linspace(-1,1,40001)
    eps=1/(1+k*(1-c))
    f=eps**2*(eps+1/eps-(1-c**2))
    sig=np.pi*re**2*np.trapz(f,c)
    sigf=np.pi*re**2*np.trapz(f*(1-eps),c)
    return sig,sigf
for E in [0.1,0.5,1.0,1.1732,1.25,1.3325]:
    s,sf=kn_tr_frac(E)
    print(E, kn_sigma(E)*ne_w, s*ne_w, sf*ne_w/ (kn_sigma(E)*ne_w), 'tab',mu_tab(E), 'resid', mu_tab(E)-kn_sigma(E)*ne_w)
