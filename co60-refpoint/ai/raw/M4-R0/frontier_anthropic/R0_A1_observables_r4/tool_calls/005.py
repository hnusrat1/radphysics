
code = r'''
import numpy as np
me=0.51099895; re=2.8179403e-13
ne_w=3.3428e23
Etab=np.array([0.01,0.015,0.02,0.03,0.04,0.05,0.06,0.08,0.1,0.15,0.2,0.3,0.4,0.5,0.6,0.8,1.0,1.25,1.5,2.0])
mutab=np.array([4.944,1.374,0.8096,0.3756,0.2683,0.2269,0.2059,0.1837,0.1707,0.1505,0.1370,0.1186,0.1061,0.0969,0.0893,0.0765,0.0707,0.0632,0.0572,0.0494])
trap=np.trapezoid
A_uv=0.01
Om=4*np.arcsin(np.sin(np.arctan(0.05))**2)
F0=Om*105.0**2
def loglog(x,xt,yt):
    return np.exp(np.interp(np.log(x),np.log(xt),np.log(yt)))
def mu_tab(E): return loglog(np.asarray(E,float),Etab,mutab)
def kn_sigma_vec(E):
    E=np.asarray(E,float); k=E/me
    t1=(1+k)/k**2*(2*(1+k)/(1+2*k)-np.log(1+2*k)/k)
    t2=np.log(1+2*k)/(2*k)
    t3=(1+3*k)/(1+2*k)**2
    return 2*np.pi*re**2*(t1+t2-t3)
_c=np.linspace(-1,1,20001)
def kn_fracs(E):
    E=float(E); k=E/me
    eps=1/(1+k*(1-_c))
    f=eps**2*(eps+1/eps-(1-_c**2))
    sig=np.pi*re**2*trap(f,_c)
    sigf=np.pi*re**2*trap(f*(1-eps),_c)
    return sig,sigf
def mu_parts(E):
    E=np.atleast_1d(np.asarray(E,float))
    tot=mu_tab(E)
    sC=kn_sigma_vec(E)*ne_w
    sF=np.array([kn_fracs(e)[1] for e in E])*ne_w
    coh=np.minimum(0.0055*(0.1/E)**2, 0.5*(tot-sC))
    resid=np.maximum(tot-sC-coh,0)
    pe=np.where(E<1.022,resid,0.0)
    pair=np.where(E>=1.022,resid,0.0)
    return dict(tot=tot, C=sC, trC=sF*1.0, coh=coh, pe=pe, pair=pair)
'''
open('phys.py','w').write(code)
import phys, numpy as np
Et=np.array([0.02,0.1,0.3,0.5,1.0,1.1732,1.3325])
for k,v in phys.mu_parts(Et).items(): print(k, np.round(v,5))
print('Om',phys.Om,'F0',phys.F0)
