
elec = r'''
import numpy as np
me=0.51099895; re=2.8179403e-13; alpha=1/137.036; NA=6.02214076e23
def beta2(T):
    tau=T/me; return tau*(tau+2)/(tau+1)**2
def density_delta(T, C,x0,x1,a,m):
    beta=np.sqrt(beta2(T)); gam=1+T/me; X=np.log10(beta*gam)
    mid=(X>=x0)&(X<x1)
    d=np.zeros_like(T)
    d=np.where(mid,4.6052*X-C+a*(x1-X)**m,d)
    d=np.where(X>=x1,4.6052*X-C,d)
    return d
WATER=dict(ZA=0.55509,I=75e-6,C=3.5017,x0=0.2400,x1=2.8004,a=0.09116,m=3.4773,ne=3.3428e23)
AIR  =dict(ZA=0.49919,I=85.7e-6,C=10.5961,x0=1.7418,x1=4.2759,a=0.10914,m=3.2125,ne=3.0062e23)
def Scol_per_mass(T,mat):
    T=np.atleast_1d(np.asarray(T,float))
    tau=T/me; b2=beta2(T)
    F=1-b2+(tau**2/8-(2*tau+1)*np.log(2))/(tau+1)**2
    delta=density_delta(T,mat['C'],mat['x0'],mat['x1'],mat['a'],mat['m'])
    L=np.log(tau**2*(tau+2)/(2*(mat['I']/me)**2))+F-delta
    return 0.153537*mat['ZA']/b2*L
def Srad_per_mass(T):
    T=np.atleast_1d(np.asarray(T,float))
    tt=T+me
    return 76*NA/18.015*alpha*re**2*tt*(4*np.log(2*tt/me)-4/3)
def moller_dsig(T, eps):
    b2=beta2(T); tau=T/me
    x=eps/T
    f=1/x**2+1/(1-x)**2+(tau/(tau+1))**2-(2*tau+1)/((tau+1)**2*x*(1-x))
    return 2*np.pi*re**2*me/b2/T**2*f
def Ldelta_per_e(T, Dcut=0.01):
    T=np.atleast_1d(np.asarray(T,float))
    out=np.zeros_like(T)
    for i,t in enumerate(T):
        if t/2<=Dcut: continue
        e=np.exp(np.linspace(np.log(Dcut),np.log(t/2),400))
        out[i]=np.trapezoid(e*moller_dsig(t,e),e)
    return out
def Sres_mass(T,mat,Dcut=0.01):
    return Scol_per_mass(T,mat) - Ldelta_per_e(T,Dcut)*mat['ne']
'''
open('/tmp/work/elec.py','w').write(elec)
import sys; sys.path.insert(0,'/tmp/work')
import elec, numpy as np
print(elec.Sres_mass(np.array([0.02,0.1,0.5,1.0]),elec.WATER))
print(elec.Scol_per_mass(np.array([0.02,0.1,0.5,1.0]),elec.WATER))
# check Moller normalisation: total delta cross section above Dcut vs known (T=1 MeV, Dcut 10 keV) ~ 
T=1.0; e=np.exp(np.linspace(np.log(0.01),np.log(0.5),400))
print('sigma_delta(T=1,>10keV) cm2/e', np.trapezoid(elec.moller_dsig(T,e),e), 'N_e*sigma', np.trapezoid(elec.moller_dsig(T,e),e)*3.3428e23)
