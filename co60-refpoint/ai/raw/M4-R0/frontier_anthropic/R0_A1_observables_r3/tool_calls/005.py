
code = r'''
import numpy as np, time
me=0.51099895; re=2.8179403e-13; ne_w=3.3428e23
SF=110.25
lines=np.array([1.1732,1.3325])
Eb=np.array([0,0.05,0.1,0.15,0.2,0.25,0.3,0.35,0.4,0.45,0.5,0.55,0.6,0.65,0.7,0.75,0.8,0.85,0.9,0.95,1,1.05,1.1,1.15,1.2,1.25,1.3,1.35,1.4])
cb=np.array([-1,-0.8,-0.6,-0.4,-0.2,0,0.2,0.4,0.6,0.8,1.0])
ebE=np.array([0.01,0.05,0.1,0.15,0.2,0.25,0.3,0.35,0.4,0.45,0.5,0.55,0.6,0.65,0.7,0.75,0.8,0.85,0.9,0.95,1,1.05,1.1,1.15,1.2,1.4])
def KN_sigma(E):
    k=E/me
    t1=(1+k)/k**2*(2*(1+k)/(1+2*k)-np.log(1+2*k)/k)
    t2=np.log(1+2*k)/(2*k)
    t3=-(1+3*k)/(1+2*k)**2
    return 2*np.pi*re**2*(t1+t2+t3)
def dsig(E,c):
    k=E/me; eps=1/(1+k*(1-c))
    return 0.5*re**2*eps**2*(eps+1/eps-(1-c**2))
def muC(E): return ne_w*KN_sigma(E)
def mupe(E): return 0.0028*(0.1/np.maximum(E,1e-4))**3
def mupair(E): return np.where(E>1.022, 0.0151*np.clip(1-1.022/np.maximum(E,1e-3),0,None)**3,0.0)
def muR(E): return 0.0053*(0.1/np.maximum(E,1e-4))**2
def mutot(E): return muC(E)+mupe(E)+mupair(E)+muR(E)
neg_w=6.02214e23*0.55509
neg_a=6.02214e23*0.49919
Iw=75e-6; Ia=85.7e-6
def rad_S(T, X0=36.08):
    f=np.minimum(0.31*np.maximum(T,1e-4)**0.3,1.0)
    return (T+me)/X0*f
def Scol_bethe(T, I, neg):
    tau=T/me; b2=tau*(tau+2)/(tau+1)**2
    Kc=2*np.pi*re**2*me*neg
    F=1-b2+(tau**2/8-(2*tau+1)*np.log(2))/(tau+1)**2
    return Kc/b2*(np.log(tau**2*(tau+2)/(2*(I/me)**2))+F)
Tg=np.logspace(-3,np.log10(1.5),600)
def Sratio(T): return rad_S(T)/(rad_S(T)+Scol_bethe(T,Iw,neg_w))
gT=np.zeros_like(Tg)
for i,T in enumerate(Tg):
    x=np.logspace(-4,np.log10(T),400)
    gT[i]=np.trapezoid(Sratio(x),x)/T
def g_of_T(T): return np.interp(np.log(np.maximum(T,1e-3)),np.log(Tg),gT)
xg,wg=np.polynomial.legendre.leggauss(300)
def KN_table(E):
    k=E/me; eps=1/(1+k*(1-xg))
    f=0.5*re**2*eps**2*(eps+1/eps-(1-xg**2))
    T=E*(1-eps)
    s=2*np.pi*np.sum(wg*f)
    Tm=np.sum(wg*f*T)/np.sum(wg*f)
    TgT=np.sum(wg*f*T*(1-g_of_T(T)))/np.sum(wg*f)
    return s,Tm,TgT
Etab=np.logspace(-3,np.log10(1.5),400)
tab=np.array([KN_table(E) for E in Etab])
def interpE(arr,E): return np.interp(np.log(np.maximum(E,1e-3)),np.log(Etab),arr)
def Tmean(E): return interpE(tab[:,1],E)
def TgT(E): return interpE(tab[:,2],E)
# energy transfer coefficients (ratio, cm^-1) for photon energy E
def ET(E):  # energy transferred to charged particles coefficient
    return muC(E)*Tmean(E)+mupe(E)*E+mupair(E)*(E-2*me)
def KC(E):  # collisional part
    return muC(E)*TgT(E)+mupe(E)*E+mupair(E)*(E-2*me)
def sample_KN(E, rng):
    """returns eps=E'/E and cos of scattering angle, vectorized rejection"""
    n=E.size; eps=np.empty(n); cc=np.empty(n)
    idx=np.arange(n)
    while idx.size>0:
        c=2*rng.random(idx.size)-1
        k=E[idx]/me
        e=1/(1+k*(1-c))
        f=e**2*(e+1/e-(1-c**2))
        acc=rng.random(idx.size)*2.0 < f
        a=idx[acc]
        eps[a]=e[acc]; cc[a]=c[acc]
        idx=idx[~acc]
    return eps,cc
def rotate(d,c,phi):
    # d: (n,3) unit; c,phi arrays
    n=d.shape[0]
    a=np.zeros_like(d); 
    mask=np.abs(d[:,0])<0.9
    a[mask,0]=1; a[~mask,1]=1
    e1=np.cross(d,a); e1/=np.linalg.norm(e1,axis=1)[:,None]
    e2=np.cross(d,e1)
    s=np.sqrt(np.clip(1-c**2,0,None))
    return c[:,None]*d+s[:,None]*(np.cos(phi)[:,None]*e1+np.sin(phi)[:,None]*e2)
def isotropic(n,rng):
    z=2*rng.random(n)-1; ph=2*np.pi*rng.random(n); s=np.sqrt(1-z*z)
    return np.stack([s*np.cos(ph),s*np.sin(ph),z],axis=1)
'''
open('/tmp/work/physics.py','w').write(code)
import sys; sys.path.insert(0,'/tmp/work')
import physics as P
import numpy as np
print(P.KN_table(1.1732), P.KN_sigma(1.1732), P.gT[-1], P.g_of_T(np.array([0.1,0.5,1.0])))
print(P.ET(1.1732)/P.mutot(1.1732), P.KC(1.1732)/P.ET(1.1732))
