
phys = r'''
import numpy as np
from scipy.integrate import quad
from scipy.interpolate import interp1d
mec2=0.51099895
re=2.8179403e-13
NA=6.02214076e23
xc=np.loadtxt('/tmp/data/xcom_water.txt'); es=np.loadtxt('/tmp/data/estar_water.txt')
Ex=xc[:,0]
def lli(col):
    f=interp1d(np.log(Ex),np.log(np.maximum(col,1e-30)),kind='linear',fill_value='extrapolate')
    return lambda E: np.exp(f(np.log(E)))
mu_coh=lli(xc[:,1]); mu_inc=lli(xc[:,2]); mu_pe=lli(xc[:,3]); mu_pn=lli(xc[:,4]); mu_pe_=lli(xc[:,5])
def mu_pair(E): return mu_pn(E)+mu_pe_(E)
def mu_tot_fn(E): return mu_coh(E)+mu_inc(E)+mu_pe(E)+mu_pair(E)
NeW=NA*0.55509   # electrons per gram water
NeAir=NA*0.49919
def kn_eps_sample(E,rng):
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
def beta2(T):
    g=1+T/mec2; return 1-1/g**2
def moller_dsde(T,eps,variant=1):
    tau=T/mec2; x=eps/T
    b2=beta2(T)
    c = (2*tau+1)/(tau+2)**2 if variant==1 else (2*tau+1)/(tau+1)**2
    B=1/x**2+1/(1-x)**2+(tau/(tau+2))**2-c/(x*(1-x))
    return 2*np.pi*re**2*mec2/(b2*T**2)*B
def above(T,D,variant=1,Ne=NeW):
    if D>=T/2: return 0.0
    f=lambda eps: eps*moller_dsde(T,eps,variant)
    pts=[x for x in [D*3,D*10,T/10] if D<x<T/2]
    v,_=quad(f,D,T/2,limit=500,points=pts or None)
    return Ne*v
def estar(T):
    # interpolate ESTAR columns in log-log: returns Scol, Srad, Stot, CSDA, Yrad, delta
    lT=np.log(es[:,0])
    out=[]
    for j in [1,2,3,4,5]:
        out.append(np.exp(np.interp(np.log(T),lT,np.log(np.maximum(es[:,j],1e-30)))))
    return out
'''
open('/tmp/phys.py','w').write(phys)
import sys; sys.path.insert(0,'/tmp')
import importlib, phys; importlib.reload(phys)
import numpy as np
def Fminus(tau):
    b2=1-1/(1+tau)**2
    return 1-b2+(tau**2/8-(2*tau+1)*np.log(2))/(tau+1)**2
def bethe_part(tau): return np.log(tau**2*(tau+2)/2)+Fminus(tau)
D=0.01
for T in [0.1,0.3,0.5,1.0,2.0]:
    tau=T/phys.mec2
    Scol_E=phys.estar(T)[0]
    a1=phys.above(T,D,1); a2=phys.above(T,D,2)
    print(T, round(Scol_E,4), round(a1,5), round(a2,5), 'bethe_part',round(bethe_part(tau),4))
