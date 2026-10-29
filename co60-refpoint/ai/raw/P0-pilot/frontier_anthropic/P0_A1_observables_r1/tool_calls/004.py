
code = r'''
import numpy as np
trap=np.trapezoid
re=2.8179403e-13; mc2=0.51099895; ne=3.3428e23
def sig_KN(E):
    k=E/mc2
    L=np.log(1+2*k)
    return 2*np.pi*re**2*((1+k)/k**2*(2*(1+k)/(1+2*k)-L/k)+L/(2*k)-(1+3*k)/(1+2*k)**2)
def mu_pe(E): return 3.0e-6*E**-3
def mu_coh(E): return 0.005*(0.1/E)**1.07
def mu_pair(E): return np.where(E>1.022, 0.0004*np.clip((E-1.022)/0.5,0,1),0.0)
def mu_c(E): return ne*sig_KN(E)
def mu_tot(E): return mu_c(E)+mu_pe(E)+mu_pair(E)+mu_coh(E)
cg=np.linspace(-1,1,20001)
Egrid=np.logspace(-3,np.log10(2.5),400)
rmean=[]
for e in Egrid:
    k=e/mc2; r=1/(1+k*(1-cg)); f=r**2*(r+1/r-(1-cg**2))
    rmean.append(trap(f*r,cg)/trap(f,cg))
rmean=np.array(rmean)
def Tbar(E): return E*(1-np.interp(np.log(E),np.log(Egrid),rmean))
def sample_KN(E,rng):
    E=np.asarray(E,float); out_c=np.empty(E.shape)
    todo=np.arange(E.size)
    while todo.size:
        k=E[todo]/mc2
        c=rng.uniform(-1,1,todo.size)
        r=1/(1+k*(1-c))
        f=r**2*(r+1/r-(1-c**2))
        acc=rng.uniform(0,2,todo.size)<f
        out_c[todo[acc]]=c[acc]
        todo=todo[~acc]
    r=1/(1+E/mc2*(1-out_c))
    return E*r, out_c
def perp_basis(d):
    a=np.where((np.abs(d[:,[0]])<0.9),np.array([[1.0,0,0]]),np.array([[0,1.0,0]]))
    u=np.cross(d,a); u/=np.linalg.norm(u,axis=1)[:,None]
    v=np.cross(d,u)
    return u,v
'''
open('phys.py','w').write(code)
import importlib, sys
sys.path.insert(0,'.')
import phys; importlib.reload(phys)
from phys import *
rng=np.random.default_rng(1)
Ep,cc=sample_KN(np.full(400000,2.0),rng)
print("sampled <E'>/E", Ep.mean()/2, "table", Tbar(2.0)/2)
print("mu_c 2MeV", mu_c(2.0), "Tbar/E", Tbar(2.0)/2)
print("KN mean ratio check at 0.1 MeV", Tbar(0.1)/0.1, sig_KN(0.1)*ne)
