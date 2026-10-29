
mod = r'''
import numpy as np
x = np.loadtxt('/tmp/data/xcom_water.txt')
e = np.loadtxt('/tmp/data/estar_water.txt')
XE = x[:,0]; XCOH=x[:,1]; XINC=x[:,2]; XPE=x[:,3]; XPAIR=x[:,4]+x[:,5]
TE = e[:,0]; SCOL=e[:,1]; SRAD=e[:,2]; STOT=e[:,3]; RANGE=e[:,4]; YLD=e[:,5]; DELTA=e[:,6]
re=2.8179403e-13; mc2=0.51099895; NA=6.02214076e23; ZA_W=0.55508; ZA_AIR=0.49919
ne=NA*ZA_W
def loglog(xg, yg, E):
    yy = np.log(np.maximum(yg,1e-30))
    return np.exp(np.interp(np.log(E), np.log(xg), yy))
def mu_coh(E): return loglog(XE,XCOH,E)
def mu_inc(E): return loglog(XE,XINC,E)
def mu_pe(E):  return loglog(XE,XPE,E)
def mu_pair(E):
    E=np.asarray(E,float)
    out = loglog(XE,np.maximum(XPAIR,1e-30),E)
    return np.where(E>1.022, out, 0.0)
def mu_tot(E): return mu_coh(E)+mu_inc(E)+mu_pe(E)+mu_pair(E)
def S_col(T): return loglog(TE,SCOL,T)
def S_rad(T): return loglog(TE,SRAD,T)
def S_tot(T): return S_col(T)+S_rad(T)
def CSDA(T): return loglog(TE,RANGE,T)
def Yld(T): return np.interp(np.log(np.maximum(T,TE[0])), np.log(TE), YLD)
def DEL(T): return np.interp(np.log(np.maximum(T,TE[0])), np.log(TE), DELTA)
def sigKN(E):
    k=np.asarray(E,float)/mc2
    L=np.log(1+2*k)
    return 2*np.pi*re**2*((1+k)/k**2*(2*(1+k)/(1+2*k)-L/k)+L/(2*k)-(1+3*k)/(1+2*k)**2)
def Srestr(T, Delta, I_eV, ZA, delta=None):
    T=np.asarray(T,float); Delta=np.minimum(np.asarray(Delta,float), T/2)
    tau=T/mc2; b2=1-1/(1+tau)**2; Ip=I_eV*1e-6/mc2; D=Delta/mc2
    if delta is None: delta=np.zeros_like(T)
    F = -1-b2+np.log((tau-D)*D)+tau/(tau-D)+(D**2/2+(2*tau+1)*np.log(1-D/tau))/(tau+1)**2
    K = 2*np.pi*re**2*mc2*NA*ZA
    return K/b2*(np.log(2*(tau+2)/Ip**2)+F-delta)
def moller(Ep, T):
    """dsigma/dT (cm2/MeV) per electron, parent kinetic Ep, delta kinetic T; T<=Ep/2"""
    tau=Ep/mc2; b2=1-1/(1+tau)**2; xx=T/Ep
    f = 1/xx**2+1/(1-xx)**2+(tau/(tau+1))**2-(2*tau+1)/((tau+1)**2*xx*(1-xx))
    return 2*np.pi*re**2*mc2/(b2*Ep**2)*f
'''
open('/tmp/work/phys.py','w').write(mod)
import sys; sys.path.insert(0,'/tmp/work')
import importlib, phys; importlib.reload(phys)
from phys import *
for T in [0.1,0.5,1.0,1.25]:
    print(T, Srestr(T,T/2,75.0,ZA_W,DEL(T)), S_col(T), Srestr(T,0.01,75.0,ZA_W,DEL(T)), Srestr(T,0.01,85.7,ZA_AIR,0.0))
