
import numpy as np, os
os.makedirs('/tmp/work',exist_ok=True)
x = np.loadtxt('/tmp/data/xcom_water.txt')
e = np.loadtxt('/tmp/data/estar_water.txt')
XE = x[:,0]; XCOH=x[:,1]; XINC=x[:,2]; XPE=x[:,3]; XPAIR=x[:,4]+x[:,5]
XTOT = XCOH+XINC+XPE+XPAIR
TE = e[:,0]; SCOL=e[:,1]; SRAD=e[:,2]; STOT=e[:,3]; RANGE=e[:,4]; YLD=e[:,5]; DELTA=e[:,6]
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
def Yld(T): return np.interp(np.log(T), np.log(TE), YLD)
def DEL(T): return np.interp(np.log(T), np.log(TE), DELTA)
# checks
for E in [1.1732,1.3325,1.0]:
    print(E, mu_coh(E), mu_inc(E), mu_pe(E), mu_pair(E), mu_tot(E), np.exp(-mu_tot(E)*5))
re=2.8179403e-13; mc2=0.51099895; ne=6.02214076e23*(0.55508)  # electrons per gram water (Z/A=0.55508)
def sigKN(E):
    k=np.asarray(E,float)/mc2
    L=np.log(1+2*k)
    return 2*np.pi*re**2*((1+k)/k**2*(2*(1+k)/(1+2*k)-L/k)+L/(2*k)-(1+3*k)/(1+2*k)**2)
print("KN per g at 1MeV", sigKN(1.0)*ne, "vs xcom inc", mu_inc(1.0))
print("KN per g at 1.25", sigKN(1.25)*ne, mu_inc(1.25))
