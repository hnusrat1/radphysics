
import numpy as np, time
rng=np.random.default_rng(12345)
me=0.51099895; re=2.8179403e-13; NA=6.02214076e23
ne=NA*10/18.015*1.0          # electrons/cm3 water
nmol=NA/18.015               # molecules/cm3
def sig_KN(E):
    a=E/me; t=np.log(1+2*a)
    return 2*np.pi*re**2*((1+a)/a**2*(2*(1+a)/(1+2*a)-t/a)+t/(2*a)-(1+3*a)/(1+2*a)**2)
X0=0.8
def sig_R(E):
    th=np.linspace(0,np.pi,20001)
    lam=0.012398/E
    x=np.sin(th/2)/lam
    f=0.5*re**2*(1+np.cos(th)**2)*100*np.exp(-(x/X0)**2)*2*np.pi*np.sin(th)
    return np.trapezoid(f,th)
def mu_c(E): return ne*sig_KN(E)
def mu_R(E): return nmol*sig_R(E)
def mu_pe(E): return 0.0025*(0.1/E)**3
def mu_pair(E):
    E=np.asarray(E,float); return np.where(E>1.022,0.0215*np.clip(1-1.022/E,0,None)**2.5,0.0)
def mu_tot(E): return mu_c(E)+mu_R(E)+mu_pe(E)+mu_pair(E)
for E in [0.1,0.2,0.5,0.511,1.0,1.1732,1.25,1.3325]:
    print(E, round(float(mu_c(E)),5), round(float(mu_R(E)),6), round(float(mu_pair(E)),6), round(float(mu_tot(E)),5))
