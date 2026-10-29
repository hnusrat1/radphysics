
code = r'''
import numpy as np, sys
sys.path.insert(0,'/tmp/work')
import xs
RE=2.8179403e-13; ME=0.51099895; NA=6.02214076e23
NEW=NA*0.55509   # electrons per g water
K0=2*np.pi*RE**2*ME*NA
DCUT=0.01
def Fx(x,tau):
    return 1/x**2+1/(1-x)**2+(tau/(tau+1))**2-(2*tau+1)/((tau+1)**2*x*(1-x))
# Moller inverse CDF table for delta energies (eps in [DCUT, T/2])
TG=np.exp(np.linspace(np.log(0.02),np.log(2.0),120))
NQD=200
QD=(np.arange(NQD)+0.5)/NQD
INVD=np.zeros((len(TG),NQD)); SIGD=np.zeros(len(TG))
for i,T in enumerate(TG):
    tau=T/ME; b2=1-1/(1+tau)**2
    x=np.linspace(DCUT/T,0.5,4001)
    f=x*Fx(x,tau)   # eps dsigma/deps * dx relation: dsig/dx = K/b2 * F/T ... use x-space
    # sigma = int dsig/deps deps = int (K0/b2)/T^2 F(x) T dx = K0/b2/T int F dx ; use dsig/dx ~ F(x)
    fx=Fx(x,tau)
    cdf=np.concatenate([[0],np.cumsum(0.5*(fx[1:]+fx[:-1])*np.diff(x))])
    SIGD[i]=NEW*K0/b2/T*cdf[-1]   # per cm (MeV-independent normalization: K0 has MeV cm2/mol)
    cdfn=cdf/cdf[-1]
    INVD[i]=np.interp(QD,cdfn,x*T)
LTG=np.log(TG)
def sig_delta(T):
    T=np.asarray(T,float)
    lt=np.log(np.clip(T,TG[0],TG[-1]))
    return np.interp(lt,LTG,SIGD)*(T>=0.02)
def sample_delta_eps(T,u):
    lt=np.log(np.clip(T,TG[0],TG[-1]))
    x=(lt-LTG[0])/(LTG[-1]-LTG[0])*(len(TG)-1)
    i0=np.clip(np.floor(x).astype(int),0,len(TG)-2); f=x-i0
    q=u*NQD-0.5; q0=np.clip(np.floor(q).astype(int),0,NQD-2); g=np.clip(q-q0,0,1)
    a=INVD[i0,q0]*(1-g)+INVD[i0,q0+1]*g
    b=INVD[i0+1,q0]*(1-g)+INVD[i0+1,q0+1]*g
    return a*(1-f)+b*f
'''
open('/tmp/work/mell.py','w').write(code)
import sys; sys.path.insert(0,'/tmp/work')
import numpy as np, mell
print(mell.SIGD[::20])
# check: delta-ray production per cm above 10 keV at 1 MeV and check total Moller S vs Berger
T=1.0
print(mell.sig_delta(np.array([0.1,0.5,1.0])))
