
import numpy as np, json
rng=np.random.default_rng(12345)
me=0.51099895
NE=6.022e23*10/18.015  # electrons per gram of water
def sKN(E):
    a=E/me
    t1=(1+a)/a**2*(2*(1+a)/(1+2*a)-np.log(1+2*a)/a)
    t2=np.log(1+2*a)/(2*a)
    t3=(1+3*a)/(1+2*a)**2
    return 2*np.pi*2.8179403e-13**2*(t1+t2-t3)  # cm^2 per electron
def mu_parts(E):
    E=np.asarray(E,float)
    inc=sKN(E)*NE
    pe=np.where(E>0.0,2.8e-3*(0.1/E)**3,0)*1.0
    pair=np.where(E>1.022,0.0038*np.clip(E-1.022,0,None)**2,0)
    coh=2e-3*(0.1/E)**2
    return inc,pe,pair,coh
for E in [0.1,0.2,0.5,1.0,1.1732,1.25,1.3325]:
    print(E,[round(x,6) for x in mu_parts(E)])
# check max of KN envelope
Es=np.array([0.05,0.2,1.1732,1.3325]); 
ths=np.linspace(0,np.pi,2001)
for E in Es:
    a=E/me; c=np.cos(ths); Ep=E/(1+a*(1-c)); eps=Ep/E
    f=eps**2*(eps+1/eps-np.sin(ths)**2)
    print(E,f.max())
