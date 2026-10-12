
import numpy as np, time
rng=np.random.default_rng(1)
me=0.51099895; re=2.8179403e-13; ne_w=3.3428e23  # electrons per cm^3 (rho=1), water
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
# checks
c=np.linspace(-1,1,200001)
for E in [0.1,1.1732,1.3325]:
    num=2*np.pi*np.trapezoid(dsig(E,c),c)
    print(E,num,KN_sigma(E), ne_w*KN_sigma(E), mutot(E))
# check max of f
k=1.0
f=lambda E,c: dsig(E,c)/(0.5*re**2)
for E in [0.01,0.1,0.5,1.33]:
    print(E, f(E,c).max(), c[np.argmax(f(E,c))])
