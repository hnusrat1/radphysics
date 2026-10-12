
import numpy as np, time
rng=np.random.default_rng(1)
me=0.51099895; re=2.8179403e-13; ne_w=3.3428e23
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
lines=np.array([1.1732,1.3325])
Eb=np.array([0,0.05,0.1,0.15,0.2,0.25,0.3,0.35,0.4,0.45,0.5,0.55,0.6,0.65,0.7,0.75,0.8,0.85,0.9,0.95,1,1.05,1.1,1.15,1.2,1.25,1.3,1.35,1.4])
cb=np.array([-1,-0.8,-0.6,-0.4,-0.2,0,0.2,0.4,0.6,0.8,1.0])
ebE=np.array([0.01,0.05,0.1,0.15,0.2,0.25,0.3,0.35,0.4,0.45,0.5,0.55,0.6,0.65,0.7,0.75,0.8,0.85,0.9,0.95,1,1.05,1.1,1.15,1.2,1.4])
def rad_S(T, X0=36.08):
    f=np.minimum(0.31*np.maximum(T,1e-4)**0.3,1.0)
    return (T+me)/X0*f
def Scol_bethe(T, I, neg):
    tau=T/me; b2=tau*(tau+2)/(tau+1)**2
    Kc=2*np.pi*re**2*me*neg
    F=1-b2+(tau**2/8-(2*tau+1)*np.log(2))/(tau+1)**2
    return Kc/b2*(np.log(tau**2*(tau+2)/(2*(I/me)**2))+F)
neg_w=6.02214e23*0.55509
neg_a=6.02214e23*0.49919
Iw=75e-6; Ia=85.7e-6
for T in [0.01,0.1,0.3,1.0,10]:
    print(T, Scol_bethe(T,Iw,neg_w), Scol_bethe(T,Ia,neg_a), rad_S(T))
