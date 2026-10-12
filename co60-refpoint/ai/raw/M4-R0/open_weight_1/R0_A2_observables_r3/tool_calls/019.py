import numpy as np
me=0.51099895
def kn_ratio(E,mu):
    a=E/me; r=1/(1+a*(1-mu)); return r*r*(r+1/r-(1-mu**2))
def compton_electron_dist(E, Nmu=4000):
    alpha=E/me
    mu=np.linspace(-1+1e-9,1-1e-9,Nmu)
    r=1/(1+alpha*(1-mu))
    Ep=E*r
    Ee=E-Ep
    dmu=mu[1]-mu[0]
    w=2*np.pi*kn_ratio(E,mu)*dmu
    dEedmu = np.abs(E*alpha*r**2)
    dPdE = w/dEedmu   # dP/dEe, positive
    dEe = dEedmu*dmu
    tot=np.sum(dPdE*dEe)
    print("tot dP:", tot)
    mean=np.sum(Ee*w)/np.sum(w)
    return Ee,dPdE,dEe,mean
Ee,dPdE,dEe,mean=compton_electron_dist(1.25)
print("mean Te:",mean)
# compare with av_electron_E computed earlier ~0.59
# test electron spectrum normalization on 25 bins
EL=np.array([0.01,0.05,0.1,0.15,0.2,0.25,0.3,0.35,0.4,0.45,0.5,0.55,0.6,0.65,0.7,0.75,
             0.8,0.85,0.9,0.95,1,1.05,1.1,1.15,1.2,1.4])
hist,bins=np.histogram(Ee,bins=EL,weights=dPdE*dEe)
print("hist sum:",hist.sum(), "fractions:",hist/hist.sum())