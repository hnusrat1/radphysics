
import numpy as np
me=0.51099895; re=2.8179403e-13; alpha=1/137.036; NA=6.02214076e23
def beta2(T): tau=T/me; return tau*(tau+2)/(tau+1)**2
def density_delta(T, C,x0,x1,a,m):
    beta=np.sqrt(beta2(T)); gam=1+T/me; X=np.log10(beta*gam)
    d=np.where(X>=x1,0.0,0.0)
    mid=(X>=x0)&(X<x1)
    d=np.where(mid,4.6052*X-C+a*(x1-X)**m,d)
    d=np.where(X>=x1,4.6052*X-C,d)
    return d
WATER=dict(ZA=0.55509,I=75e-6,C=3.5017,x0=0.2400,x1=2.8004,a=0.09116,m=3.4773,ne=3.3428e23)
AIR  =dict(ZA=0.49919,I=85.7e-6,C=10.5961,x0=1.7418,x1=4.2759,a=0.10914,m=3.2125,ne=3.0062e23)
def Scol_per_mass(T,mat):
    tau=T/me; b2=beta2(T)
    F=1-b2+(tau**2/8-(2*tau+1)*np.log(2))/(tau+1)**2
    delta=density_delta(T,mat['C'],mat['x0'],mat['x1'],mat['a'],mat['m'])
    L=np.log(tau**2*(tau+2)/(2*(mat['I']/me)**2))+F-delta
    return 0.153537*mat['ZA']/b2*L   # MeV cm2/g
def Srad_per_mass(T):
    Z=76.0/ (1)  # sum Z(Z+1) per molecule, water
    tt=T+me
    return 76*NA/18.015*alpha*re**2*tt*(4*np.log(2*tt/me)-4/3)
for T in [0.1,0.5,1.0]:
    print(T, 'water Scol',Scol_per_mass(T,WATER),'air',Scol_per_mass(T,AIR),'rad water',Srad_per_mass(T))
