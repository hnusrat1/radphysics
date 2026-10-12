import numpy as np
# scattered photon spectrum fractions in all 28 bins
w=np.array([.003,.004,.007,.012,.018,.025,.032,.040,.047,.052,.058,.062,.066,.070,.073,.073,.070,.066,.060,.052,.043,.033,.023,.014,.008,.004,.002,.001],float); w/=w.sum()
print('wsum',w.sum(), 'mean bin energy', np.dot(w,np.arange(28)*.05+.025))
print('fluence bins', (w*.16).tolist())
# primary weighted exact energy
prim=.737
print('prim each',prim/2)
# mean total using scattered bin midpoint
muP=(1.1732+1.3325)/2
scatter_mean=np.dot(w,np.arange(28)*.05+.025)
print('mean', (prim*muP+.16*scatter_mean)/(prim+.16), 'f sc',.16/(prim+.16), 'energy sc frac',.16*scatter_mean/(prim*muP+.16*scatter_mean))
# primary KN hist integration
me=.51099895; ee=np.array([.01,.05,.1,.15,.2,.25,.3,.35,.4,.45,.5,.55,.6,.65,.7,.75,.8,.85,.9,.95,1,1.05,1.1,1.15,1.2,1.4])
def khist(E):
    mu=np.linspace(-1,1,200001); a=E/me; ep=E/(1+a*(1-mu)); T=E-ep
    r=ep/E; ww=r*r*(r+1/r-(1-mu*mu))
    h,_=np.histogram(T,bins=ee,weights=ww)
    return h/h.sum(), (T*ww).sum()/ww.sum()
p1,m1=khist(1.1732);p2,m2=khist(1.3325); pp=(p1+p2)/2; pp=pp/pp.sum()
# construct scatter parent initial energy via spectrum sampling in each photon bin, KN scatter angles
rng=np.random.default_rng(1)
# use integration random sample sampled photon energy per bins based fluence*approx attenuation coeff; use bins random uniform
N=2500000
idx=rng.choice(28,size=N,p=w)
E=(idx+rng.random(N))*.05
# sample mu from rejection via grid approximate probability distribution condition each E is hard; sample isotropic proposal and weighting
mu=rng.uniform(-1,1,N); a=E/me; ep=E/(1+a*(1-mu));r=ep/E
weights=r*r*(r+1/r-(1-mu*mu))
T=E-ep
hist,_=np.histogram(T,bins=ee,weights=weights)
ps=hist/hist.sum()
# combine 80% primary,20% scattered, renormalized >.01 distributions
pall=.8*pp+.2*ps;pall/=pall.sum()
print('primary',pp.tolist(), 'mean',sum(pp*((ee[:-1]+ee[1:])/2)))
print('scatter',ps.tolist(), 'mean',sum(ps*((ee[:-1]+ee[1:])/2)))
print('all',pall.tolist(), 'mean',sum(pall*((ee[:-1]+ee[1:])/2)))
# L4 spectrum weights
w4=np.array([.02,.045,.06,.07,.075,.075,.075,.07,.065,.06,.055,.05,.045,.04,.035,.03,.025,.02,.015,.012,.009,.006,.004,.002,.002]);w4=w4/w4.sum()
print('w4 sum',w4.sum(),'mean',np.dot(w4,(ee[:-1]+ee[1:])/2),'array total', (w4*.017).tolist())