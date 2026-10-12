import numpy as np
E=2.0; m=.51099895
mu=np.linspace(-1,1,1000001)
ep=E/(1+(E/m)*(1-mu))
r=ep/E
kn=r*r*(r+1/r-(1-mu**2)) # dσ/dΩ const
T=E-ep
edges=np.array([.01,.05,.1,.15,.2,.25,.3,.35,.4,.45,.5,.55,.6,.65,.7,.75,.8,.85,.9,.95,1,1.05,1.1,1.15,1.2,1.4])
idx=np.digitize(T,edges)-1
h=np.bincount(np.clip(idx,0,24),weights=kn,minlength=25)
# treatment below 10 keV excluded; above 1.4 into last bin
h/=h.sum()
print(h, h.sum())
print('means',sum(h*(edges[:-1]+edges[1:])/2))
print('under',np.sum(kn[T<.01])/sum(kn),'above',np.sum(kn[T>1.4])/sum(kn))
