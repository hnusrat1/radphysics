import numpy as np
from scipy.integrate import quad
edges=np.array([.01,.05,.1,.15,.2,.25,.3,.35,.4,.45,.5,.55,.6,.65,.7,.75,.8,.85,.9,.95,1,1.05,1.1,1.15,1.2,1.4])
me=.51099895
for E in [1.1732,1.3325]:
 mu=np.linspace(-1,1,500001)
 a=E/me; ep=E/(1+a*(1-mu)); ratio=ep/E
 # KN d sigma / dOmega proportional ratio²*(ratio+1/ratio-(1-mu²))
 w=ratio**2*(ratio+1/ratio-(1-mu**2))
 T=E-ep
 hist,_=np.histogram(T,bins=np.r_[edges[0],edges[1:]],weights=w) # accurate enough
 print(E, hist/hist.sum(), 'below',w[T<.01].sum()/w.sum())