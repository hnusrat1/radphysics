import numpy as np
me=.51099895
mu=np.linspace(-1,1,500001)
ce=np.arange(-1,1.0001,.2)
for E in [1.1732,1.3325]:
 a=E/me; ep=E/(1+a*(1-mu)); T=E-ep
 w=(ep/E)**2*((ep/E)+(E/ep)-(1-mu**2))
 pz=E-ep*mu; pt=ep*np.sqrt(1-mu**2); cos=pz/np.sqrt(pz*pz+pt*pt)
 h,_=np.histogram(cos[T>.01],bins=ce,weights=w[T>.01]); print(h/h.sum())