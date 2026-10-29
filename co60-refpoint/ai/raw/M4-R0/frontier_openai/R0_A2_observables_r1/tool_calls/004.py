import numpy as np
from scipy.integrate import quad
edges=np.array([.01,.05,.1,.15,.2,.25,.3,.35,.4,.45,.5,.55,.6,.65,.7,.75,.8,.85,.9,.95,1,1.05,1.1,1.15,1.2,1.4])
ctedges=np.linspace(-1,1,11)
m=.51099895
# quadrature over mu KN, map electron T and its angle cos

def dist(E, bins=edges):
 mus=np.linspace(-1,1,200001)
 ep=E/(1+E/m*(1-mus)); r=ep/E
 kn=r*r*(r+1/r-(1-mus**2))
 T=E-ep
 p=np.sqrt(T*(T+2*m))
 c=np.divide(E-ep*mus,p,out=np.ones_like(T),where=p>1e-10)
 h=np.histogram(T,bins=bins,weights=kn)[0]; hd=np.histogram(c,bins=ctedges,weights=kn)[0]
 return h/h.sum(),hd/hd.sum(),np.sum(T*kn)/kn.sum()
for E in [1.1732,1.3325,.8,.4]:
 a,b,av=dist(E);print(E,av, a.round(3),b.round(3))