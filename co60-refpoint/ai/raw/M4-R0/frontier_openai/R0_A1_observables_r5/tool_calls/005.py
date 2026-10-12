import numpy as np
from scipy.integrate import quad
m=.51099895
edges=np.array([.01,.05,.1,.15,.2,.25,.3,.35,.4,.45,.5,.55,.6,.65,.7,.75,.8,.85,.9,.95,1,1.05,1.1,1.15,1.2,1.4])
def pdf(T,E):
 if T<=0 or T>=2*E*E/(m+2*E): return 0
 c=1-m*T/(E*(E-T)); q=(E-T)/E
 return np.pi*q*q*(q+1/q-(1-c*c))*m/(E-T)**2
for E in [.6,.8,1.0]:
 v=np.array([quad(lambda t:pdf(t,E),a,min(b,2*E*E/(m+2*E)))[0] if a<2*E*E/(m+2*E) else 0 for a,b in zip(edges[:-1],edges[1:])]); print(E,np.round(v/v.sum(),4))