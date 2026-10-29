import numpy as np
from scipy.integrate import quad
edges=np.array([.01,.05,.1,.15,.2,.25,.3,.35,.4,.45,.5,.55,.6,.65,.7,.75,.8,.85,.9,.95,1,1.05,1.1,1.15,1.2,1.4])
m=.51099895

def kn_pdf(T,E):
    if T<=0 or T>=2*E*E/(m+2*E): return 0
    c=1-m*T/(E*(E-T))
    c=np.clip(c,-1,1)
    q=(E-T)/E
    sin2=1-c*c
    dsdo=.5*q*q*(q+1/q-sin2)
    return 2*np.pi*dsdo*m/(E-T)**2
for E in [1.1732,1.3325]:
    vals=np.array([quad(lambda t:kn_pdf(t,E),a,min(b,2*E*E/(m+2*E)))[0] if a<2*E*E/(m+2*E) else 0 for a,b in zip(edges[:-1],edges[1:])])
    print(E, vals.sum(), vals/vals.sum(), np.dot(vals,(edges[:-1]+edges[1:])/2)/vals.sum())