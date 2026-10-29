import numpy as np
from numpy.polynomial.legendre import leggauss
x,gw=leggauss(1000);mu=x;s=np.array([.00005,.00015,.00035,.0007,.0012,.002,.003,.004,.0055,.007,.0085,.0095,.010,.0105,.0105,.010,.009,.008,.0065,.005,.0038,.0028,.002,.0015,.0008,.0005,.0003,.0001]);es=np.arange(28)*.05+.025
weights=[.5*.729,.5*.740]+[w*np.clip(1.25/e,1,10)**.45 for e,w in zip(es,s) if w>0]; energies=[1.1732,1.3325]+[e for e,w in zip(es,s) if w>0]
def mean(e):
 a=e/.51099895;ep=1/(1+a*(1-mu));T=e*(1-ep); kn=ep**2*(ep+1/ep-(1-mu**2));return (gw*kn*T).sum()/(gw*kn).sum(),(gw*kn).sum()
vals=[mean(e) for e in energies]; print(sum(w*v[1]*v[0] for w,v in zip(weights,vals))/sum(w*v[1] for w,v in zip(weights,vals)))
pweights=[.5*.729,.5*.740]; print(sum(w*v[1]*v[0] for w,v in zip(pweights,vals[:2]))/sum(w*v[1] for w,v in zip(pweights,vals[:2])))