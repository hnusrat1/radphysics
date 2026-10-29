import numpy as np
from numpy.polynomial.legendre import leggauss
edges=np.array([.01,.05,.1,.15,.2,.25,.3,.35,.4,.45,.5,.55,.6,.65,.7,.75,.8,.85,.9,.95,1,1.05,1.1,1.15,1.2,1.4])
s=np.array([.00005,.00015,.00035,.0007,.0012,.002,.003,.004,.0055,.007,.0085,.0095,.010,.0105,.0105,.010,.009,.008,.0065,.005,.0038,.0028,.002,.0015,.0008,.0005,.0003,.0001]); Ecent=np.arange(28)*.05+.025
mur=np.clip(1.25/Ecent,1,10)**0.45
ph_es=[1.1732,1.3325]+[e for e,w in zip(Ecent,s) if w>0]; weights=[.5*.729,.5*.740]+[w*m for w,m in zip(s,mur) if w>0]
cs=np.array([-1,-.8,-.6,-.4,-.2,0,.2,.4,.6,.8,1]); x,gw=leggauss(1000);mu=x; gw=gw

def intg(e,w):
    alpha=e/.51099895; ep=1/(1+alpha*(1-mu)); T=e*(1-ep); kn=ep**2*(ep+1/ep-(1-mu**2)); ww=gw*kn*w
    hist=np.histogram(T,bins=edges,weights=ww)[0]
    p=np.sqrt(T*(T+2*.51099895)); pz=e-e*ep*mu; c=pz/p
    histd=np.histogram(c,bins=cs,weights=ww)[0]
    return hist,histd,ww.sum()
ha=np.zeros(25);hd=np.zeros(10);hp=np.zeros(25);hdp=np.zeros(10)
for e,w in zip(ph_es,weights):
    a,b,c=intg(e,w);ha+=a;hd+=b
for e,w in zip([1.1732,1.3325],[.5*.729,.5*.740]):
    a,b,c=intg(e,w);hp+=a;hdp+=b
print('all',ha/ha.sum());print('p',hp/hp.sum());print('d',hd/hd.sum());print('dp',hdp/hdp.sum())