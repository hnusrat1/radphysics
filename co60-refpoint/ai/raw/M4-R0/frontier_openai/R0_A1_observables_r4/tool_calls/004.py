import numpy as np
from numpy.polynomial.legendre import leggauss
edges=np.array([.01,.05,.1,.15,.2,.25,.3,.35,.4,.45,.5,.55,.6,.65,.7,.75,.8,.85,.9,.95,1,1.05,1.1,1.15,1.2,1.4])
# photon bins and relative scatter fluence
s=np.array([.00005,.00015,.00035,.0007,.0012,.002,.003,.004,.0055,.007,.0085,.0095,.010,.0105,.0105,.010,.009,.008,.0065,.005,.0038,.0028,.002,.0015,.0008,.0005,.0003,.0001])
# ratio interaction coefficients relative μ at ~1.2, use increasing low energies
Ecent=np.arange(28)*.05+.025
mur=np.clip(1.25/Ecent,1,10)**0.45 # rough
# use scatter contributions by bins; each generate event at midpoint, weighted
ph_es=[1.1732,1.3325]+[e for e,w in zip(Ecent,s) if w>0]
weights=[.5*.729,.5*.740]+[w*m for w,m in zip(s,mur) if w>0]
# Primary lines separate weights for parent-primary

def integrate(e, w, out_edges, angle_edges=None):
    x,gw=leggauss(1800); mu=(x+1)/2; gw=gw/2
    alpha=e/.51099895; ep=1/(1+alpha*(1-mu)); T=e*(1-ep)
    kn=ep**2*(ep+1/ep-(1-mu**2))
    ww=gw*kn*w
    idx=np.searchsorted(out_edges,T,side='right')-1
    h=np.bincount(idx[(idx>=0)&(idx<len(out_edges)-1)],weights=ww[(idx>=0)&(idx<len(out_edges)-1)],minlength=len(out_edges)-1)
    if angle_edges is not None:
        p=np.sqrt(T*(T+2*.51099895)); pz=e-e*ep*mu
        c=np.divide(pz,p,out=np.ones_like(pz),where=p>0)
        ii=np.searchsorted(angle_edges,c,side='right')-1
        hh=np.bincount(ii[(ii>=0)&(ii<len(angle_edges)-1)],weights=ww[(ii>=0)&(ii<len(angle_edges)-1)],minlength=len(angle_edges)-1)
        return h,hh,ww.sum()
    return h,ww.sum()
cs=np.array([-1,-.8,-.6,-.4,-.2,0,.2,.4,.6,.8,1])
ha=np.zeros(25); haP=np.zeros(25); hdir=np.zeros(10); hdirP=np.zeros(10); tot=totP=0
for e,w in zip(ph_es,weights):
    h,hd,n=integrate(e,w,edges,cs); ha+=h; hdir+=hd;tot+=n
for e,w in zip([1.1732,1.3325],[.5*.729,.5*.740]):
    h,hd,n=integrate(e,w,edges,cs);haP+=h;hdirP+=hd;totP+=n
print('all spec',ha/ha.sum()); print('primary',haP/haP.sum());print('direction',hdir/hdir.sum());print('dirP',hdirP/hdirP.sum()); print('tot primary/all',totP/tot)
