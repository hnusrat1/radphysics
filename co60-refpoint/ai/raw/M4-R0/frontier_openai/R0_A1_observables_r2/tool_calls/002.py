import numpy as np
ph_edges=np.array([0,.05,.1,.15,.2,.25,.3,.35,.4,.45,.5,.55,.6,.65,.7,.75,.8,.85,.9,.95,1,1.05,1.1,1.15,1.2,1.25,1.3,1.35,1.4])
w=np.array([.005,.01,.02,.035,.05,.06,.06,.055,.05,.045,.04,.04,.04,.04,.04,.04,.04,.04,.04,.04,.04,.04,.04,.04,.035,.03,.025,.015]); w=w/w.sum()
scat=.1*w
prim=np.zeros(28); prim[23]=.364;prim[26]=.364
# bins e spectrum
Ee=np.array([.01,.05,.1,.15,.2,.25,.3,.35,.4,.45,.5,.55,.6,.65,.7,.75,.8,.85,.9,.95,1,1.05,1.1,1.15,1.2,1.4])
cb=np.linspace(-1,1,20001); dc=cb[1]-cb[0]
def kndist(E):
    alpha=E/.51099895
    ep=1/(1+alpha*(1-cb)); r=ep
    d=r*r*(r+1/r-(1-cb*cb))
    T=E*(1-r)
    p=np.sqrt(np.maximum(E*E+(E*r)**2-2*E*E*r*cb,0))
    c_el=np.divide(E-E*r*cb,p,out=np.ones_like(p),where=p>1e-12)
    hist,_=np.histogram(T,bins=Ee,weights=d)
    # ignore under 10kev, normalized later
    hdir,_=np.histogram(c_el,bins=np.linspace(-1,1,11),weights=d*(T>=.01))
    return hist,hdir,d.sum()
# primary electron spectra
hp=np.zeros(25); hd=np.zeros(10); totp=0
for E, weight in [(1.1732,.5),(1.3325,.5)]:
    h,_,tot=kndist(E); hp += weight*h; totp += weight*h.sum()
    # direction
    # redo
    alpha=E/.51099895; r=1/(1+alpha*(1-cb)); d=r*r*(r+1/r-(1-cb*cb)); T=E*(1-r)
    p=np.sqrt(np.maximum(E*E+(E*r)**2-2*E*E*r*cb,0)); ce=(E-E*r*cb)/np.maximum(p,1e-15)
    hd+=weight*np.histogram(ce,bins=np.linspace(-1,1,11),weights=d*(T>=.01))[0]
hp/=hp.sum(); hd/=hd.sum()
# scattered electron energies approximated scatter spectrum centers; weight with incoherent interaction proxy mass attenuation rising at low E, use broad approx factor
hs=np.zeros(25); hds=np.zeros(10); ts=0
for E, weight in zip((ph_edges[:-1]+ph_edges[1:])/2,scat):
    if E<.01: continue
    h,_,tot=kndist(E)
    # interaction rate weighting extra low energy approx relative to primary ~ 1+(1-E)/1.2 capped
    fac=weight*(.063 + .015/(E+.08))
    hs+=fac*h; ts+=fac*h.sum()
    # direction
    alpha=E/.51099895; r=1/(1+alpha*(1-cb)); d=r*r*(r+1/r-(1-cb*cb)); T=E*(1-r)
    p=np.sqrt(np.maximum(E*E+(E*r)**2-2*E*E*r*cb,0)); ce=(E-E*r*cb)/np.maximum(p,1e-15)
    hds+=fac*np.histogram(ce,bins=np.linspace(-1,1,11),weights=d*(T>=.01))[0]
hs/=hs.sum(); hds/=hds.sum()
print('primary spec',hp.tolist(), 'mean',sum(hp*(Ee[:-1]+Ee[1:])/2), 'ang',hd)
# totals parent weights primary events ~ .728*.063 =.0459, scattered event estimate via sum scatter*mu
ws=np.sum(scat*(.063+.015/(((ph_edges[:-1]+ph_edges[1:])/2)+.08)))
wp=.728*.063
f=ws/(wp+ws)
ht=(1-f)*hp+f*hs
hdt=(1-f)*hd+f*hds
print('fscat',f,'total spec',ht.tolist(),'mean',sum(ht*(Ee[:-1]+Ee[1:])/2),'dir',hdt.tolist())
print('scatter',scat.tolist(), 'sum',scat.sum(), 'mean',sum(scat*(ph_edges[:-1]+ph_edges[1:])/2)/.1)
