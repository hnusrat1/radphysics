import numpy as np
from scipy.integrate import quad
edges=np.array([.01,.05,.1,.15,.2,.25,.3,.35,.4,.45,.5,.55,.6,.65,.7,.75,.8,.85,.9,.95,1,1.05,1.1,1.15,1.2,1.4])
# KN discrete quadrature for electron distribution

def recoil(E, n=40000):
 mu=np.linspace(-1,1,n)
 a=E/.51099895
 ep=E/(1+a*(1-mu))
 kn=.5*(ep/E)**2*(ep/E+E/ep-(1-mu**2))
 te=E-ep
 pe=np.sqrt(np.maximum(E**2+ep**2-2*E*ep*mu,0))
 c=np.divide(E-ep*mu,pe,out=np.ones_like(mu),where=pe>1e-10)
 w=kn
 hist=np.histogram(te,bins=edges,weights=w)[0]
 hist=hist.astype(float); hist[0] # energy below .01 dropped
 dirs=np.histogram(c,bins=np.linspace(-1,1,11),weights=w)[0]
 return hist,dirs, np.sum(w*(te>=.01)),np.sum(w*te)/np.sum(w)
hists=[]
for E in [1.1732,1.3325]:
 h,d,n,mean=recoil(E)
 print(E,h/h.sum(),d/d.sum(),mean)
 hists.append(h)
hpri=(hists[0]+hists[1])/2
hpri=hpri/hpri.sum()
print('primary spec',hpri.tolist())
# scattered photons spectra .14 scaled from unscaled list
sc=np.array([.005,.006,.008,.009,.009,.009,.009,.008,.008,.007,.007,.006,.006,.005,.005,.005,.004,.004,.004,.004,.003,.003,.003,.003,.003,.002,.002,.002]);sc*=.14/sc.sum()
photon_edges=np.arange(0,1.4001,.05)
cent=(photon_edges[:-1]+photon_edges[1:])/2
# scattered-parent recoil spectrum weighted photon flux times incoh cross section
hs=[]
for E,flux in zip(cent,sc):
 if E<.01: continue
 h,_,_,_=recoil(E,n=3000)
 # actual event weighting fluence*incoh coefficient rough; get xcom interp
 xx=np.loadtxt('/tmp/data/xcom_water.txt')
 sig=np.exp(np.interp(np.log(E),np.log(xx[:,0]),np.log(xx[:,2])))
 hs.append(h*flux*sig)
hsc=np.sum(hs,axis=0);hsc/=hsc.sum()
print('scatter electron spectrum', hsc.tolist())
# mix at 20% interaction source
htot=.8*hpri+.2*hsc;htot/=htot.sum()
print('total L3',htot.tolist(), 'sum',htot.sum())