import numpy as np
from scipy.integrate import quad
from scipy.interpolate import interp1d
x=np.loadtxt('/tmp/data/xcom_water.txt')
Ee_edges=np.array([.01,.05,.1,.15,.2,.25,.3,.35,.4,.45,.5,.55,.6,.65,.7,.75,.8,.85,.9,.95,1,1.05,1.1,1.15,1.2,1.4])
ph_edges=np.arange(0,1.4001,.05); centers=(ph_edges[:-1]+ph_edges[1:])/2
s=np.array([.005,.006,.008,.009,.009,.009,.009,.008,.008,.007,.007,.006,.006,.005,.005,.005,.004,.004,.004,.004,.003,.003,.003,.003,.003,.002,.002,.002]);s*=.14/s.sum()
p=np.zeros(28);p[23]=.5*np.exp(-.06514178121359064*5);p[26]=.5*np.exp(-.061047572748315*5)

def sigs(E):
 return np.array([np.exp(np.interp(np.log(max(E,0.001)),np.log(x[:,0]),np.log(np.maximum(x[:,i],1e-300)))) for i in range(1,6)])
def cfrac(E):
 a=E/.51099895
 def kn(m):
  q=1/(1+a*(1-m)); return .5*q*q*(q+1/q-(1-m*m))
 den=quad(kn,-1,1)[0]
 return quad(lambda m:kn(m)*(1-1/(1+a*(1-m))),-1,1)[0]/den
cats={};trs={}
for label, spectra in [('primary',[(1.1732,p[23]),(1.3325,p[26])]),('scattered',list(zip(centers,s)))]:
 cc=np.zeros(4);tt=np.zeros(4)
 for E,fl in spectra:
  if fl==0:continue
  co,inc,photo,pn,pe=sigs(E)
  rates=np.array([inc,photo,pn+pe,co])*fl
  transfer=np.array([inc*E*cfrac(E),photo*E,(pn+pe)*max(E-1.022,0),0])*fl
  cc+=rates;tt+=transfer
 cats[label]=cc/cc.sum();trs[label]=tt/tt.sum()
 print(label,'interaction categories',cats[label], 'transfer',trs[label], 'rate',cc.sum())
print('scatter interaction share',(.14*sum(sigs(e)[1:].sum() for e in []) if False else 0))
# by process total rates
rates_by={'primary':0,'scattered':0}
for label,spectra in [('primary',[(1.1732,p[23]),(1.3325,p[26])]),('scattered',list(zip(centers,s)))]:
 rr=0
 for E,fl in spectra:
  if fl: rr+=fl*sigs(E).sum()
 rates_by[label]=rr
print(rates_by, rates_by['scattered']/sum(rates_by.values()))
# electron spectra hist from KN in each incident energy, weights per Compton interaction
def recoil_hist(E,n=12000):
 mu=np.linspace(-1,1,n);a=E/.51099895; ep=E/(1+a*(1-mu))
 kn=.5*(ep/E)**2*(ep/E+E/ep-(1-mu**2))
 te=E-ep; mom=np.sqrt(np.maximum(E*E+ep*ep-2*E*ep*mu,0));cost=np.divide(E-ep*mu,mom,out=np.ones_like(mu),where=mom>1e-12)
 h=np.histogram(te,bins=Ee_edges,weights=kn)[0]
 d=np.histogram(cost,bins=np.linspace(-1,1,11),weights=kn)[0]
 return h,d
hp=np.zeros(25);dp=np.zeros(10);hs=np.zeros(25);ds=np.zeros(10)
for E,fl in [(1.1732,p[23]),(1.3325,p[26])]:
 h,d=recoil_hist(E)
 wt=fl*sigs(E)[1]
 hp+=h*wt;dp+=d*wt
for E,fl in zip(centers,s):
 if E<=.005:continue
 h,d=recoil_hist(E,3000)
 wt=fl*sigs(E)[1]
 hs+=h*wt;ds+=d*wt
hp/=hp.sum();hs/=hs.sum()
share=rates_by['primary']/(rates_by['primary']+rates_by['scattered'])
ht=share*hp+(1-share)*hs;ht/=ht.sum()
meanp=np.dot(hp,(Ee_edges[:-1]+Ee_edges[1:])/2)
meant=np.dot(ht,(Ee_edges[:-1]+Ee_edges[1:])/2)
print('electron share primary',share,'means',meanp,meant)
print('hp',hp.tolist());print('ht',ht.tolist())
print('direction primary', (dp/dp.sum()).tolist(), 'scatter',(ds/ds.sum()).tolist(), 'total',((share*dp/dp.sum()+(1-share)*ds/ds.sum())).tolist())