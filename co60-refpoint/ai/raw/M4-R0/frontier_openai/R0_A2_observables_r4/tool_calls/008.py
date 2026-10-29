import numpy as np
from scipy.interpolate import interp1d
from scipy.integrate import quad
xx=np.loadtxt('/tmp/data/xcom_water.txt')
p=np.zeros(28); p[23]=.5*np.exp(-.06514178121359064*5);p[26]=.5*np.exp(-.061047572748315*5)
s=np.array([.005,.006,.008,.009,.009,.009,.009,.008,.008,.007,.007,.006,.006,.005,.005,.005,.004,.004,.004,.004,.003,.003,.003,.003,.003,.002,.002,.002]);s*=.14/s.sum()
edges=np.arange(0,1.4001,.05); Ecen=(edges[:-1]+edges[1:])/2
# interpolation xcom sig process 1 coherent 2 inc 3 photo 4-5 pair
sig=np.zeros((28,5))
for j,col in enumerate([1,2,3,4,5]):
 sig[:,j]=np.exp(np.interp(np.log(np.maximum(Ecen,xx[0,0])),np.log(xx[:,0]),np.log(np.maximum(xx[:,col],1e-300))))
# primary separate
ps=np.array([.5*np.exp(-.06514178121359064*5),0,.5*np.exp(-.061047572748315*5)])
Eprimary=np.array([1.1732,1.1732,1.3325])
# use category interaction rate based local spectral fluence
phis=np.r_[p,s]
Es=np.r_[np.where(p>0,Ecen, Ecen),Ecen]
# collect only nonzero
cat=np.zeros(5); trans=np.zeros(5); rate=0; transsum=0

def compfrac(E):
 a=E/.51099895
 def f(mu):
  q=1/(1+a*(1-mu));return .5*q*q*(q+1/q-(1-mu*mu))
 den=quad(f,-1,1,epsabs=1e-10)[0]
 return quad(lambda m:f(m)*(1-1/(1+a*(1-m))),-1,1,epsabs=1e-10)[0]/den
# exact prim + scatter bin spectrum
for E,fl in [(1.1732,p[23]),(1.3325,p[26])]:
 vals=[np.exp(np.interp(np.log(E),np.log(xx[:,0]),np.log(np.maximum(xx[:,i],1e-300)))) for i in range(1,6)]
 # categories coh,comp,photo,pair total
 transfer=[0,vals[1]*E*compfrac(E), vals[2]*E, (vals[3]+vals[4])*max(E-1.022,0)]
 cat[:4]+=fl*np.array([vals[0],vals[1],vals[2],vals[3]+vals[4]])
 trans[:4]+=fl*np.array(transfer)
for E,fl in zip(Ecen,s):
 if fl==0 or E<.005:continue
 vals=[np.exp(np.interp(np.log(E),np.log(xx[:,0]),np.log(np.maximum(xx[:,i],1e-300)))) for i in range(1,6)]
 transfer=[0,vals[1]*E*compfrac(E), vals[2]*E, (vals[3]+vals[4])*max(E-1.022,0)]
 cat[:4]+=fl*np.array([vals[0],vals[1],vals[2],vals[3]+vals[4]])
 trans[:4]+=fl*np.array(transfer)
print('fluences',p.sum(),s.sum(),p.sum()+s.sum())
print('cat',cat,cat/cat.sum())
print('trans',trans,trans/trans.sum(),trans.sum())
print('Kcol dose coef MeV cm2/g',trans.sum(),'Gy',trans.sum()*1.602176634e-10)
print('primary energy transfer coef')
# means spectra
print('meanE photons',(np.dot(p,Eprimary[0:2].tolist()+[] ) if False else 0))
Etot=np.dot(p,Ecen)+np.dot(s,Ecen);print('mean photon',Etot/(p.sum()+s.sum()),'energy scat',np.dot(s,Ecen)/s.sum())
