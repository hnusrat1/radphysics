# energy transfer coef per primary/scattered from above spectra
import numpy as np
x=np.loadtxt('/tmp/data/xcom_water.txt')
def sval(E,col):return np.exp(np.interp(np.log(E),np.log(x[:,0]),np.log(np.maximum(x[:,col],1e-300))))
def fcomp(E):
 from scipy.integrate import quad
 a=E/.51099895
 def f(m):
  q=1/(1+a*(1-m));return .5*q*q*(q+1/q-(1-m*m))
 den=quad(f,-1,1)[0]
 return quad(lambda m:f(m)*(1-1/(1+a*(1-m))),-1,1)[0]/den
p=np.zeros(28);p[23]=.5*np.exp(-.06514178121359064*5);p[26]=.5*np.exp(-.061047572748315*5)
s=np.array([.005,.006,.008,.009,.009,.009,.009,.008,.008,.007,.007,.006,.006,.005,.005,.005,.004,.004,.004,.004,.003,.003,.003,.003,.003,.002,.002,.002]);s*=.14/s.sum()
cent=np.arange(.025,1.4,.05)
def trcoef(spec):
 out=0
 for E,fl in spec:
  if not fl:continue
  out+=fl*(sval(E,2)*E*fcomp(E)+sval(E,3)*E+(sval(E,4)+sval(E,5))*max(E-1.022,0))
 return out
print(trcoef([(1.1732,p[23]),(1.3325,p[26])]),trcoef(list(zip(cent,s))))
