import numpy as np
from scipy.integrate import quad
x=np.loadtxt('/tmp/data/xcom_water.txt',comments='#')
def coeff(E,col):
 k=np.searchsorted(x[:,0],E);k=np.clip(k,1,len(x)-1);a,b=x[k-1,col],x[k,col];e0,e1=x[k-1,0],x[k,0]
 if a>0 and b>0:return np.exp(np.log(a)+np.log(E/e0)/np.log(e1/e0)*np.log(b/a))
 return a+(b-a)*(E-e0)/(e1-e0)
def Tfrac(E):
 eps=E/.51099895
 def f(mu):
  q=1/(1+eps*(1-mu));return q*q*(q+1/q-(1-mu*mu))
 norm=quad(f,-1,1)[0]; remain=quad(lambda mu:f(mu)/(1+eps*(1-mu)),-1,1)[0]
 return 1-remain/norm
# weighted energies from p/s spectra
lines=[(1.1732,.3607301346),(1.3325,.368286654)]
binenergies=np.arange(28)*.05+.025
specs=[lines,[(e,w) for e,w in zip(binenergies,sh) if w>0]]
procTot=np.zeros(4)
for spec in specs:
 for E,w in spec:
  ic=coeff(E,2); photo=coeff(E,3);pair=coeff(E,4)+coeff(E,5)
  procTot[0]+=w*ic*E*Tfrac(E)
  procTot[1]+=w*photo*E
  procTot[2]+=w*pair*max(0,E-1.021998)
print(procTot,procTot/procTot.sum(),Tfrac(1.25))