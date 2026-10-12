# calculate primary Ktr
import numpy as np,scipy.integrate as si
xx=np.loadtxt('/tmp/data/xcom_water.txt',comments='#')
def ic(E,c):return np.exp(np.interp(np.log(E),np.log(xx[:,0]),np.log(np.maximum(xx[:,c],1e-30))))
def cf(E):
 a=E/.51099895
 def f(m):
  r=1/(1+a*(1-m));return r*r*(r+1/r-(1-m*m))
 d=si.quad(lambda m:f(m),-1,1)[0];n=si.quad(lambda m:f(m)*(1-1/(1+a*(1-m))),-1,1)[0];return n/d
for E,F in [(1.1732,.3610092),(1.3325,.3681599)]:
 mutr=ic(E,2)*cf(E)+ic(E,3)+(ic(E,4)+ic(E,5))*max(0,(E-1.022)/E)
 print(E,mutr,F*E*mutr)
