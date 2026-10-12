import numpy as np
x=np.loadtxt('/tmp/data/xcom_water.txt');
def interp(E,col): return np.exp(np.interp(np.log(E),np.log(x[:,0]),np.log(np.maximum(x[:,col],1e-300))))
def transfrac(E):
 m=.51099895;u=np.linspace(-1,1,30001); ep=E/(1+E/m*(1-u)); r=ep/E; w=r*r*(r+1/r-(1-u*u));return np.trapezoid(w*(1-ep/E),u)/np.trapezoid(w,u)
# all values point
for EE in [.3,.5,.8,1.1,1.3,1.7,2]:
 print(EE,transfrac(EE),[interp(EE,j) for j in range(1,7)])