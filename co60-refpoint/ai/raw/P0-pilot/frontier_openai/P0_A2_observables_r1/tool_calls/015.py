import numpy as np
x=np.loadtxt('/tmp/data/xcom_water.txt')
def interp(E,col): return np.exp(np.interp(np.log(E),np.log(x[:,0]),np.log(np.maximum(x[:,col],1e-300))))
def tf(E):
 m=.51099895; u=np.linspace(-1,1,30001); ep=E/(1+E/m*(1-u)); r=ep/E; w=r*r*(r+1/r-(1-u*u));return np.trapezoid(w*(1-ep/E),u)/np.trapezoid(w,u)
def mutr(E):
 c=interp(E,2); ph=interp(E,3); pair=interp(E,4)+interp(E,5); return c*tf(E)+ph+pair*max(E-1.0219979,0)/E
edges=np.arange(0,1.40001,.05); mids=(edges[:-1]+edges[1:])/2
shape=np.exp(-.5*((mids-.78)/.42)**2)*(0.35+.65*mids/1.4); spec=shape/shape.sum()*.19
phif=.6108
energycoeff=phif*2*mutr(2)+sum(spec[j]*mutr(mids[j])*mids[j] for j in range(28))+.05*1.7*mutr(1.7)
print(energycoeff,energycoeff*1.602176634e-7, 'primary',phif*2*mutr(2))