import numpy as np
m=.51099895
for E in [1.1732,1.3325,1.25285]:
 c=np.linspace(-1,1,200001); eps=1/(1+E/m*(1-c)); ds=eps**2*(eps+1/eps-(1-c*c))/2
 T=E*(1-eps)
 mean=np.trapezoid(ds*T,c)/np.trapezoid(ds,c)
 print(E,mean,mean/E)
