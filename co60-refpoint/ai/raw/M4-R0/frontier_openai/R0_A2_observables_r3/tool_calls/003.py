import numpy as np
m=.51099895
for E in [1.1732,1.3325,1.25285]:
 c=np.linspace(-1,1,200001); eps=1/(1+E/m*(1-c)); # KN differential / r_e^2 per solid angle
 ds=eps**2*(eps+1/eps-(1-c*c))/2
 T=E*(1-eps)
 mean=np.trapz(ds*T,c)/np.trapz(ds,c)
 print(E,mean,mean/E)
