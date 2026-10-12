import numpy as np
E=2.;m=.51099895; u=np.linspace(-1,1,1000001); ep=E/(1+E/m*(1-u)); r=ep/E; w=r*r*(r+1/r-(1-u*u)); mean=np.trapezoid(w*(1-ep/E),u)/np.trapezoid(w,u); print(mean,mean*0.048903)