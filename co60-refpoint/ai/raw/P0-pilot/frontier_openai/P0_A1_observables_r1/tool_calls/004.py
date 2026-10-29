import numpy as np
m=.51099895
for E in [2,.8]:
 mu=np.linspace(-1,1,1000001); ep=E/(1+E/m*(1-mu));r=ep/E;w=r*r*(r+1/r-(1-mu**2));T=E-ep
 print(E, np.trapz(w*T,mu)/np.trapz(w,mu), np.trapz(w[T<.01],mu[T<.01])/np.trapz(w,mu))