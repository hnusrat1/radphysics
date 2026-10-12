import numpy as np
from numpy.polynomial.legendre import leggauss
x,gw=leggauss(1000);mu=x;E0=[1.1732,1.3325];
for e in E0:
 al=e/.51099895;ep=1/(1+al*(1-mu));T=e*(1-ep);kn=ep**2*(ep+1/ep-(1-mu**2));print(e,(gw*kn*T).sum()/(gw*kn).sum())