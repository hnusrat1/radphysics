
import numpy as np, os
os.makedirs('/tmp/work',exist_ok=True)
x=np.loadtxt('/tmp/data/xcom_water.txt')
e=np.loadtxt('/tmp/data/estar_water.txt')
print(x.shape,e.shape)
# solid angle of pyramid: 10x10 at 100 cm, uniform in solid angle
from scipy import integrate
a=0.05
f=lambda ty,tx: (1+tx*tx+ty*ty)**-1.5
Om=integrate.dblquad(f,-a,a,-a,a)[0]
print("Omega sr",Om, 4*np.arctan(25/(100*np.sqrt(25+25+1e4))))
print("mu_tot 2MeV (with coh)",x[x[:,0]==2.0][0])
