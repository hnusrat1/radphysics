import numpy as np
a=np.loadtxt('/tmp/data/estar_water.txt');
for E in [.1,.2,.5,1,2]:
 r=np.interp(E,a[:,0],a[:,2]);c=np.interp(E,a[:,0],a[:,1]); print(E,c,r,r/c)