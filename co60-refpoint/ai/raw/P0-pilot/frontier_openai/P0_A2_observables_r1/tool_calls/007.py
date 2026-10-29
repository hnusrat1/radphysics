import numpy as np
a=np.loadtxt('/tmp/data/estar_water.txt');
for E in [.1,.2,.5,1,2]: print(E,np.interp(E,a[:,0],a[:,5]))