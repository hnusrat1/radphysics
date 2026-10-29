import numpy as np
x=np.loadtxt('/tmp/data/xcom_water.txt',comments='#')
for e in [1.1732,1.25,1.3325]:
 i=np.searchsorted(x[:,0],e);print(e,x[i-1:i+1,:])