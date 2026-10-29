import numpy as np
x=np.loadtxt('/tmp/data/xcom_water.txt')
for e in [1.1732,1.25,1.3325]:
 i=np.argmin(abs(x[:,0]-e));print(x[i])