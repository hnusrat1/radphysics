import numpy as np
x=np.loadtxt('/tmp/data/xcom_water.txt',comments='#')
for E in [1.1732,1.3325]:
 row=np.array([np.interp(np.log(E),np.log(x[:,0]),np.log(x[:,i])) for i in range(1,8)])
 print(E,np.exp(row))