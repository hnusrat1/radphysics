import numpy as np
x=np.loadtxt('/tmp/data/xcom_water.txt')
for E in [1.1732,1.3325]:
 vals=np.array([np.exp(np.interp(np.log(E),np.log(x[:,0]),np.log(x[:,i]))) for i in range(1,7)])
 print(E, vals, 'total',np.exp(np.interp(np.log(E),np.log(x[:,0]),np.log(x[:,6]))), 'prim',np.exp(-np.exp(np.interp(np.log(E),np.log(x[:,0]),np.log(x[:,6])))*5))