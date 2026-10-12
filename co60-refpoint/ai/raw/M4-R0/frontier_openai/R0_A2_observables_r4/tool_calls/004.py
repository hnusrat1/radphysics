import numpy as np
x=np.loadtxt('/tmp/data/xcom_water.txt')
for E in [1.1732,1.3325]:
 vals=np.exp(np.interp(np.log(E),np.log(x[:,0]),np.log(x[:,1:])))
 print(E,vals)
 print('prim',np.exp(-vals[-2]*5))