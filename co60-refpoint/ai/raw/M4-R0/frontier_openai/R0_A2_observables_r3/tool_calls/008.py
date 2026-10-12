import numpy as np
f='/tmp/data/xcom_water.txt'
a=np.loadtxt(f,comments='#')
for E in [1.1732,1.3325,1.0,0.5,0.1]:
 logE=np.log(a[:,0]);
 vals=[np.exp(np.interp(np.log(E),logE,np.log(np.maximum(a[:,j],1e-300)))) for j in range(1,7)]
 print(E,vals)