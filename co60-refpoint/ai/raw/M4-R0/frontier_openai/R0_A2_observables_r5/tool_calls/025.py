import numpy as np
x=np.loadtxt('/tmp/data/xcom_water.txt',comments='#');
for e in [1.1732,1.25,1.3325,.8,.5,.3]:
 i=np.searchsorted(x[:,0],e); lo=x[i-1];hi=x[i]; v=np.exp(np.log(lo[0])+(np.log(e/lo[0]))/(np.log(hi[0]/lo[0]))*(np.log(hi[1:6]/lo[1:6])));print(e,v, v/sum(v))