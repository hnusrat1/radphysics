import numpy as np
x=np.loadtxt('/tmp/data/xcom_water.txt',comments='#')
for e in [1.1732,1.25,1.3325]:
 i=np.searchsorted(x[:,0],e);lo=x[i-1];hi=x[i]; q=np.log(e/lo[0])/np.log(hi[0]/lo[0]); v=[]
 for col in range(1,6):
  v.append(np.exp(np.log(lo[col])*(1-q)+np.log(hi[col])*q) if lo[col]>0 and hi[col]>0 else 0)
 v=np.array(v); print(e,v, v/v.sum())