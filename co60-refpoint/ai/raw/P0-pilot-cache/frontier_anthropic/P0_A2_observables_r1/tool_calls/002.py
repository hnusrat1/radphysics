import numpy as np
xc=np.loadtxt('/tmp/data/xcom_water.txt'); es=np.loadtxt('/tmp/data/estar_water.txt')
print(xc[-15:]); print(es[-10:])
print(np.interp(2.0,xc[:,0],xc[:,6]))
import numpy as np
def ix(E,col): 
    return np.exp(np.interp(np.log(E),np.log(xc[:,0]),np.log(xc[:,col])))
for c in range(1,8): print(c, ix(2.0,c), ix(1.0,c))