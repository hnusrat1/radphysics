import numpy as np
D=np.loadtxt('/tmp/data/xcom_water.txt'); e=D[:,0]; i=np.argmin(abs(e-2)); print(D[i]); print(D[i-1:i+2])