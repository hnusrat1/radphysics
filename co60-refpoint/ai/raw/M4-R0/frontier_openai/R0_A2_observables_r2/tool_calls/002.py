import numpy as np
x=np.loadtxt('/tmp/data/xcom_water.txt',comments='#'); print(x.shape,x[-5:]); e=np.loadtxt('/tmp/data/estar_water.txt',comments='#');print(e.shape,e[-5:])