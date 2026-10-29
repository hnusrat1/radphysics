import numpy as np
x=np.loadtxt('/tmp/data/xcom_water.txt',comments='#'); print(x.shape,x[-5:])