import numpy as np
xc=np.loadtxt('/tmp/data/xcom_water.txt'); es=np.loadtxt('/tmp/data/estar_water.txt')
print(xc.shape, es.shape); print(xc[-5:]); print(es[-5:])
print(open('/tmp/data/xcom_water.txt').read()[-400:])
print(open('/tmp/data/estar_water.txt').read()[-400:])