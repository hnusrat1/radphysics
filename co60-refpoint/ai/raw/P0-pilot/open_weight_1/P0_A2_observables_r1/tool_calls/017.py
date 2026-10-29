import numpy as np
print("test")
xcom = np.loadtxt('/tmp/data/xcom_water.txt')
print(xcom.shape)
estar = np.loadtxt('/tmp/data/estar_water.txt')
print(estar.shape)
print("OK")