import numpy as np
xcom = np.loadtxt('/tmp/data/xcom_water.txt')
print(xcom.shape)
print("Rows 0-5, cols:", xcom[0:6])
# Check energy values
print("Energies:", xcom[:,0])
