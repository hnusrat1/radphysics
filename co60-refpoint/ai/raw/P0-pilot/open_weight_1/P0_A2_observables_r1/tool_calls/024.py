import numpy as np
print("numpy ok")
from scipy.interpolate import interp1d
print("interp1d ok")
xcom = np.loadtxt('/tmp/data/xcom_water.txt')
print(f"xcom shape: {xcom.shape}")
estar = np.loadtxt('/tmp/data/estar_water.txt')
print(f"estar shape: {estar.shape}")