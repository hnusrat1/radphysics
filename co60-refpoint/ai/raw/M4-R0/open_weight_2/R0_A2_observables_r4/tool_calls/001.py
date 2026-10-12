import numpy as np
from scipy.interpolate import interp1d

# Load XCOM water data
xcom = np.loadtxt('/tmp/data/xcom_water.txt')
print("XCOM columns:", xcom.shape[1])
print("Columns description:")
with open('/tmp/data/xcom_water.txt', 'r') as f:
    print(f.readline())

print("\nFirst few rows:")
for i in range(5):
    print(xcom[i])

print("\nEnergy range:", xcom[:, 0].min(), "to", xcom[:, 0].max(), "MeV")
