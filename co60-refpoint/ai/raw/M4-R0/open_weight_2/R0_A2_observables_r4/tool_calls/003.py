import numpy as np
from scipy.interpolate import interp1d

# Load ESTAR water data
estar = np.loadtxt('/tmp/data/estar_water.txt')
print("ESTAR columns:", estar.shape[1])
with open('/tmp/data/estar_water.txt', 'r') as f:
    for line in f:
        if line.startswith('#'):
            print(line.strip())

print("\nFirst few rows:")
for i in range(5):
    print(estar[i])

print("\nEnergy range:", estar[:, 0].min(), "to", estar[:, 0].max(), "MeV")
