import numpy as np
estar = np.loadtxt('/tmp/data/estar_water.txt')
print("ESTAR shape:", estar.shape)
lines = open('/tmp/data/estar_water.txt').readlines()
print(lines[0].strip())
print(estar[0:5])
