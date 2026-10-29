import numpy as np
x = np.loadtxt('/tmp/data/xcom_water.txt')
print("loaded xcom, shape:", x.shape)
y = np.loadtxt('/tmp/data/estar_water.txt')
print("loaded estar, shape:", y.shape)
print("First energy in xcom:", x[0,0])
print("Last energy in xcom:", x[-1,0])
print("First energy in estar:", y[0,0])
print("Last energy in estar:", y[-1,0])