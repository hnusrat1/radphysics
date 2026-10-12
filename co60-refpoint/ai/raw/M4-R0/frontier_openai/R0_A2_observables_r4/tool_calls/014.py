import numpy as np
x=np.loadtxt('/tmp/data/estar_water.txt')
for E in [.1,.2,.4,.6,.8,1,1.2]:
 i=np.argmin(abs(x[:,0]-E));print(x[i,:6])