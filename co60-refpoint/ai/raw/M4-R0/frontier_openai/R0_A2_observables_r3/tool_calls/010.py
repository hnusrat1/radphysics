import numpy as np
x=np.loadtxt('/tmp/data/estar_water.txt',comments='#')
for E in [.1,.2,.3,.5,.6,.8,1,1.2,1.3]:
 row=x[np.argmin(abs(x[:,0]-E))]
 print(E,row[:6])
