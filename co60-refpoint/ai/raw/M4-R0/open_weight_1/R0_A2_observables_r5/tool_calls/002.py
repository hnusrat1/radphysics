import numpy as np
d = np.loadtxt('/tmp/data/xcom_water.txt', comments='#')
print('xcom shape', d.shape)
print('E range', d[0,0], d[-1,0])
e = np.loadtxt('/tmp/data/estar_water.txt', comments='#')
print('estar shape', e.shape)
print('E range', e[0,0], e[-1,0])
# inspect around 1.1-1.4 MeV for xcom
m = (d[:,0]>=0.9)&(d[:,0]<=1.5)
print(d[m])
print()
# estar around 0.01-1.5 MeV
m = (e[:,0]>=0.01)&(e[:,0]<=1.5)
print(e[m])