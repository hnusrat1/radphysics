import numpy as np
est=np.loadtxt('/tmp/data/estar_water.txt',comments='#')
# log interpolate collision SP
mid=(ee[:-1]+ee[1:])/2
Sc=np.exp(np.interp(np.log(mid),np.log(est[:,0]),np.log(est[:,1])))
print(np.dot(v,Sc),np.dot(v,Sc)*1.602176634e-10)
print(Sc)
