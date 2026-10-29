
import time,sys
sys.path.insert(0,'/tmp')
import numpy as np, phys
t0=time.time(); import pickle
t=pickle.load(open('/tmp/tallies.pkl','rb')); print('load',time.time()-t0)
t0=time.time()
for T in np.linspace(0.01,1.4,279): phys.estar(T)
print('estar loop',time.time()-t0)
t0=time.time()
x=phys.moller_dsde(0.5,np.linspace(0.01,0.2,6),1); print('moller',time.time()-t0)
t0=time.time()
from scipy.interpolate import interp1d
import mc
print('import mc',time.time()-t0)
