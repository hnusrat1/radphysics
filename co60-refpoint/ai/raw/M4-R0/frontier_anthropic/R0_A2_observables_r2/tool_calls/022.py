
import time, sys
sys.path.insert(0,'/tmp')
import numpy as np
import phys
from scipy.integrate import quad
t0=time.time()
print(phys.tail_fast(np.array([0.1,1.0]),0.01), 'time',time.time()-t0)
v,_=quad(lambda e: e*phys.moller_dsde(1.0,e,1), 0.01,0.5, limit=500)
print(phys.NeW*v)
