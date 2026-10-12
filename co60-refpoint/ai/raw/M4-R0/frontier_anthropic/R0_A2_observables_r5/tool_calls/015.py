
import sys, time; sys.path.insert(0,'/tmp/work')
import numpy as np
import importlib, test_slab
from mcmod import *
importlib.reload(test_slab)
rng=np.random.default_rng(5)
t0=time.time()
TLc, TLp = test_slab.analog_slab(40000, rng)
print("analog slab track per src (scat, prim):", TLc, TLp, "time", time.time()-t0)
t0=time.time()
Sn = test_slab.nee_slab(40000, rng)
print("NEE slab scattered track per src:", Sn, time.time()-t0)
