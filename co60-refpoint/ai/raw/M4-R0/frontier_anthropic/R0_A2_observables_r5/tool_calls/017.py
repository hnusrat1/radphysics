
import sys, time; sys.path.insert(0,'/tmp/work')
import numpy as np
import importlib, test_slab
from mcmod import *
res=[]
for sd in [11,12,13]:
    rng=np.random.default_rng(sd)
    a,b = test_slab.analog_slab(150000, rng)
    n_ = test_slab.nee_slab(150000, rng)
    res.append((a,b,n_))
    print(sd, a, b, n_)
