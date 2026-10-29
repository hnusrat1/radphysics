
import sys, time, pickle; sys.path.insert(0,'/tmp/work')
import numpy as np, importlib
import mcana; importlib.reload(mcana)
from mcana import *
t0=time.time()
tal=new_tally(); rng=np.random.default_rng(100)
nchunks=0
while time.time()-t0<420 and nchunks<60:
    tal=chunk(100000, rng, tal); nchunks+=1
print("chunks",nchunks,"time",time.time()-t0,"nh",tal['nh'],"nseg",tal['nseg'])
pickle.dump(tal,open('/tmp/work/tal_run1.pkl','wb'))
print("scat fluence/Phi0", tal['TLV']/tal['nh']/VCYL/Phi0_src)
