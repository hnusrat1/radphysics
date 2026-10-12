
import sys, time, pickle; sys.path.insert(0,'/tmp/work')
import numpy as np, importlib
import mcana
mcana.LAT = 1.0
from mcana import *
t0=time.time()
tal2=new_tally(); rng=np.random.default_rng(999)
nch=0
while time.time()-t0<400 and nch<70:
    tal2=chunk(100000, rng, tal2); nch+=1
print("chunks",nch,"nh",tal2['nh'],"nseg",tal2['nseg'],"time",time.time()-t0)
pickle.dump(tal2,open('/tmp/work/tal_run2_lat1.pkl','wb'))
print("scat/Phi0 run2", tal2['TLV']/tal2['nh']/VCYL/Phi0_src, " run1 0.42146")
