
import sys; sys.path.insert(0,'/tmp/work')
import importlib, mcmod, mcrun
importlib.reload(mcmod); importlib.reload(mcrun)
from mcrun import *
import time
t0=time.time()
tal=new_tally(); rng=np.random.default_rng(1)
tal=chunk(20000, rng, tal)
print("time", time.time()-t0, "ncoll", tal['ncoll'], "nh", tal['nh'])
print("arr_tot per src", tal['arr_tot']/tal['nh'], "Phi0 src", Phi0_src)
print("scattered/Phi0 =", tal['arr_tot']/tal['nh']/Phi0_src)
