
import sys, importlib, time
sys.path.insert(0,'/tmp/work')
import numpy as np
import sim
t=time.time()
o=sim.run_chunk(200000,11)
print("time",time.time()-t)
print("Fl_sc",o['Fl_sc'],"Kd[:10]",o['Kd'][:10])
print(o['nproc'])
