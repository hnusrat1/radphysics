
import sys, os, time
sys.path.insert(0,'/tmp/work')
import sim
t=time.time()
seg,dep=sim.run(20000,1)
print(time.time()-t, seg['tl'].size, dep['z'].size)
