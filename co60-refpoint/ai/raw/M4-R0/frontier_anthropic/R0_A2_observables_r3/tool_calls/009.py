
import sys
sys.path.insert(0,'/tmp/work')
import numpy as np, time
import mc
rng=np.random.default_rng(1)
t=time.time()
o=mc.run_batch(200000, rng)
print(time.time()-t, o['nsteps'], o['Tsc'].sum()*mc.OMEGA*mc.DIST**2/200000/mc.V_POI)
