
import sys, time, numpy as np
sys.path.insert(0,'/tmp/work')
import mc
t=time.time()
res=mc.run(20000, [0.5,5.0], np.random.default_rng(2), store_for=5.0)
print(time.time()-t, res['arr']['E'].size, res['dscat'])
