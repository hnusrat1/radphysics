import sys, os
sys.path.insert(0,'/tmp/work'); os.chdir('/tmp/work')
import numpy as np
import mcbeam_debug as mb
r=mb.photon_mc(50000, rng=np.random.default_rng(42), store_src=False)
print('done, niter', r['niter'])
