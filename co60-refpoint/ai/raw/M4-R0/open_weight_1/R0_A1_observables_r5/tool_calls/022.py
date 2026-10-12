import sys, os, time, importlib
sys.path.insert(0,'/tmp/work'); os.chdir('/tmp/work')
import numpy as np
import mcbeam
importlib.reload(mcbeam)
mcbeam.photon_mc.debug = None
t0=time.time()
r=mcbeam.photon_mc(300000, rng=np.random.default_rng(7))
print('time',time.time()-t0)
print('debug', mcbeam.photon_mc.debug)
print('L0_sum',r['L0_sum'])
print('ph_p_tot',r['ph_p_tot'],'ph_s_tot',r['ph_s_tot'])
print('prim ratio',r['ph_p_tot']/r['L0_sum'], 'total ratio',(r['ph_p_tot']+r['ph_s_tot'])/r['L0_sum'])
print('L2p',r['L2p'],'L2s',r['L2s'])
