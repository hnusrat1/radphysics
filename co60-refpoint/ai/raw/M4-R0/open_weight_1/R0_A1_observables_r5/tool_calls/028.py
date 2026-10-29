import sys, os, importlib
sys.path.insert(0,'/tmp/work'); os.chdir('/tmp/work')
import numpy as np
import mcbeam
importlib.reload(mcbeam)
mcbeam.photon_mc.debug = None
r=mcbeam.photon_mc(300000, rng=np.random.default_rng(7))
d=mcbeam.photon_mc.debug
print('itrkp',d['itrkp'])
print('ntrkp total',d['ntrkp'],'ntrks',d['ntrks'])
