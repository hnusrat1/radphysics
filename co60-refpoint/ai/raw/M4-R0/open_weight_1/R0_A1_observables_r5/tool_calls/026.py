
import sys, os, time, importlib
sys.path.insert(0,'/tmp/work'); os.chdir('/tmp/work')
import numpy as np
import mcbeam
importlib.reload(mcbeam)
mcbeam.photon_mc.debug = None
r=mcbeam.photon_mc(300000, rng=np.random.default_rng(7))
d=mcbeam.photon_mc.debug
print('s frac <5.1:', d['sfrac'], '/', d['sfracN'], '=', d['sfrac']/d['sfracN'], ' mean s:', d['ssum']/d['sfracN'])
print('mu at 1.1732:', mcbeam.mu_tot_w(np.array([1.1732,1.3325])))
print('expected P(s<5.1):', 1-np.exp(-mcbeam.mu_tot_w(1.1732)*5.1))
print('L0',r['L0_sum'],'ph_p',r['ph_p_tot'],'ph_s',r['ph_s_tot'])
