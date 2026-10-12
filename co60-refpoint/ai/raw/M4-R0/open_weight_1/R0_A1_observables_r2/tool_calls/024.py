import sys, os
sys.path.insert(0, os.getcwd())
import numpy as np
import importlib, mc_photon
importlib.reload(mc_photon)
rng=np.random.default_rng(3); N=200000
p0=rng.uniform([-15,-15,0],[15,15,30],(N,3))
th=rng.uniform(0,2*np.pi,N); ph=np.arccos(rng.uniform(-1,1,N))
d=np.stack([np.sin(ph)*np.cos(th), np.sin(ph)*np.sin(th), np.cos(ph)],axis=1)
p1=p0+d*rng.exponential(15.0,N)[:,None]
L=mc_photon.seg_in_cyl_vec(p0,p1)
print('random cube hits', (L>0).sum(), 'of', N, 'frac', (L>0).mean(), 'meanL', L[L>0].mean() if (L>0).any() else 0)
print('onaxis', mc_photon.seg_in_cyl_vec(np.array([[0,0,4.95]]),np.array([[0,0,5.05]])))