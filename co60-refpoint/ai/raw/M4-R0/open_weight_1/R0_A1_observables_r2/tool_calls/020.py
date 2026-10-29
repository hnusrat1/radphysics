import sys, os
sys.path.insert(0, os.getcwd())
import mc_photon as mcp
import numpy as np
rng=np.random.default_rng(3)
N=200000
p0=rng.uniform([-15,-15,0],[15,15,30],(N,3))
th=rng.uniform(0,2*np.pi,N); ph=np.arccos(rng.uniform(-1,1,N))
d=np.stack([np.sin(ph)*np.cos(th), np.sin(ph)*np.sin(th), np.cos(ph)],axis=1)
s=rng.exponential(15.0,N)
p1=p0+d*s[:,None]
L=mcp.seg_in_cyl_vec(p0,p1)
print('hits', (L>0).sum(), 'of', N, 'frac', (L>0).mean())
print('mean L among hits', L[L>0].mean())
# expected ~ S/4 * <len> /V = 7.54/4*15/27000 = 0.00105