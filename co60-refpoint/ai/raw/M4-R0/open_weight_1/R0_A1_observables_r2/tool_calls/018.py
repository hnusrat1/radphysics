import sys, os
sys.path.insert(0, os.getcwd())
import mc_photon as mcp
import numpy as np
# simple tests
for (p0,p1,name) in [ (np.array([[0,0,4.95],[0,0,0],[0,0,4.0],[5.5,0,4.95],[0,0,5.0]]), np.array([[0,0,5.05],[0,0,10],[0,0,6.0],[5.5,0,5.05],[15,0,5.0]]), 'onaxis,long,below,same,side') ]:
    L = mcp.seg_in_cyl_vec(p0,p1)
    print(name, L)
# random scattered-like: start inside cylinder moving random direction, step length ~ mfp
rng = np.random.default_rng(0)
P0 = rng.uniform(-0.3,0.3,(20,3)); P0[:,2]=rng.uniform(4.93,5.07,20)
th = rng.uniform(0,2*np.pi,20); ph = np.arccos(rng.uniform(-1,1,20))
d = np.stack([np.sin(ph)*np.cos(th), np.sin(ph)*np.sin(th), np.cos(ph)],axis=1)
s = rng.exponential(8.0,20)
P1 = P0 + d*s[:,None]
L = mcp.seg_in_cyl_vec(P0,P1)
print('random scatter-like L', L)
print('sum', L.sum())
