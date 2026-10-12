import numpy as np
from scipy.integrate import quad
import sys, os
sys.path.insert(0, os.getcwd())
import mc_photon as mcp

def mean_transfer(E):
    f1 = lambda th: mcp.kn_dsig_dOm(E, np.cos(th))*np.sin(th)*(E - E/(1+(E/mcp.mec2)*(1-np.cos(th))))
    f0 = lambda th: mcp.kn_dsig_dOm(E, np.cos(th))*np.sin(th)
    a1,_ = quad(f1,0,np.pi,epsabs=1e-30,limit=400)
    a0,_ = quad(f0,0,np.pi,epsabs=1e-30,limit=400)
    return a1/a0
for E in [1.1732,1.3325,1.25]:
    print(E, 'mean transfer', mean_transfer(E))
