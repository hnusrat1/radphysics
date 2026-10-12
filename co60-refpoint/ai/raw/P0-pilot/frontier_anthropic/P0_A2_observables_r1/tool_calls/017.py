
import sys, importlib, time
sys.path.insert(0,'/tmp/work')
import numpy as np
import electrons
rng=np.random.default_rng(3)
t=time.time()
r=electrons.transport(np.full(2000,1.0),np.tile([[0,0,1.]],(2000,1)),np.ones(2000),np.zeros(2000,bool),rng)
print(time.time()-t, r['it'], "track per electron (cm)",r['tot']/2000, "delta frac",r['delt']/r['tot'], "ndelta per e",r['nd']/2000)
print("Sr/Sc",r['Sr']/r['Sc'])
print("CSDA range of 1 MeV (ESTAR g/cm2 -> cm):", np.interp(1.0,phys.EE,phys.E_[:,4]))
