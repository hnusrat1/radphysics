
s=open('/tmp/work/depthk.py').read()
s=s.replace("dep=np.maximum(dE-Tsum,0)","dep=dE-Tsum")
open('/tmp/work/depthk.py','w').write(s)
import sys, importlib, time, pickle
sys.path.insert(0,'/tmp/work')
import numpy as np
import sources, depthk
from phys import mu_all
seg=pickle.load(open('/tmp/work/photon_tallies.pkl','rb'))['seg']
Phi_p=float(np.exp(-mu_all(np.array([2.0]))[4][0]*10))
rng=np.random.default_rng(5)
cat=sources.build_sources(seg,4000000,rng,Phi_p)
t=time.time()
rng2=np.random.default_rng(7)
hist=depthk.transport_depth(cat['T'],cat['u'],cat['w'],rng2)
print("time",time.time()-t, "sum hist",hist.sum(), "sum expected", (cat['w']*cat['T']).sum())
pickle.dump(dict(hist=hist,ZB=depthk.ZB),open('/tmp/work/kern.pkl','wb'))
