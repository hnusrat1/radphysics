
import sys, time, numpy as np
sys.path.insert(0,'/tmp/work')
src=open('/tmp/work/phys.py').read()
src+='''
_Eg=np.geomspace(0.01,1.5,400)
_ETR=mu_tr_c(_Eg)
def ETR(E):
    return np.interp(np.log(np.asarray(E,float)), np.log(_Eg), _ETR)
'''
open('/tmp/work/phys.py','w').write(src)
m=open('/tmp/work/mc.py').read().replace("mu_tr_c(Ep)","ETR(Ep)")
open('/tmp/work/mc.py','w').write(m)
import importlib, phys, mc
importlib.reload(phys); importlib.reload(mc)
t=time.time()
res=mc.run(100000, [0.5,5.0], np.random.default_rng(3), store_for=5.0)
print(time.time()-t, res['arr']['E'].size, res['dscat'])
