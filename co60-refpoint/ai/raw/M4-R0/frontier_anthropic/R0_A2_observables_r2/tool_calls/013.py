
src=open('/tmp/mc.py').read()
src=src.replace("a=np.where(np.abs(din[:,0])<0.9,np.array([1,0,0]),np.array([0,1,0]))",
                "a=np.zeros_like(din); a[:,0]=(np.abs(din[:,0])<0.9); a[:,1]=(np.abs(din[:,0])>=0.9)")
open('/tmp/mc.py','w').write(src)
import sys, time
sys.path.insert(0,'/tmp')
import numpy as np
import mc as M
def new_tallies():
    return dict(fl_E=np.zeros((2,M.NEB)), fl_cos=np.zeros((2,10)), W_int=np.zeros((2,4)), W_etr=np.zeros((2,4)),
                slab_TL=np.zeros((2,M.NSLAB)), slab_K=np.zeros((2,M.NSLAB)),
                recs=dict(T=[],cos=[],w=[],flag=[],proc=[]))
t=new_tallies()
t0=time.time()
M.run(20000,1,t)
print('time',time.time()-t0)
print(t['fl_E'].sum(axis=1), t['W_int'])
print(M.OMEGA)
