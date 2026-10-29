
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
