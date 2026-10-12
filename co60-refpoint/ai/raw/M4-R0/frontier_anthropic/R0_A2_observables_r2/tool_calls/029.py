
src=open('/tmp/mc.py').read()
src=src.replace("        k=kmin+j\n        valid=k<=kmax\n","        k=np.minimum(kmin+j,NSLAB-1)\n        valid=(kmin+j)<=kmax\n")
open('/tmp/mc.py','w').write(src)
import sys, time, pickle
sys.path.insert(0,'/tmp')
import numpy as np
import mc as M
def new_tallies():
    return dict(fl_E=np.zeros((2,M.NEB)), fl_cos=np.zeros((2,10)), W_int=np.zeros((2,4)), W_etr=np.zeros((2,4)),
                slab_TL=np.zeros((2,M.NSLAB)), slab_K=np.zeros((2,M.NSLAB)),
                recs=dict(T=[],cos=[],w=[],flag=[],proc=[]))
t=new_tallies()
t0=time.time(); NTOT=0
for c in range(24):
    M.run(1000000,900+c,t); NTOT+=1000000
    if time.time()-t0>400: break
print('N',NTOT,'time',time.time()-t0)
rec={k:np.concatenate(v) for k,v in t['recs'].items()}
t['recs']=rec; t['N']=NTOT
pickle.dump(t,open('/tmp/tallies2.pkl','wb'))
print('slabK', t['slab_K'].sum(axis=0)[::10].round(5))
