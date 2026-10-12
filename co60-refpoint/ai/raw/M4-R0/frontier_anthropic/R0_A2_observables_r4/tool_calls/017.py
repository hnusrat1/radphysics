
src=open('/tmp/work/mc3.py').read().replace("_tab=np.load('/tmp/work/tables.npz')","_tab={k:v for k,v in np.load('/tmp/work/tables.npz').items()}")
open('/tmp/work/mc3.py','w').write(src)
import sys; sys.path.insert(0,'/tmp/work')
import importlib, mc3, numpy as np, time, pickle
from concurrent.futures import ProcessPoolExecutor
def job(seed): return mc3.run(4000000,seed)
if __name__=='__main__':
    t=time.time()
    with ProcessPoolExecutor(4) as ex:
        res=list(ex.map(job,[101,102,103,104]))
    print('time',time.time()-t)
    def merge(res):
        out={}
        for k in res[0]:
            vals=[r[k] for r in res]
            if isinstance(vals[0],np.ndarray): out[k]=np.sum(vals,axis=0)
            else: out[k]=sum(vals)
        return out
    SM=merge(res)
    pickle.dump(SM,open('/tmp/work/SM.pkl','wb'))
    Om=0.009975072683099257
    F=Om*105**2/SM['N']
    print(SM['N'], (SM['g']*F))
    print('L3 w',SM['L3_w'], SM['L3_wT']/SM['L3_w'])
