
src=open('/tmp/work/mc3.py').read()
src=src.replace("DEPTHS=[0.25,0.5,0.75,1.0,5.0]\nDET4=4","DEPTHS=[0.1,0.25,0.5,0.75,1.0,1.5,2.0,2.5,3.0,4.0,5.0,6.0,7.0]\nDET4=10")
open('/tmp/work/mc3.py','w').write(src)
import sys; sys.path.insert(0,'/tmp/work')
import importlib, mc3
importlib.reload(mc3)
print(mc3.DEPTHS, mc3.DET4, mc3.DEPTHS[mc3.DET4])
from concurrent.futures import ProcessPoolExecutor
import numpy as np, pickle, time
def job(seed): return mc3.run(4000000,seed)
if __name__=='__main__':
    t=time.time()
    with ProcessPoolExecutor(4) as ex:
        res=list(ex.map(job,[201,202,203,204]))
    print('time',time.time()-t)
    def merge(res):
        out={}
        for k in res[0]:
            vals=[r[k] for r in res]
            if isinstance(vals[0],np.ndarray): out[k]=np.sum(vals,axis=0)
            else: out[k]=sum(vals)
        return out
    SM=merge(res)
    pickle.dump(SM,open('/tmp/work/SM2.pkl','wb'))
    Om=0.009975072683099257
    print(SM['N'], SM['g']*Om*105**2/SM['N'])
