
import sys; sys.path.insert(0,'/tmp/work')
import mc3, numpy as np, time
from concurrent.futures import ProcessPoolExecutor
def job2(seed): return mc3.run(1000000,seed)
if __name__=='__main__':
    t=time.time()
    with ProcessPoolExecutor(4) as ex:
        rr=list(ex.map(job2,range(500,508)))
    print(time.time()-t)
    v=np.array([r['g'][4]*0.009975072683099257*105**2/r['N'] for r in rr])
    print(v, v.mean(), v.std()/np.sqrt(len(v)))
