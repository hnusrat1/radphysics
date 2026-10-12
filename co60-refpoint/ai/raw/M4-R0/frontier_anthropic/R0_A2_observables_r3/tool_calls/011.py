
driver = r'''
import sys, numpy as np, time, pickle
sys.path.insert(0,'/tmp/work')
import mc
from multiprocessing import Pool
def job(seed_n):
    seed, n = seed_n
    rng = np.random.default_rng(seed)
    return mc.run_batch(n, rng)
if __name__=='__main__':
    nb = int(sys.argv[1]); n = int(sys.argv[2]); tag=sys.argv[3]
    t=time.time()
    with Pool(4) as p:
        res = p.map(job, [(1000*nb+i+int(tag[1:] if tag[1:].isdigit() else 0)*7919, n) for i in range(nb)])
    # combine
    out = {}
    for k in ['Tsc','Tco','Kdep','Tdep','Esum_w']:
        out[k] = sum(r[k] for r in res)
    for k in ['cnt','etr']:
        out[k] = {kk: sum(r[k][kk] for r in res) for kk in res[0][k]}
    for k in ['birth_T','birth_w','birth_proc','birth_cos','birth_sign']:
        out[k] = np.concatenate([r[k] for r in res])
    out['N'] = nb*n
    pickle.dump(out, open(f'/tmp/work/res_{tag}.pkl','wb'))
    print('done', time.time()-t, out['N'])
'''
open('/tmp/work/driver.py','w').write(driver)
import subprocess
r=subprocess.run("cd /tmp/work && python driver.py 4 250000 t0",shell=True,capture_output=True,text=True)
print(r.stdout, r.stderr[-2000:])
