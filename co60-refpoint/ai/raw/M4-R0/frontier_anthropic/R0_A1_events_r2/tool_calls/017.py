
import subprocess
t=r'''
import numpy as np, sim as S, time
from phys import *
t0=time.time()
C=S.run(600000, chunk=100000)
print('cands',len(C), time.time()-t0)
w=C[:,1]; ty=C[:,0]; E=C[:,2]
print('total weight', w.sum())
for k in [0,1,2,3]:
    m=ty==k; print(k, m.sum(), w[m].sum()/w.sum(), w[m].max()/w.sum())
np.save('C.npy',C)
'''
open('final_run.py','w').write(t)
r=subprocess.run(['python','final_run.py'],capture_output=True,text=True,cwd='.'); print(r.stdout,r.stderr[-2000:])
