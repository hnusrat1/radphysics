
import sys; sys.path.insert(0,'/tmp/work')
import numpy as np, final, time, json
t=time.time()
out,B=final.compute(100000,11,Nel=20000)
print(time.time()-t)
for k,v in out.items():
    if isinstance(v,list): print(k, np.round(v,4).tolist())
    elif isinstance(v,dict): print(k, {a:round(b,4) for a,b in v.items()})
    else: print(k, v)
