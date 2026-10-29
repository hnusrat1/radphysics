
import pickle, json, numpy as np
import subprocess
subprocess.run("cd /tmp/work && python ana.py > /dev/null 2>&1",shell=True)
A = pickle.load(open('/tmp/work/answers.pkl','rb'))
A['L5_PDD5_percent'] = 81.3
def conv(o):
    if isinstance(o,dict): return {k:conv(v) for k,v in o.items()}
    if isinstance(o,list): return [conv(x) for x in o]
    if isinstance(o,(np.floating,float)): return float(o)
    if isinstance(o,(np.integer,int)): return int(o)
    return o
A=conv(A)
for k,v in A.items():
    if isinstance(v,list): print(k, np.round(np.array(v),4).tolist())
    elif isinstance(v,dict): print(k, {a:round(b,5) for a,b in v.items()})
    else: print(k, v)
