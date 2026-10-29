
pipe = r'''
import sys, pickle, json; sys.path.insert(0,'/tmp/work')
import numpy as np
import mcana, mcmod, ana as A, final as F
from mcana import *
from mcmod import *

def merge(t1,t2):
    out={}
    for k in t1:
        a=t1[k]; b=t2[k]
        if isinstance(a,np.ndarray): out[k]=a+b
        elif isinstance(a,(int,float,np.floating,np.integer)): out[k]=a+b
        else: out[k]=a
    return out

t1=pickle.load(open('/tmp/work/tal_run1.pkl','rb'))
t2=pickle.load(open('/tmp/work/tal_run2_lat1.pkl','rb'))
tal=merge(t1,t2)
# profile from run2 only (LAT=1)
prof_tal = t2
nh=tal['nh']
rng=np.random.default_rng(7)
ptal,Pper=A.primary_electron_tally(rng, Ns=400000)
out,aux=F.main(tal, ptal, Pper, nh)
pickle.dump((out,aux),open('/tmp/work/out_main.pkl','wb'))
pickle.dump((tal,ptal),open('/tmp/work/tal_all.pkl','wb'))
print("main done")
for k,v in out.items():
    if isinstance(v,(list,)): print(k, np.round(np.array(v),5).tolist())
    else: print(k, v)
'''
open('/tmp/work/pipe.py','w').write(pipe)
import subprocess
r=subprocess.run(['python3','/tmp/work/pipe.py'],capture_output=True,text=True,cwd='/tmp/work')
print(r.stdout[-6000:], r.stderr[-3000:])
