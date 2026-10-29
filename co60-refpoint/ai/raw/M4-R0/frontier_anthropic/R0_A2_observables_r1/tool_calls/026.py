
a=open('/tmp/work/ana.py').read()
a=a.replace("    cphi=(E-Ep*mug)/pe\n","    with np.errstate(invalid='ignore',divide='ignore'):\n        cphi=np.where(pe>1e-12,(E-Ep*mug)/np.maximum(pe,1e-12),1.0)\n    cphi=np.clip(cphi,-1,1)\n")
open('/tmp/work/ana.py','w').write(a)
m=open('/tmp/work/main.py').read()
m+='''
def conv(o):
    if isinstance(o,dict): return {k:conv(v) for k,v in o.items()}
    if isinstance(o,(list,tuple)): return [conv(v) for v in o]
    if isinstance(o,np.ndarray): return o.tolist()
    if isinstance(o,(np.floating,np.integer)): return float(o)
    return o
json.dump(conv(out),open('/tmp/work/results.json','w'),indent=1)
'''
m="import json\n"+m
open('/tmp/work/main.py','w').write(m)
import subprocess,sys,time,json
p=subprocess.run([sys.executable,'/tmp/work/main.py'],capture_output=True,text=True,cwd='/tmp/work')
print(p.returncode, p.stderr[-2000:])
R=json.load(open('/tmp/work/results.json'))
for k,v in R.items():
    if isinstance(v,list): print(k, np.round(np.array(v),5) if False else [round(x,5) for x in v])
    elif isinstance(v,dict): print(k, {a:round(b,5) for a,b in v.items()})
    else: print(k, v)
