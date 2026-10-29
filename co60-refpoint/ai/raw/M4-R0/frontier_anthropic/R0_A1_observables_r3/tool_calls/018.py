
s=open('/tmp/work/full.py').read()
s=s.replace("def Kcol_prim(z): return 0.5*np.sum(np.exp(-np.array([P.mutot(E) for E in lines])*z)*KC(lines))",
"def Kcol_prim(z): return (105.0/(100.0+z))**2*0.5*np.sum(np.exp(-np.array([P.mutot(E) for E in lines])*z)*KC(lines))")
s=s.replace("for m in [0.15,0.2,0.25,0.3]:","for m in [0.1,0.15,0.2,0.3]:")
open('/tmp/work/full.py','w').write(s)
import subprocess
r=subprocess.run(['python3','/tmp/work/full.py'],capture_output=True,text=True,cwd='/tmp/work')
print(r.stdout[-3000:], r.stderr[-3000:])
import pickle
