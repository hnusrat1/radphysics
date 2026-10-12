
s=open('/tmp/work/full.py').read()
s=s.replace("Lk=np.interp(Tall[mE],np.concatenate([[0.01],Tk]),np.concatenate([[0],Lcum[:-0 or None]]) if False else np.concatenate([[0],Lcum]) )",
"Lgrid=np.cumsum(h/Sw)\nLk=np.where(Tall[mE]>=0.01, np.interp(Tall[mE],Tk,Lgrid), 0.0)")
open('/tmp/work/full.py','w').write(s)
import subprocess
r=subprocess.run(['python3','/tmp/work/full.py'],capture_output=True,text=True,cwd='/tmp/work')
print(r.stdout[-3000:], r.stderr[-3000:])
