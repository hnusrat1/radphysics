
ana=open('/tmp/work/ana.py').read().replace("list(map(float,100*D/Dmax)[:60])","[float(v) for v in (100*D/Dmax)[:60]]")
open('/tmp/work/ana.py','w').write(ana)
import subprocess
r=subprocess.run(['python','/tmp/work/ana.py'],capture_output=True,text=True,cwd='/tmp/work')
print(r.stdout[-6000:],r.stderr[-3000:])
