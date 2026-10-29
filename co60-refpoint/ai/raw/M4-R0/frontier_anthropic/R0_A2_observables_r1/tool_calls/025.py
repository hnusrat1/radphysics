
import subprocess, sys, time, json
t0=time.time()
p=subprocess.run([sys.executable,'/tmp/work/main.py'],capture_output=True,text=True,cwd='/tmp/work')
print(p.returncode, p.stderr[-3000:], time.time()-t0)
