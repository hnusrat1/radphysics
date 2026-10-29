
import subprocess, sys, time
t0=time.time()
p=subprocess.run([sys.executable,'/tmp/work/sim.py','1000000','12','r12'],capture_output=True,text=True,cwd='/tmp/work')
print(p.returncode, p.stdout[-500:], p.stderr[-1500:], time.time()-t0)
