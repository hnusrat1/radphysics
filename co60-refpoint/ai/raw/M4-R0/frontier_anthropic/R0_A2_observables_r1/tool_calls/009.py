
import sys, subprocess
print(sys.executable)
r=subprocess.run([sys.executable,'/tmp/work/sim.py','20000','1','test'],capture_output=True,text=True,cwd='/tmp/work')
print(r.stdout, r.stderr[-3000:])
