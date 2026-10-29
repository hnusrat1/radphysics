
import subprocess, time
t0=time.time()
procs=[subprocess.Popen(['python3','/tmp/work/mc.py','30000000',str(s),'500000'],stdout=subprocess.PIPE,stderr=subprocess.PIPE,text=True,cwd='/tmp/work') for s in [11,12,13,14]]
outs=[p.communicate()[0] for p in procs]
print(outs, time.time()-t0)
