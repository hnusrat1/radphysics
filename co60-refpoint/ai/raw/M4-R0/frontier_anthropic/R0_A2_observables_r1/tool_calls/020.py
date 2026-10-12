
import subprocess, sys, time
t0=time.time()
procs=[subprocess.Popen([sys.executable,'/tmp/work/sim.py','1000000',str(s),f'r{s}'],stdout=subprocess.PIPE,stderr=subprocess.STDOUT,cwd='/tmp/work') for s in [11,12,13,14]]
outs=[p.communicate()[0].decode() for p in procs]
print(outs, time.time()-t0)
