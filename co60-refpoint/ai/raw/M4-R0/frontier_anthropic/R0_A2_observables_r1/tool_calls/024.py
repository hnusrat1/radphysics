
import subprocess, sys, time
t0=time.time()
ps=[subprocess.Popen([sys.executable,'/tmp/work/sim.py','1000000',str(s),f'r{s}'],stdout=open(f'/tmp/work/log{s}.txt','w'),stderr=subprocess.STDOUT,cwd='/tmp/work') for s in [13,14]]
for p in ps: p.wait()
print([p.returncode for p in ps], time.time()-t0)
import glob; print(sorted(glob.glob('/tmp/work/H_*.npz')))
