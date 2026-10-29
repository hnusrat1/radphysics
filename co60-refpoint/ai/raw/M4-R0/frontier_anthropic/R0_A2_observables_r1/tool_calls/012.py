
import subprocess, sys
procs=[subprocess.Popen([sys.executable,'/tmp/work/sim.py','1000000',str(s),f'r{s}'],stdout=open(f'/tmp/work/log{s}.txt','w'),stderr=subprocess.STDOUT,cwd='/tmp/work') for s in [11,12,13,14]]
print("started")
