
import subprocess, time, json
t=time.time()
procs=[subprocess.Popen(["python3","mc.py","2000000",str(s),f"run{s}.json"],stdout=subprocess.PIPE,text=True) for s in range(1,5)]
outs=[p.communicate()[0] for p in procs]
print(outs, time.time()-t)
