
import subprocess,time
t=time.time()
procs=[subprocess.Popen(['python','mc.py',str(s),'2500000'],stdout=subprocess.PIPE,stderr=subprocess.PIPE,text=True) for s in range(2,6)]
outs=[p.communicate() for p in procs]
print([o[0] for o in outs],[o[1][-500:] for o in outs], time.time()-t)
