
import subprocess,time
t=time.time()
procs=[subprocess.Popen(['python','mc.py',str(s),'10000000'],stdout=subprocess.PIPE,stderr=subprocess.PIPE,text=True) for s in [11,12,13,14]]
for p in procs:
    o,e=p.communicate(); print(o[-200:],e[-800:])
print(time.time()-t)
