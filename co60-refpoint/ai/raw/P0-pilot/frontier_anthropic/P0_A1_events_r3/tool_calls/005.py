
import subprocess,time
t=time.time()
r=subprocess.run(['python','mc.py','300000'],capture_output=True,text=True)
print(r.stdout, r.stderr[-2000:], time.time()-t)
