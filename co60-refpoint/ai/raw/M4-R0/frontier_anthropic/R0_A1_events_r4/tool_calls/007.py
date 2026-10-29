
import subprocess,time
t=time.time()
r=subprocess.run(['python','mc2.py','99','20000000'],capture_output=True,text=True)
print(r.stdout, r.stderr[-2000:], time.time()-t)
