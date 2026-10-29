
import subprocess,time
code=open('mc.py').read()
print('check the bound' in code)
t=time.time()
r=subprocess.run(['python','mc.py','1','20000'],capture_output=True,text=True)
print(r.stdout[-2000:], r.stderr[-3000:], time.time()-t)
