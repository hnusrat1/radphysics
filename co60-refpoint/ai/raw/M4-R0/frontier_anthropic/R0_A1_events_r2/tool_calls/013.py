
import subprocess
t=open('t5.py').read().replace("from scipy.integrate import quad, dblquad\n","")
open('t5.py','w').write(t)
r=subprocess.run(['python','t5.py'],capture_output=True,text=True,cwd='.'); print(r.stdout,r.stderr[-2000:])
