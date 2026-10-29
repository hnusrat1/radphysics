
s=open('/tmp/work/final2.py').read()
s=s.replace("np.trapz(","np.trapezoid(")
open('/tmp/work/final2.py','w').write(s)
import subprocess
r=subprocess.run(['python3','/tmp/work/final2.py'],capture_output=True,text=True,cwd='/tmp/work')
print(r.stdout[-6000:], r.stderr[-3000:])
