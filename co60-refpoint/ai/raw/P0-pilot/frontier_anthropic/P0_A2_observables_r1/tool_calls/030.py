
s=open('/tmp/work/pdd.py').read()
s=s.replace("Gz=1.602e-10*(np.exp(-muT*ZD)*float(dep2['col'][0])+Kd)","Gz=1.602e-10*((110.0/(100.0+ZD))**2*np.exp(-muT*ZD)*float(dep2['col'][0])+Kd)")
open('/tmp/work/pdd.py','w').write(s)
import subprocess
r=subprocess.run(['python3','/tmp/work/pdd.py'],capture_output=True,text=True,cwd='/tmp/work')
print(r.stdout[-5000:], r.stderr[-3000:])
