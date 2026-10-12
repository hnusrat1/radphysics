
s = open('/tmp/work/ana.py').read()
s = s.replace("# primary-origin e- births\nselb", "prod = Phi_tot*dT*np.array([sigma_delta_above(E) if E>0.02 else 0.0 for E in Tnode])\n# primary-origin e- births\nselb")
s = s.replace("th0 = 13.6/(beta*pc)*np.sqrt(x)*(1+0.038*np.log(x))","th0 = 13.6/(beta*pc)*np.sqrt(x)*(1+0.038*np.log(x))\n        th0 = np.minimum(th0, 1.0)")
open('/tmp/work/ana.py','w').write(s)
import subprocess
r=subprocess.run("cd /tmp/work && python ana.py",shell=True,capture_output=True,text=True)
out=r.stdout
print(out[:700]); print(r.stderr[-600:])
import json
