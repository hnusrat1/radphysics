
s = open('/tmp/work/ana.py').read()
s = s.replace("    f = dsig_deltaE(Ep, e)*e\n    cdf = np.concatenate([[0],np.cumsum(0.5*(f[1:]+f[:-1])*np.diff(e))]); cdf/=cdf[-1]\n    u = rng.random(m_.size)",
"    f = dsig_deltaE(Ep, e)\n    cdf = np.concatenate([[0],np.cumsum(0.5*(f[1:]+f[:-1])*np.diff(e))]); cdf/=cdf[-1]\n    u = rng.random(m_.size)")
open('/tmp/work/ana.py','w').write(s)
import subprocess
r=subprocess.run("cd /tmp/work && python ana.py",shell=True,capture_output=True,text=True)
print(r.stdout[:800])
