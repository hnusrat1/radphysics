
s=open('/tmp/work/mell.py').read()
s=s.replace("SIGD[i]=NEW*K0/b2/T*cdf[-1]","SIGD[i]=0.55509*K0/b2/T*cdf[-1]")
s=s.replace("    cdfn=cdf/cdf[-1]","    cdfn=cdf/max(cdf[-1],1e-30)")
open('/tmp/work/mell.py','w').write(s)
import sys; sys.path.insert(0,'/tmp/work')
import importlib, numpy as np, mell
print(mell.sig_delta(np.array([0.1,0.5,1.0,2.0])))
