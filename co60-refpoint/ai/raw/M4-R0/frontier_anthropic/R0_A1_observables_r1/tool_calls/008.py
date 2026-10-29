
import sys, numpy as np
src=open('/tmp/work/phys.py').read().replace("e=np.asarray(E,float); L=np.log(1+2*e)","e=np.asarray(E,float)/mc2; L=np.log(1+2*e)")
open('/tmp/work/phys.py','w').write(src)
import importlib; sys.path.insert(0,'/tmp/work'); import phys; importlib.reload(phys)
print(phys.mu_tot(phys.Lines), phys.mu_tr_c(phys.Lines), phys.mu_pair(phys.Lines))
print(phys.mu_tot(np.array([0.1,0.5,1.0])), phys.mu_tr_c(np.array([0.1,0.5,1.0])))
