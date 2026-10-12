
dbg = r'''
import numpy as np, sys
sys.path.insert(0,'/tmp/work')
import ea
from phys import me, ne_w as ne
dT=0.001
Tg=np.arange(0.0005,1.4+1e-9,dT)
# test delta production from a monoenergetic electron of T=0.4 MeV: number per cm above 10 keV
for T in [0.2,0.4,1.0]:
    eps=np.arange(0.01,T/2,1e-4)
    N=ne*np.trapz(ea.dsig_deps(T,eps),eps)
    print(T,'deltas per cm >10keV',N, 'beta2',ea.beta2_T(T))
    print(' Rutherford check at eps=0.02', ea.dsig_deps(T,0.02), 2*np.pi*ea.re**2*me/(ea.beta2_T(T)*0.02**2))
'''
open('/tmp/work/dbg.py','w').write(dbg)
o,e=runpy('/tmp/work/dbg.py'); print(o,e)
