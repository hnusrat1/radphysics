
import subprocess,sys
code=r'''
import sys; sys.path.insert(0,'/tmp/work')
import numpy as np, lib
from lib import *
EL=np.array([1.1732,1.3325])
for E in EL:
    e=np.array([E]); mug,w=None,None
    mg=np.linspace(-1,1,401); f=kn_f(E,mg); w=f/f.sum()
    Ep=E/(1+E/mec2*(1-mg)); T=np.sum(w*(E-Ep))
    print("E",E,"mu_inc",mu_inc(e)[0],"mu_tr_c/rho approx", mu_inc(e)[0]*T/E*E, "<T>",T, "mu_inc*<T>",mu_inc(e)[0]*T)
    print(" total mu",mutot(e)[0], "fraction tr", T/E)
'''
open('/tmp/work/dbg.py','w').write(code)
print(subprocess.run([sys.executable,'/tmp/work/dbg.py'],capture_output=True,text=True).stdout)
