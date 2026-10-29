
ana_code = r'''
import numpy as np
from phys import *
from scipy.integrate import quad
import mc
M_all=None
def kn_pdf_grid(E, neps=4000):
    """return eps grid, normalised pdf f(eps) on [eps0,1] for KN at energy E (per unit eps), and sin^2"""
    k=E/mc2; eps0=1/(1+2*k)
    eg=np.exp(np.linspace(np.log(eps0),0,neps))
    c=1-(1/eg-1)/k; s2=1-c*c
    h=eg+1/eg-s2
    return eg,h
def kn_eps_stats(E):
    eg,h=kn_pdf_grid(E)
    Z=np.trapezoid(h,eg)
    # sigma check
    k=E/mc2
    sig_num=np.pi*re**2/k*Z
    mean_eps=np.trapezoid(h*eg,eg)/Z
    return sig_num, mean_eps
if __name__=="__main__":
    for E in [0.1,0.5,1.1732,1.3325]:
        s,me=kn_eps_stats(E); print(E, s/kn_total(E), 1-me)
'''
open('/tmp/work/ana.py','w').write(ana_code)
import importlib, ana; importlib.reload(ana)
import subprocess
print(subprocess.run(['python','/tmp/work/ana.py'],capture_output=True,text=True,cwd='/tmp/work'))
