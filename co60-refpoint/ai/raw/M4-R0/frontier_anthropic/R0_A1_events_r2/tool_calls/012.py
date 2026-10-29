
import subprocess
t=r'''
import numpy as np
from phys import *
from sim import KN_q
from scipy.integrate import quad, dblquad
E=1.25
mu=float(mu_tot(E)); mc=float(mu_c(E))
# angular integral over downstream hemisphere: dOmega = 2pi dc, c=cos theta in (0,1]
def phi_s(h):
    def inner(zp):
        def f(c):
            qc=float(KN_q(np.array([E]),np.array([c]))[0]) if False else None
            return 0
        return 0
    # use cos theta grid
    c=np.linspace(1e-4,1,4000)
    q=KN_q(np.full_like(c,E),c)
    # scatter angle theta measured from beam: cos(theta_scatter) = c
    zz=np.linspace(0,h,2001)
    tot=0
    vals=[]
    for zp in zz:
        ell=(h-zp)/c
        integrand=2*np.pi*q*np.exp(-mu*ell)/c
        vals.append(np.trapezoid(integrand,c))
    vals=np.array(vals)
    return np.trapezoid(mc*np.exp(-mu*zz)*vals,zz)
h=5.0
ps=phi_s(h)
print('infinite broad beam: phi_s/phi0 =', ps/np.exp(-mu*h))
'''
open('t5.py','w').write(t)
r=subprocess.run(['python','t5.py'],capture_output=True,text=True,cwd='.'); print(r.stdout,r.stderr[-2000:])
