
code = r'''
import numpy as np
from scipy import integrate
re=2.8179403e-13; me=0.51099895
NA=6.02214076e23
wH,wO=0.111894,0.888106
ZA_w=wH*1/1.00794+wO*8/15.9994
ne_w=NA*ZA_w
ZA_air=0.49919
ne_air=NA*ZA_air
def kn_sigma(E):
    k=E/me
    t1=(1+k)/k**2*(2*(1+k)/(1+2*k)-np.log(1+2*k)/k)
    t2=np.log(1+2*k)/(2*k)
    t3=(1+3*k)/(1+2*k)**2
    return 2*np.pi*re**2*(t1+t2-t3)
def dsig_dOmega(E,c):
    k=E/me
    eps=1/(1+k*(1-c))
    return 0.5*re**2*eps**2*(eps+1/eps-(1-c**2))
'''
open('phys.py','w').write(code)
import importlib, sys
sys.path.insert(0,'.')
import phys
from scipy import integrate
import numpy as np
for E in [0.1,1.25]:
    s,_=integrate.quad(lambda c: phys.dsig_dOmega(E,c)*2*np.pi,-1,1)
    print(E,s,phys.kn_sigma(E))
    tr,_=integrate.quad(lambda c: phys.dsig_dOmega(E,c)*2*np.pi*E*(1-1/(1+E/phys.me*(1-c))),-1,1)
    print(' tr/rho', tr*phys.ne_w, 'frac',tr/s)
