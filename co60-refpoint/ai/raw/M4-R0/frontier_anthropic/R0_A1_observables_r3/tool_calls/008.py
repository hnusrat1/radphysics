
import sys; sys.path.insert(0,'/tmp/work')
import os; print(os.path.exists('/tmp/work/mc.py'))
s=open('/tmp/work/mc.py').read()
s=s.replace("def run(N, zpts, seed=7, Emin=0.005, store_main=None):","def run(N, zpts, seed=7, Emin=0.005, store_main=None, maxgen=200):")
s=s.replace("if gen>200: break","if gen>=maxgen: break")
open('/tmp/work/mc.py','w').write(s)
import mc as M, physics as P
import numpy as np
from scipy import integrate
res1=M.run(400000,[5.0],seed=11,maxgen=1)
print("MC first-order scatter/Phi0:", P.SF/400000*res1['A_tot'][0])
def integrand(z,rho,E):
    field_half=5*(100+z)/100
    if rho>field_half: return 0.0
    phip=(105/(100+z))**2*np.exp(-P.mutot(E)*z)
    rv_z=5-z; r=np.sqrt(rv_z**2+rho**2)
    cpsi=rv_z/r
    eps=1/(1+E/P.me*(1-cpsi)); Ep=E*eps
    dsig=0.5*P.re**2*eps**2*(eps+1/eps-(1-cpsi**2))
    p=dsig/P.KN_sigma(E)
    att=np.exp(-P.mutot(Ep)*r)
    return P.muC(E)*phip*p*att/r**2*2*np.pi*rho
tot=0
for E,wt in [(1.1732,0.5),(1.3325,0.5)]:
    val,err=integrate.dblquad(lambda rho,z: integrand(z,rho,E), 0,30, lambda z:0, lambda z:min(5*(100+z)/100,15), epsabs=1e-8, epsrel=1e-4)
    tot+=wt*val
print("numerical first-order:", tot)
