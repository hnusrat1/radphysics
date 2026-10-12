
import numpy as np, sys, time
sys.path.insert(0,'/tmp/work')
import importlib, mc, phys
from scipy.integrate import dblquad
Om=dblquad(lambda y,x:(1+x*x+y*y)**-1.5,-0.05,0.05,-0.05,0.05)[0]
print("Omega",Om)
t=time.time()
R=[mc.run(500000,s) for s in range(4)]
print('time',time.time()-t)
Ntot=sum(r['N'] for r in R)
def merge(R):
    return {k:np.concatenate([r[k] for r in R]) for k in ['det','g','Ea','typ','nx','ny','nz']}
M=merge(R)
phi0_norm = Om*105**2/Ntot   # fluence per Phi0 = g * Om*105^2/N
for j,dz in enumerate(mc.DEPTHS):
    sel=M['det']==j
    tot=M['g'][sel].sum()*phi0_norm
    pri=0.5*(np.exp(-phys.mu_tot(1.1732)*dz)+np.exp(-phys.mu_tot(1.3325)*dz))
    print(dz, 'scatt/phi0', tot, 'primary', pri, 'buildup', (tot+pri)/pri)
