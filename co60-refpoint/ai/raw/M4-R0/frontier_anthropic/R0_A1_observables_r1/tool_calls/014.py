
import sys, time, numpy as np
sys.path.insert(0,'/tmp/work')
import phys, mc
depths=[0.25,0.5,1,2,3,5,8,10,15,20]
N=400000
r=mc.run(N,depths,np.random.default_rng(77),store_for=5.0)
sc=r['dscat']/N*phys.OMEGA*105**2 
mu=phys.mu_tot(phys.Lines)
def prim(z):
    return sum(0.5*phys.ETR(E)*np.exp(-m*z)*(105/(100+z))**2 for E,m in zip(phys.Lines,mu))
for z,s in zip(depths,sc):
    p=prim(z); print(z, p, s, s/p, (p+s)/p)
