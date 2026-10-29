
import numpy as np, sys
sys.path.insert(0,'/tmp/work')
import phys
rng=np.random.default_rng(1)
E=np.full(200000,2.0)
eps,c=phys.sample_KN(E,rng)
print("KN mean eps at 2MeV", eps.mean(), "mean cos", c.mean())
print(phys.mu_all(np.array([2.0,0.1,0.5])))
print(phys.Yrad(np.array([0.01,1.0,2.0])), phys.Scol(np.array([1.0])))
