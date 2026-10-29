
s=open('/tmp/work/mcmod.py').read().replace('np.trapz','np.trapezoid')
open('/tmp/work/mcmod.py','w').write(s)
import importlib, mcmod
importlib.reload(mcmod)
from mcmod import *
print(Phi0_src, Omega)
rng=np.random.default_rng(0)
Ep=np.full(200000,1.25)
eps,cs=sample_kn(Ep,rng); print("mean eps", eps.mean(), "mean T/E", (1-eps).mean())
print(mu_tr_c(1.25), mu_en_c(1.25), mu_tr_c(1.0), mu_en_c(0.5))
