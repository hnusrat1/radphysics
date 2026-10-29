
import sys; sys.path.insert(0,'/tmp/work')
import numpy as np, physics as P
rng=np.random.default_rng(0)
E=np.full(200000,1.3325)
eps,cc=P.sample_KN(E,rng)
T=E*(1-eps)
print(T.max(), np.mean(T>1.15), (T>1.12).sum())
E=np.full(200000,1.1732); eps,cc=P.sample_KN(E,rng); print((E*(1-eps)).max())
# electrons function from full.py's photo / pair for primary: check photo weights
print(P.mupe(np.array([1.3325,1.1732])), P.mupair(np.array([1.3325])))
