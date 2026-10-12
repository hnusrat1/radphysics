
import numpy as np
from scipy.interpolate import interp1d
def load(fn):
    rows=[l.split() for l in open(fn) if not l.startswith('#') and l.strip()]
    return np.array([[float(x) for x in r] for r in rows])
X=load('/tmp/data/xcom_water.txt'); E=load('/tmp/data/estar_water.txt')
print(X.shape,E.shape); print(E[-3:,:])
xe=X[:,0]
def lg(xs,ys):
    f=interp1d(np.log(xs),np.log(np.maximum(ys,1e-300)),kind='linear',fill_value='extrapolate')
    return lambda e: np.exp(f(np.log(e)))
mu_coh=lg(xe,X[:,1]); mu_inc=lg(xe,X[:,2]); mu_pe=lg(xe,X[:,3])
mu_pn=lg(xe,X[:,4]+X[:,5]+1e-30)  # pair total
mu_tot=lg(xe,X[:,6])
for e in [1.1732,1.3325]:
    print(e, mu_coh(e),mu_inc(e),mu_pe(e),mu_pn(e),mu_tot(e), 0.5*(np.exp(-mu_tot(e)*5)))
