
import numpy as np
from scipy import integrate
def load(fn):
    rows=[l.split() for l in open(fn) if l.strip() and not l.startswith('#')]
    return np.array([[float(x) for x in r] for r in rows])
X=load('/tmp/data/xcom_water.txt')   # E, coh, incoh, pe, pair_n, pair_e, tot_with, tot_without
S=load('/tmp/data/estar_water.txt')  # T, Scol, Srad, Stot, range, yield, delta
Ex=X[:,0]
def lli(x,y,xn):
    return np.exp(np.interp(np.log(xn),np.log(x),np.log(np.maximum(y,1e-300))))
def mu(E,which):
    cols={'coh':1,'inc':2,'pe':3,'pair':None}
    if which=='pair': return lli(Ex,X[:,4]+X[:,5],E)
    return lli(Ex,X[:,cols[which]],E)
def mutot(E): return lli(Ex,X[:,6],E)
rho=1.0
print(mutot(np.array([1.1732,1.3325])), mu(np.array([1.1732,1.3325]),'pair'), mu(np.array([1.1732,1.3325]),'inc'))
# primary attenuation to depth 5 cm
Ea,Eb=1.1732,1.3325
pa=np.exp(-mutot(np.array([Ea]))[0]*5); pb=np.exp(-mutot(np.array([Eb]))[0]*5)
print("primary per Phi0 (ratio to 105 cm, both lines):",0.5*(pa+pb), pa,pb)
