
code = r'''
import numpy as np
from scipy.interpolate import interp1d
from scipy.integrate import quad
def load(fn):
    rows=[l.split() for l in open(fn) if not l.startswith('#') and l.strip()]
    return np.array([[float(x) for x in r] for r in rows])
X=load('/tmp/data/xcom_water.txt'); ES=load('/tmp/data/estar_water.txt')
xe=X[:,0]
def lg(xs,ys):
    f=interp1d(np.log(xs),np.log(np.maximum(ys,1e-300)),kind='linear',fill_value='extrapolate')
    return lambda e: np.exp(f(np.log(np.asarray(e,float))))
mu_coh=lg(xe,X[:,1]); mu_inc=lg(xe,X[:,2]); mu_pe=lg(xe,X[:,3])
mu_pair=lg(xe,X[:,4]+X[:,5]+1e-30)
mu_tot=lg(xe,X[:,6])
# ESTAR
ee=ES[:,0]
def sc(col):
    return interp1d(np.log(ee),ES[:,col],kind='linear',fill_value='extrapolate')
Scol_f=lambda T: np.exp(interp1d(np.log(ee),np.log(ES[:,1]),fill_value='extrapolate')(np.log(T)))
Srad_f=lambda T: np.exp(interp1d(np.log(ee),np.log(ES[:,2]),fill_value='extrapolate')(np.log(T)))
Y_f=lambda T: np.interp(np.log(T),np.log(ee),ES[:,5])
Rcsda_f=lambda T: np.exp(interp1d(np.log(ee),np.log(ES[:,4]),fill_value='extrapolate')(np.log(T)))
re=2.8179403e-13; mc2=0.51099895; ne=3.3428e23
def kn_total(E):
    k=E/mc2; L=np.log(1+2*k)
    return 2*np.pi*re**2*((1+k)/k**2*(2*(1+k)/(1+2*k)-L/k)+L/(2*k)-(1+3*k)/(1+2*k)**2)
def moller_dsig(T,Td):
    tau=T/mc2; g=1+tau; b2=1-1/g**2
    eps=Td/T
    F=1/eps**2+1/(1-eps)**2+(tau/(tau+2))**2-(2*tau+1)/((tau+1)**2*eps*(1-eps))
    return 2*np.pi*re**2*mc2/(b2*T**2)*F
'''
open('/tmp/work/phys.py','w').write(code) if __import__('os').path.isdir('/tmp/work') else None
import os; os.makedirs('/tmp/work',exist_ok=True); open('/tmp/work/phys.py','w').write(code)
import sys; sys.path.insert(0,'/tmp/work')
import numpy as np
from phys import *
from scipy.integrate import quad
for e in [0.1,0.5,1.1732,1.3325]:
    print(e, kn_total(e)*ne, mu_inc(e))
for T in [0.1,0.5,1.0,2.0]:
    Ls=ne*quad(lambda td: td*moller_dsig(T,td),0.075e-3,T/2,limit=400)[0]
    print(T, Ls, Scol_f(T))
