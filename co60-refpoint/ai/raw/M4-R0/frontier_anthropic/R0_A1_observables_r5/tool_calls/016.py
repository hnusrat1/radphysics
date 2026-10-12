
code = r'''
import numpy as np
from phys import re, me, ne_w as ne, ne_air, NA
alpha=1/137.036
def beta2_T(T):
    tau=T/me; return tau*(tau+2)/(tau+1)**2
def S_col_bethe(T,I=75e-6,n=ne):
    tau=T/me; b2=beta2_T(T)
    F=(1-b2)*(1+tau**2/8-(2*tau+1)*np.log(2))
    L=np.log(tau**2*(tau+2)/(2*(I/me)**2))
    return 2*np.pi*re**2*me*n/b2*(L+F)
def moller_f(x,tau):
    return 1/x**2+1/(1-x)**2+(tau/(tau+1))**2-(2*tau+1)/((tau+1)**2*x*(1-x))
def dsig_deps(T,eps):
    tau=T/me; b2=beta2_T(T)
    x=eps/T
    return 2*np.pi*re**2*me/(b2*T**2)*moller_f(x,tau)
def S_rad(T):
    # empirical: S_rad/S_col ~ 0.0075*T (MeV), consistent with ESTAR-like values (~0.013 at 1 MeV, ~0.18 at 10 MeV)
    return S_col_bethe(T)*0.0075*T
'''
open('/tmp/work/ea.py','w').write(code)
import sys; sys.path.insert(0,'/tmp/work')
import importlib, ea; importlib.reload(ea)
import numpy as np
from scipy import integrate
for T in [0.1,0.5,1.0,1.3]:
    val,_=integrate.quad(lambda e: e*ea.dsig_deps(T,e)*ea.ne,75e-6,T/2,limit=500,points=[1e-3,1e-2])
    print(T, val, ea.S_col_bethe(T), ea.S_col_bethe(T)-val)
