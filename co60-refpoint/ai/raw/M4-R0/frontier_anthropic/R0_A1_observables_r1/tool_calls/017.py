
elec = r'''
import numpy as np
from phys import re, mc2, ne
ne_air=0.49919*6.02214e23
def beta2(T):
    g=1+np.asarray(T,float)/mc2; return 1-1/g**2
def f_moller(T,eps):
    tau=T/mc2; return 1/eps**2 + 1/(T-eps)**2 + (((tau+2)/(tau+1))**2-(2*tau+1)/(tau+1)**2)/(eps*(T-eps))
def J_restr(T,I,emax):
    """integral eps*f from I to emax (MeV units) -> dimensionless; multiply by 2pi re^2 mc2 n/beta^2"""
    T=float(T)
    emax=min(emax,T/2)
    if emax<=I: return 0.0
    le=np.linspace(np.log(I),np.log(emax),3000)
    e=np.exp(le)
    return np.trapezoid(e*f_moller(T,e)*e,le)
def dens_delta(T, plasma_I=75.0):
    # Sternheimer for water (liquid)
    C=3.5017; x0=0.24; x1=2.8004; a=0.09116; m=3.4773
    g=1+T/mc2; bg=np.sqrt(g*g-1); X=np.log10(bg)
    if X<x0: return 0.0
    if X>x1: d=4.6052*X-C
    else: d=4.6052*X-C+a*(x1-X)**m
    return d
def S_col_BS(T,I,n,dens=False):
    tau=T/mc2; b2=beta2(T)
    L=np.log(tau**2*(tau+2)/(2*(I/mc2)**2))
    F=1-b2+(tau**2/8-(2*tau+1)*np.log(2))/(tau+1)**2
    d=dens_delta(T) if dens else 0.0
    return 2*np.pi*re**2*mc2*n/b2*(L+F-d)
def S_rad_w(T):
    T=np.asarray(T,float)
    return 0.0129*T*(T+mc2)/(1+mc2)
def S_res(T,Delta,medium):
    """restricted collision stopping power (MeV cm2/g), cutoff Delta (MeV)."""
    if medium=='w':
        I=75e-6; n=ne; dens=True
    else:
        I=85.7e-6; n=ne_air; dens=False
    Sun=S_col_BS(T,I,n,dens)
    # subtract energy transfers above Delta, using same normalisation as Bethe formula
    # calibrate normalisation of f_moller against Berger-Seltzer (without density)
    b2=beta2(T)
    K=2*np.pi*re**2*mc2*n/b2
    Jabove=J_restr(T,Delta,T/2)
    return Sun - K*Jabove if Delta<T/2 else Sun
'''
open('/tmp/work/elec.py','w').write(elec)
import sys; sys.path.insert(0,'/tmp/work')
import elec, numpy as np
for T in [0.02,0.1,0.5,1.0]:
    b2=elec.beta2(T); K=2*np.pi*elec.re**2*elec.mc2*elec.ne/b2
    print(T, K*elec.J_restr(T,75e-6,T/2), elec.S_col_BS(T,75e-6,elec.ne,False))
