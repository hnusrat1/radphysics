
import sys, numpy as np
sys.path.insert(0,'/tmp/work')
import phys
from phys import re, mc2, ne
# Moller-type delta spectrum shape and checks against Berger-Seltzer collision stopping power
def Kconst(n): return 2*np.pi*re**2*mc2*n   # MeV cm^2 per g(with n per cm3)
def f_moller(T,eps):
    tau=T/mc2; return 1/eps**2 + 1/(T-eps)**2 + (((tau+2)/(tau+1))**2-(2*tau+1)/(tau+1)**2)/(eps*(T-eps))
def beta2(T):
    g=1+T/mc2; return 1-1/g**2
def J(T,I,emax):
    # integral of eps*f from I to emax (log grid)
    if emax<=I: return 0.0
    le=np.linspace(np.log(I),np.log(emax),4000)
    e=np.exp(le)
    y=e*f_moller(T,e)*e
    return np.trapezoid(y,le)
def S_BS(T,I=75e-6,dens=True):
    tau=T/mc2; b2=beta2(T)
    L=np.log(tau**2*(tau+2)/(2*(I/mc2)**2))
    F=1-b2+(tau**2/8-(2*tau+1)*np.log(2))/(tau+1)**2
    return Kconst(ne)/(2*np.pi*re**2*mc2)*(2*np.pi*re**2*mc2)/b2*(L+F)
for T in [0.1,0.5,1.0]:
    tau=T/mc2; b2=beta2(T)
    Jv=J(T,75e-6,T/2)
    Kc=2*np.pi*re**2*mc2*ne/b2
    print(T, Kc*Jv, S_BS(T))
