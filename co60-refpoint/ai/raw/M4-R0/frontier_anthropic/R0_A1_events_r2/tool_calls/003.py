
code = r'''
import numpy as np
me=0.51099895; re=2.8179403e-13; NA=6.02214076e23
ne=NA*10/18.015
nmol=NA/18.015
X0=0.4
def sig_KN(E):
    a=E/me; t=np.log(1+2*a)
    return 2*np.pi*re**2*((1+a)/a**2*(2*(1+a)/(1+2*a)-t/a)+t/(2*a)-(1+3*a)/(1+2*a)**2)
def sig_R(E,x0=None):
    x0 = X0 if x0 is None else x0
    th=np.linspace(0,np.pi,20001)
    lam=0.012398/E
    x=np.sin(th/2)/lam
    f=0.5*re**2*(1+np.cos(th)**2)*100*np.exp(-(x/x0)**2)*2*np.pi*np.sin(th)
    return np.trapezoid(f,th)
def mu_c(E): return ne*sig_KN(E)
def mu_R(E): return nmol*sig_R(E)
def mu_pe(E): return 0.0025*(0.1/E)**3
def mu_pair(E):
    E=np.asarray(E,float); return np.where(E>1.022,0.0215*np.clip(1-1.022/E,0,None)**2.5,0.0)
def mu_tot(E): return mu_c(E)+mu_R(E)+mu_pe(E)+mu_pair(E)
'''
open('phys.py','w').write(code)
import subprocess
test = r'''
import phys
from phys import *
for X in [0.4,0.5,0.8]:
    phys.X0=X
    print(X, [round(float(mu_R(E)),6) for E in [0.1,0.5,1.25]])
'''
open('t.py','w').write(test)
print(subprocess.run(['python','t.py'],capture_output=True,text=True).stdout)
