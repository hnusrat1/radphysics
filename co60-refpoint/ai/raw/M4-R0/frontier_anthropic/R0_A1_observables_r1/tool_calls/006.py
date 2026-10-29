
code = r'''
import numpy as np
from numpy.polynomial.legendre import leggauss
re=2.8179403e-13; mc2=0.51099895; ne=3.3428e23
Lines=np.array([1.1732,1.3325])
def sigKN(E):
    e=np.asarray(E,float); L=np.log(1+2*e)
    return 2*np.pi*re**2*((1+e)/e**2*(2*(1+e)/(1+2*e)-L/e)+L/(2*e)-(1+3*e)/(1+2*e)**2)
def fKN(E,c):
    Ep=1/(1+E/mc2*(1-c)); s2=1-c*c
    return re**2/2*Ep**2*(Ep+1/Ep-s2)
def mu_pe(E): return 0.0022*(0.1/np.asarray(E,float))**3
def mu_coh(E): return 0.0030*(0.1/np.asarray(E,float))**2
def mu_pair(E):
    x=np.clip((np.asarray(E,float)-1.022)/0.308,0,None); return 0.0002*x**2
def mu_c(E): return ne*sigKN(E)
def mu_tot(E): return mu_c(E)+mu_coh(E)+mu_pe(E)+mu_pair(E)
_x,_w=leggauss(200)
def mu_tr_c(E):
    E=np.atleast_1d(np.asarray(E,float))
    out=np.zeros(E.shape)
    for i,e in enumerate(E):
        Ep=1/(1+e/mc2*(1-_x))
        out[i]=ne*2*np.pi*np.sum(_w*(e-e*Ep)*fKN(e,_x))
    return out
OMEGA=4*np.arcsin(25/10025.)
'''
open('/tmp/work/phys.py','w').write(code)
sys.path.insert(0,'/tmp/work')
import phys
print(phys.mu_tot(phys.Lines), phys.mu_tr_c(phys.Lines), phys.mu_pair(phys.Lines), phys.OMEGA)
print(phys.mu_tot(np.array([0.1,0.5,1.0])), phys.mu_tr_c(np.array([0.1,0.5,1.0])))
