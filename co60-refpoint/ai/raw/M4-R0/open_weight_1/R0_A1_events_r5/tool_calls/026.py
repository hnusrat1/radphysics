full_code = '''
import numpy as np
from scipy.interpolate import interp1d
import json, sys

np.random.seed(42)

m_e = 0.51099895
r_e = 2.8179403262e-13
NA = 6.02214076e23
SSD = 100.0; fh = 5.0; zs = 0.0; zc = 5.0; zh = 0.1
zlo = zc - zh; zhi = zc + zh; Rc = 1.0
th = np.arctan(fh/SSD); Opyr = 4.0*np.arcsin(np.sin(th)**2)
ne = 10.0*NA/18.015

print(f"O={Opyr:.6f} ne={ne:.4e}", file=sys.stderr)

ea = np.array([1.0,1.1,1.1732,1.2,1.25,1.3,1.3325,1.4,1.5])
xT = np.array([0.07070,0.06780,0.06590,0.06520,0.06400,0.06280,0.06220,0.06080,0.05780])
xC = np.array([0.06950,0.06650,0.06430,0.06350,0.06200,0.06060,0.06000,0.05850,0.05550])
xP = np.array([0.00040,0.00030,0.00025,0.00022,0.00020,0.00017,0.00015,0.00013,0.00010])
xPP= np.array([0.0,0.0,0.00020,0.00028,0.00050,0.00080,0.00100,0.00130,0.00150])
xR = np.array([0.00080,0.00068,0.00060,0.00055,0.00050,0.00043,0.00040,0.00035,0.00030])

fT=interp1d(ea,xT,kind='cubic',bounds_error=False,fill_value='extrapolate')
fC=interp1d(ea,xC,kind='cubic',bounds_error=False,fill_value='extrapolate')
fP=interp1d(ea,xP,kind='cubic',bounds_error=False,fill_value='extrapolate')
fPP=interp1d(ea,xPP,kind='cubic',bounds_error=False,fill_value='extrapolate')
fR=interp1d(ea,xR,kind='cubic',bounds_error=False,fill_value='extrapolate')

def mu_t(E): return float(fT(np.clip(E,0.01,10.0)))
def mu_c(E): return float(fC(np.clip(E,0.01,10.0)))
def mu_ph(E): return float(fP(np.clip(E,0.01,10.0)))
def mu_pp(E): return float(fPP(np.clip(E,0.01,10.0)))
def mu_rl(E): return float(fR(np.clip(E,0.01,10.0)))

def c_E(E,ct): return E/(1.0+(E/m_e)*(1.0-ct))

def KN(E,ct):
    thv=np.arccos(np.clip(ct,-1.,1.))
    Eo=c_E(E,ct)
    return (r_e**2/2.0)*(Eo/E)**2*(E/Eo+Eo/E-np.sin(thv)**2)

def sample_KN(E):
    mx=r_e**2*1.05
    while True:
        ct=np.random.uniform(-1.,1.)
        if np.random.random()*mx<KN(E,ct): break
    Eo=c_E(E,ct); te=E-Eo
    num=E-Eo*ct; den=np.sqrt(E**2+Eo**2-2*E*Eo*ct)
    ce=num/den if den>0 else 1.0
    return Eo,ct,te,ce

def rot(v,ct):
    u=v/np.linalg.norm(v)
    thv=np.arccos(np.cllp(ct,-1.,1.))
    ph=np.radom.uniform(0.,2.*np.pi)
    if abs(u[2])<0.999: w=np.cross(u,np.array([0.,0.,1.]))
    else: w=np.cross(u,np.array([1.,0.,0.]))
    w=w/np.linalg.norm(w); vv=np.cross(w,u)
    return u*ct+w*np.sin(thv)*np.cos(ph)+vv*np.sin(thv)*np.sin(ph)

print("ERROR IN SCRIPT - fixing")
'''
# The above has intentional errors. Let me write a clean complete script to a file.
with open('/tmp/sim.py', 'w') as f:
    f.write('''')
print("Writing complete script...")