import numpy as np
from scipy.interpolate import interp1d
import json

np.random.seed(42)

# ============ CONSTANTS & CROSS SECTIONS ============
m_e = 0.51099895
r_e = 2.8179403262e-13
NA = 6.02214076e23

ea = np.array([1.0, 1.1, 1.1732, 1.2, 1.25, 1.3, 1.3325, 1.4, 1.5])
ma = {  # [total, comp, phot, pair, rayl]
    't': np.array([0.07070, 0.06780, 0.06590, 0.06520, 0.06400, 0.06280, 0.06220, 0.06080, 0.05780]),
    'c': np.array([0.06950, 0.06650, 0.06430, 0.06350, 0.06200, 0.06060, 0.06000, 0.05850, 0.05550]),
    'p': np.array([0.00040, 0.00030, 0.00025, 0.00022, 0.00020, 0.00017, 0.00015, 0.00013, 0.00010]),
    'pp': np.array([0.0, 0.0, 0.00020, 0.00028, 0.00050, 0.00080, 0.00100, 0.00130, 0.00150]),
    'r': np.array([0.00080, 0.00068, 0.00060, 0.00055, 0.00050, 0.00043, 0.00040, 0.00035, 0.00030])
}
fi = {}
for k in ['t','c','p','pp','r']:
    fi[k] = interp1d(ea, ma[k], kind='cubic', bounds_error=False, fill_value='extrapolate')

def mu_t(E): return float(fi['t'](np.clip(E,0.01,10.0)))
def mu_c(E): return float(fi['c'](np.clip(E,0.01,10.0)))
def mu_ph(E): return float(fi['p'](np.clip(E,0.01,10.0)))
def mu_pp(E): return float(fi['pp'](np.clip(E,0.01,10.0)))
def mu_rl(E): return float(fi['r'](np.clip(E,0.01,10.0)))

# ============ GEOMETRY ============
SSD=100.0; fh=5.0; zs=0.0; zc=5.0; zh=0.1
zlo=zc-zh; zhi=zc+zh; Rc=1.0
ctr = np.array([0.,0.,zc])
th = np.arctan(fh/SSD)
Opyr = 4*np.arcsin(np.sin(th)**2)
ne = 10.0*NA/18.015

print(f"Omega_pyr={Opyr:.6f} sr, ne={ne:.4e}")

# ============ PHYSICS ============
def c_E(E,ct): return E/(1.0+(E/m_e)*(1.0-ct))

def KN(E,ct):
    th = np.arccos(np.clip(ct,-1.,1.))
    Eo = c_E(E,ct)
    return (r_e**2/2.0)*(Eo/E)**2*(E/Eo+Eo/E-np.sin(th)**2)

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
    u=v/np.linalg.norm(v); th=np.arccos(np.clip(ct,-1.,1.))
    ph=np.random.uniform(0.,2.*np.pi)
    if abs(u[2])<0.999: w=np.cross(u,np.array([0.,0.,1.]))
    else: w=np.cross(u,np.array([1.,0.,0.]))
    w=w/np.linalg.norm(w); vv=np.cross(w,u)
    return u*ct+w*np.sin(th)*np.cos(ph)+vv*np.sin(th)*np.sin(ph)