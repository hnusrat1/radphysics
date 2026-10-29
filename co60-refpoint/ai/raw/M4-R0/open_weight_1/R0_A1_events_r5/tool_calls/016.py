import numpy as np
from scipy.interpolate import interp1d
import json

np.random.seed(42)

# ============ CONSTANTS & CROSS SECTIONS ============
m_e = 0.51099895
r_e = 2.8179403262e-13
NA = 6.02214076e23

ea = np.array([1.0, 1.1, 1.1732, 1.2, 1.25, 1.3, 1.3325, 1.4, 1.5])
mt = np.array([0.07070, 0.06780, 0.06590, 0.06520, 0.06400, 0.06280, 0.06220, 0.06080, 0.05780])
mc = np.array([0.06950, 0.06650, 0.06430, 0.06350, 0.06200, 0.06060, 0.06000, 0.05850, 0.05550])
mph = np.array([0.00040, 0.00030, 0.00025, 0.00022, 0.00020, 0.00017, 0.00015, 0.00013, 0.00010])
mpp = np.array([0.0, 0.0, 0.00020, 0.00028, 0.00050, 0.00080, 0.00100, 0.00130, 0.00150])
mrl = np.array([0.00080, 0.00068, 0.00060, 0.00055, 0.00050, 0.00043, 0.00040, 0.00035, 0.00030])

ft = interp1d(ea, mt, kind='cubic', bounds_error=False, fill_value='extrapolate')
fc = interp1d(ea, mc, kind='cubic', bounds_error=False, fill_value='extrapolate')
fph = interp1d(ea, mph, kind='cubic', bounds_error=False, fill_value='extrapolate')
fpp = interp1d(ea, mpp, kind='cubic', bounds_error=False, fill_value='extrapolate')
frl = interp1d(ea, mrl, kind='cubic', bounds_error=False, fill_value='extrapolate')

def mu_tot(E): return float(ft(np.clip(E, 0.01, 10.0)))
def mu_comp(E): return float(fc(np.clip(E, 0.01, 10.0)))
def mu_phot(E): return float(fph(np.clip(E, 0.01, 10.0)))
def mu_pair(E): return float(fpp(np.clip(E, 0.01, 10.0)))
def mu_rayl(E): return float(frl(np.clip(E, 0.01, 10.0)))

# ============ GEOMETRY ============
SSD=100.0; src_z=-SSD; fh=5.0; zs=0.0; zc=5.0; zh=0.1
zlo=zc-zh; zhi=zc+zh; Rc=1.0
ctr = np.array([0.,0.,zc])
th = np.arctan(fh/SSD)
Opyr = 4*np.arcsin(np.sin(th)**2)
ne = 10.0*NA/18.015
n_e = ne  # alias

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

def src_samp():
    E=1.1732 if np.random.random()<0.5 else 1.3325
    x=np.random.uniform(-fh,fh); y=np.random.uniform(-fh,fh)
    d=np.array([x,y,SSD]); d=d/np.linalg.norm(d)
    return E,d,x,y

def ray_pos(x_surf, y_surf, z):
    """Return (x,y) position of ray at depth z, given surface coordinates."""
    return x_surf*(SSD+z)/SSD, y_surf*(SSD+z)/SSD

def intersect_cyl(ori, direc, R, zL, zH):
    d=direc/np.linalg.norm(direc)
    if d[2]>0:
        tzL=(zL-ori[2])/d[2]; tzH=(zH-ori[2])/d[2]
    elif d[2]<0:
        tzL=(zH-ori[2])/d[2]; tzH=(zL-ori[2])/d[2]
    else:
        if ori[2]<zL or ori[2]>zH: return None
        tzL=-1e30; tzH=1e30
    a=d[0]**2+d[1]**2
    if a<1e-15:
        if ori[0]**2+ori[1]**2<=R**2:
            return (tzL,tzH) if tzL<tzH else (tzH,tzL)
        return None
    b=2.0*(ori[0]*d[0]+ori[1]*d[1]); c=ori[0]**2+ori[1]**2-R**2
    disc=b*b-4.0*a*c
    if disc<0: return None
    sq=np.sqrt(disc); t1=(-b-sq)/(2.0*a); t2=(-b+sq)/(2.0*a)
    if t1>t2: t1,t2=t2,t1
    te=max(tzL,t1); tx=min(tzH,t2)
    if te>=tx or tx<0: return None
    return (te,tx)

def sample_proc(E):
    t=mu_comp(E)+mu_phot(E)+mu_pair(E)+mu_rayl(E); r=np.random.random()*t
    if r<mu_comp(E): return 'compton'
    elif r<mu_comp(E)+mu_phot(E): return 'photoelectric'
    elif r<mu_comp(E)+mu_phot(E)+mu_pair(E): return 'pair'
    else: return 'rayleigh'

# Typo fix
def sample_proc(E):
    t = mu_comp(E) + mu_phot(E) + mu_pair(E) + mu_rayl(E)
    r = np.random.random() * t
    if r < mu_comp(E): return 'compton'
    elif r < mu_comp(E) + mu_phot(E): return 'photoelectric'
    elif r < mu_comp(E) + mu_phot(E) + mu_pair(E): return 'pair'
    else: return 'rayleigh'