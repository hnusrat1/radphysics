
code = r'''
import numpy as np
M_E = 0.51099895
RE = 2.8179403e-13  # cm
NA = 6.02214076e23
ZA_W = 0.55508   # Z/A for water
NEL_W = NA*ZA_W  # electrons per gram water (approx 3.343e23)

# ---------- data -------------
xc = np.loadtxt('/tmp/data/xcom_water.txt')
XE = xc[:,0]
XCOH, XINC, XPE, XPN, XPE2 = xc[:,1], xc[:,2], xc[:,3], xc[:,4], xc[:,5]
def _ll(x, y):
    y = np.maximum(y, 1e-30)
    return lambda E: np.exp(np.interp(np.log(E), np.log(XE), np.log(y)))
# pair with zero values: handle via linear interpolation of pair in E
_pair = XPN + XPE2
def mu_coh(E): return np.exp(np.interp(np.log(E), np.log(XE), np.log(np.maximum(XCOH,1e-30))))
def mu_inc(E): return np.exp(np.interp(np.log(E), np.log(XE), np.log(XINC)))
def mu_pe(E):  return np.exp(np.interp(np.log(E), np.log(XE), np.log(np.maximum(XPE,1e-30))))
def mu_pair(E):
    E = np.asarray(E, float)
    out = np.zeros_like(E)
    m = E > 1.022
    if np.any(m):
        out[m] = np.exp(np.interp(np.log(E[m]), np.log(XE), np.log(np.maximum(_pair,1e-30))))
    # pairs have cross-section below 1.022 only by interpolation artifacts; zero them
    return out
def mu_tot(E):
    return mu_coh(E)+mu_inc(E)+mu_pe(E)+mu_pair(E)

# ---------- Compton / KN -------------
def kn_sample(E, rng):
    """E array (MeV). returns eps=E'/E, cos theta (lab-scattering angle)"""
    a = E/M_E
    e0 = 1.0/(1.0+2*a)
    n = E.size
    eps = np.empty(n); cth = np.empty(n)
    todo = np.arange(n)
    while todo.size:
        aa = a[todo]
        e = e0[todo]**rng.random(todo.size)
        c = 1.0 - (1.0/e - 1.0)/aa
        s2 = 1.0 - c*c
        acc = rng.random(todo.size)*2.0 < (e*e + 1.0 - e*s2)
        idx = todo[acc]
        eps[idx] = e[acc]; cth[idx] = c[acc]
        todo = todo[~acc]
    return eps, cth

def kn_mean_eps_table(Egrid):
    out = []
    for E in Egrid:
        a = E/M_E
        e0 = 1/(1+2*a)
        e = np.linspace(e0, 1, 40001)
        c = 1 - (1/e-1)/a
        s2 = 1-c*c
        f = e + 1/e - s2
        out.append(np.trapz(e*f,e)/np.trapz(f,e))
    return np.array(out)

_EG = np.logspace(-3, np.log10(20), 400)
_EPS_MEAN = kn_mean_eps_table(_EG)
def eps_mean(E):
    return np.interp(np.log(E), np.log(_EG), _EPS_MEAN)

def rotate_dir(d, cth, phi):
    """d: (n,3) unit vectors; returns scattered directions with polar cth and azimuth phi"""
    ref = np.zeros_like(d); 
    m = np.abs(d[:,0]) < 0.9
    ref[m,0] = 1.0; ref[~m,1] = 1.0
    a = np.cross(d, ref); a /= np.linalg.norm(a,axis=1)[:,None]
    b = np.cross(d, a)
    s = np.sqrt(np.maximum(0, 1-cth*cth))
    nd = cth[:,None]*d + s[:,None]*(np.cos(phi)[:,None]*a + np.sin(phi)[:,None]*b)
    nd /= np.linalg.norm(nd,axis=1)[:,None]
    return nd

def rayleigh_cos(n, rng):
    out = np.empty(n); todo = np.arange(n)
    while todo.size:
        c = 2*rng.random(todo.size)-1
        acc = rng.random(todo.size)*2 < (1+c*c)
        out[todo[acc]] = c[acc]; todo = todo[~acc]
    return out

def overlap(px,py,pz,dx,dy,dz,L,zlo,zhi,R=1.0):
    """length of ray p+t d, t in [0,L], inside cylinder radius R (axis z) and slab zlo..zhi"""
    n = px.size
    lo = np.zeros(n); hi = L.copy()
    with np.errstate(divide='ignore', invalid='ignore'):
        ta = (zlo-pz)/dz; tb = (zhi-pz)/dz
    t1 = np.minimum(ta,tb); t2 = np.maximum(ta,tb)
    small = np.abs(dz) < 1e-12
    inside = (pz>=zlo)&(pz<=zhi)
    t1 = np.where(small, np.where(inside,-np.inf,np.inf), t1)
    t2 = np.where(small, np.where(inside,np.inf,-np.inf), t2)
    lo = np.maximum(lo, t1); hi = np.minimum(hi, t2)
    a = dx*dx+dy*dy; b = 2*(px*dx+py*dy); c = px*px+py*py-R*R
    sm = a < 1e-14
    disc = b*b-4*a*c
    sq = np.sqrt(np.maximum(disc,0))
    with np.errstate(divide='ignore', invalid='ignore'):
        r1 = (-b-sq)/(2*a); r2 = (-b+sq)/(2*a)
    r1 = np.where(sm, np.where(c<=0,-np.inf,np.inf), r1)
    r2 = np.where(sm, np.where(c<=0,np.inf,-np.inf), r2)
    r1 = np.where(disc<0, np.inf, r1); r2 = np.where(disc<0, -np.inf, r2)
    lo = np.maximum(lo, r1); hi = np.minimum(hi, r2)
    return np.maximum(0.0, hi-lo)
'''
import os
os.makedirs('/tmp/work',exist_ok=True)
open('/tmp/work/phys.py','w').write(code)
import sys; sys.path.insert(0,'/tmp/work')
import importlib, phys
importlib.reload(phys)
import numpy as np
E=np.array([1.1732,1.3325])
print(phys.mu_tot(E), phys.mu_pair(E), phys.mu_coh(E)/phys.mu_tot(E))
print(phys.eps_mean(E))
