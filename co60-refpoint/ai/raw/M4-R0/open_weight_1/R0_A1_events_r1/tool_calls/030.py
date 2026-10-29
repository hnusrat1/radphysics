import numpy as np
from numpy.random import default_rng
import time, json

rng = default_rng(42)

# ============================================================
# CONSTANTS
# ============================================================
mec2 = 0.5110034        # electron rest mass (MeV)
re = 2.8179e-13         # classical electron radius (cm)
nel_cm3 = 6.022e23 * 10 / 18.015  # e-/cm^3 in water

# ============================================================
# CROSS SECTIONS
# ============================================================
def sigma_KN(E):
    a = E / mec2
    t = ((1+a)/a**2 * (2*(1+a)/(1+2*a) - np.log(1+2*a)/a) +
         np.log(1+2*a)/(2*a) - (1+3*a)/(1+2*a)**2)
    return 2*np.pi*re**2 * t

def mu_c(E): return nel_cm3 * sigma_KN(E)
def mu_p(E): return 0.00057 * (1/E)**3.5
def mu_pp(E):
    if E <= 1.022: return 0.0
    return 0.00035 * (E-1.022)/(1.3325-1.022)
def mu_r(E): return 0.00033 * (1/E)**2
def mu_t(E): return mu_c(E)+mu_p(E)+mu_pp(E)+mu_r(E)

# ============================================================
# GEOMETRY
# ============================================================
src = np.array([0.0, 0.0, -100.0])
# Phantom: z in [0,30], x,y in [-15,15]
# Scoring: cylinder radius 1.0 cm, z in [4.9,5.1]

def in_scoring(p):
    return 4.9 <= p[2] <= 5.1 and (p[0]**2 + p[1]**2) <= 1.0

def dist_exit(pos, di):
    """Distance from pos along direction di to exit the phantom"""
    eps = 1e-12
    tmin = float('inf')
    # z boundaries
    for zval in [0.0, 30.0]:
        dz = zval - pos[2]
        if abs(di[2]) > eps and dz/di[2] > 0:
            t = dz / di[2]
            if t > eps:
                cp = pos + t * di
                if abs(cp[0]) <= 15+eps and abs(cp[1]) <= 15+eps:
                    tmin = min(tmin, t)
    # x boundaries
    for xval in [-15.0, 15.0]:
        dx = xval - pos[0]
        if abs(di[0]) > eps and dx/di[0] > 0:
            t = dx / di[0]
            if t > eps:
                cp = pos + t * di
                if -eps <= cp[2] <= 30+eps and abs(cp[1]) <= 15+eps:
                    tmin = min(tmin, t)
    # y boundaries
    for yval in [-15.0, 15.0]:
        dy = yval - pos[1]
        if abs(di[1]) > eps and dy/di[1] > 0:
            t = dy / di[1]
            if t > eps:
                cp = pos + t * di
                if -eps <= cp[2] <= 30+eps and abs(cp[0]) <= 15+eps:
                    tmin = min(tmin, t)
    return tmin

# ============================================================
# SOURCE
# ============================================================
def source_E():
    return 1.1732 if rng.random() < 0.5 else 1.3325

def source_dir():
    x = rng.uniform(-5.0, 5.0)
    y = rng.uniform(-5.0, 5.0)
    return np.array([x, y, 100.0]) / np.sqrt(x**2 + y**2 + 10000.0)

# ============================================================
# PROCESS SELECTION & KINEMATICS
# ============================================================
def choose_proc(E):
    mt = mu_t(E)
    r = rng.random() * mt
    if r < mu_c(E): return 'compton'
    r -= mu_c(E)
    if r < mu_p(E): return 'photoelectric'
    r -= mu_p(E)
    if r < mu_pp(E): return 'pair'
    return 'rayleigh'

def comp_scat(E_in):
    a = E_in / mec2
    while True:
        ct = rng.uniform(-1.0, 1.0)
        E_out = E_in / (1.0 + a*(1.0-ct))
        if E_out <= 0 or E_out > E_in:
            continue
        ratio = E_out / E_in
        kn = ratio**2 * (ratio + 1/ratio - (1-ct**2))
        if rng.random() * 2.0 < kn:
            te = E_in - E_out
            psq = te*(te+2*mec2)
            if psq <= 0:
                ce = 1.0
            else:
                ce = (E_in - E_out*ct)/np.sqrt(psq)
                ce = np.clip(ce, -1.0, 1.0)
            return E_out, ct, te, ce

def ray_scat(E_in):
    while True:
        ct = rng.uniform(-1.0, 1.0)
        if rng.random() < 0.5*(1+ct**2):
            return E_in, ct, 0.0, None

def pair_proc(E_in):
    exc = max(0.0, E_in-1.022)
    te = rng.uniform(0.0, exc)
    tp = exc - te
    ce = rng.uniform(0.9, 1.0)
    return Non, Non, te, ce, tp, None

def perp_basis(v):
    v = v/np.linalg.norm(v)
    ex = np.array([1.,0.,0.])
    if abs(np.dot(ex,v)) > 0.9:
        ex = np.array([0.,1.,0.])
    ex = ex - np.dot(ex,v)*v
    ex = ex/np.linalg.norm(ex)
    ey = np.cross(v,ex)
    return ex, ey

def scatter_dir(di, ct):
    st = np.sqrt(1-ct**2)
    phi = rng.random()*2*np.pi
    ex, ey = pep_basis(di)
    return (st*np.cos(phi)*ex + st*np.sin(phi)*ey + ct*di)/np.linalg.norm(st*np.cos(phi)*ex + st*np.sin(h)*ey + ct*di)

print("All functions defined. Running simulation...")