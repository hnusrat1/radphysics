import numpy as np
from numpy.random import default_rng
import math

rng = default_rng(42)

# ============================================================
# Constants and cross sections
# ============================================================
mec2 = 0.5110034
re = 2.8179403262e-13
NA = 6.02214076e23
nel_cm3 = NA * 10 / 18.01528  # electrons per cm^3 in H2O, rho=1

def sigma_KN(E):
    a = E / mec2
    term = ((1+a)/a**2 * (2*(1+a)/(1+2*a) - np.log(1+2*a)/a) +
            np.log(1+2*a)/(2*a) - (1+3*a)/(1+2*a)**2)
    return 2 * np.pi * re**2 * term

def mu_c(E): return nel_cm3 * sigma_KN(E)
def mu_p(E): return 0.00057 * (1.0/E)**3.5
def mu_pp(E):
    if E <= 1.022: return 0.0
    return 0.00035 * (E-1.022)/(1.3325-1.022)
def mu_r(E): return 0.00033 * (1.0/E)**2
def mu_t(E): return mu_c(E)+mu_p(E)+mu_pp(E)+mu_r(E)

# ============================================================
# Geometry
# ============================================================
src = np.array([0.0, 0.0, -100.0])
SCZMIN, SCZMAX = 4.9, 5.1
SCRAD = 1.0
PH_ZMIN, PH_ZMAX = 0.0, 30.0
PH_HALF = 15.0

def in_scoring(p):
    return SCZMIN <= p[2] <= SCZMAX and p[0]**2 + p[1]**2 <= SCRAD**2

def in_phantom(p):
    return PH_ZMIN <= p[2] <= PH_ZMAX and abs(p[0]) <= PH_HALF and abs(p[1]) <= PH_HALF

def dist_to_exit(pos, dir):
    eps = 1e-12
    tmin = float('inf')
    # Check all 6 phantom boundaries
    for plane_coord, dir_idx, plane_val, dim1, dim2 in [
        (PH_ZMIN, 2, 0, 0, 1), (PH_ZMAX, 2, 0, 0, 1),
        (-PH_HALF, 0, 0, 1, 2), (PH_HALF, 0, 0, 1, 2),
        (-PH_HALF, 1, 0, 0, 2), (PH_HALF, 1, 0, 0, 2)
    ]:
        pass  # Too complex, let's do it more simply

# Simpler version
def dist_to_boundary(pos, dir):
    """Distance from pos in dir to exit phantom, or inf if never exits"""
    eps = 1e-12
    tmin = float('inf')
    
    # z = 0 (entrance)
    if dir[2] > eps:
        t = -pos[2] / dir[2]
        if t > eps:
            cp = pos + t*dir
            if abs(cp[0]) <= PH_HALF+eps and abs(cp[1]) <= PH_HALF+eps:
                tmin = min(tmin, t)
    # z = 30
    if dir[2] < -eps:
        t = (PH_ZMAX-pos[2])/dir[2]
        if t > eps:
            cp = pos + t*dir
            if abs(cp[0]) <= PH_HALF+eps and abs(cp[1]) <= PH_HALF+eps:
                tmin = min(tmin, t)
    # x = -15
    if dir[0] > eps:
        t = (-PH_HALF-pos[0])/dir[0]
        if t > eps:
            cp = pos + t*dir
            if 0-eps <= cp[2] <= PH_ZMAX+eps and abs(cp[1]) <= PH_HALF+eps:
                tmin = min(tmin, t)
    # x = 15
    if dir[0] < -eps:
        t = (PH_HALF-pos[0])/dir[0]
        if t > eps:
            cp = pos + t*dir
            if 0-eps <= cp[2] <= PH_ZMAX+eps and abs(cp[1]) <= PH_HALF+eps:
                tmin = min(tmin, t)
    # y = -15
    if dir[1] > eps:
        t = (-PH_HALF-pos[1])/dir[1]
        if t > eps:
            cp = pos + t*dir
            if 0-eps <= cp[2] <= PH_ZMAX+eps and abs(cp[0]) <= PH_HALF+eps:
                tmin = min(tmin, t)
    # y = 15
    if dir[1] < -eps:
        t = (PH_HALF-pos[1])/dir[1]
        if t > eps:
            cp = pos + t*dir
            if 0-eps <= cp[2] <= PH_ZMAX+eps and abs(cp[0]) <= PH_HALF+eps:
                tmin = min(tmin, t)
    return tmin

# ============================================================
# Source
# ============================================================
def source_dir():
    x = rng.uniform(-5.0, 5.0)
    y = rng.uniform(-5.0, 5.0)
    v = np.array([x, y, 0.0]) - src
    return v / np.linalg.norm(v)

def source_E():
    return 1.1732 if rng.random() < 0.5 else 1.3325

# ============================================================
# Processes
# ============================================================
def choose_proc(E):
    r = rng.random() * mu_t(E)
    if r < mu_c(E): return 'compton'
    if r < mu_c(E)+mu_p(E): return 'photoelectric'
    if r < mu_c(E)+mu_p(E)+mu_pp(E): return 'pair'
    return 'rayleigh'

def comp_kn(E_in):
    a = E_in / mec2
    while True:
        ct = rng.uniform(-1.0, 1.0)
        Eout = E_in / (1.0 + a*(1.0-ct))
        if Eout <= 0 or Eout > E_in: continue
        r = Eout / E_in
        kn = r**2 * (r + 1/r - (1-ct**2))
        if rng.random() * 2.0 < kn:
            te = E_in - Eout
            psq = te * (te + 2*mec2)
            if psq <= 0: ce = 1.0
            else:
                ce = (E_in - Eout*ct) / np.sqrt(psq)
                ce = np.clip(ce, -1.0,1.0)
            return Eout, ct, te, ce

def rayl_scat(E_in):
    while True:
        ct = rng.uniform(-1.0,1.0)
        if rng.random() < 0.5*(1+ct**2):
            return E_in, ct, 0.0, None

def pair_proc(E_in):
    exc = max(0.0, E_in - 1.022)
    te = rng.uniform(0.0, exc)
    tp = exc - te
    ce = rng.uniform(0.9, 1.0)
    # We only need t_e, cos_electron, t_positron in the fields
    return None, None, te, ce, tp, None

# Test
print("Cross sections:")
for E in [1.1732, 1.3325]:
    print(f"E={E:.4f}: comp={mu_c(E):.6f}, photo={mu_p(E):.6f}, pair={mu_pp(E):.6f}, ray={mu_r(E):.6f}, tot={mu_t(E):.6f}")

print("\nCompton test:")
for _ in range(3):
    eo, ct, te, ce = comp_kn(1.1732)
    print(f"  Eout={eo:.4f}, cos={ct:.4f}, te={te:.4f}, ce={ce:.4f}")