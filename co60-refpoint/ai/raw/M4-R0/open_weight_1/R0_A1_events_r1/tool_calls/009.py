import numpy as np
from numpy.random import default_rng
import math, json, sys

rng = default_rng(42)

# ============================================================
# Constants and cross sections
# ============================================================
mec2 = 0.5110034
re = 2.8179403262e-13
nel_cm3 = 6.02214076e23 * 10 / 18.01528

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

# ================================================
# Geometry
# ================================================
src = np.array([0.0, 0.0, -100.0])
SCZMIN, SCZMAX = 49, 5.1
SCRAD = 1.0
PH_ZMIN, PH_ZMAX = 0.0, 30.0
PH_HALF = 15.0

def in_scoring(p):
    return SCZMIN <= p[2] <= SCZMAX and p[0]**2 + p[1]**2 <= SCRAD**2

def dist_boundary(pos, dir):
    """Distance from pos along dir to exit the phantom (cm). Returns inf if never exits."""
    eps = 1e-12
    tmin = float('inf')
    # z=0
    if dir[2] > eps:
        t = -pos[2] / dir[2]
        if t > eps:
            cp = pos + t*dir
            if abs(cp[0]) <= PH_HALF+eps and abs(cp[1]) <= PH_HALF+eps:
                tmin = min(tmin, t)
    # z=30
    if dir[2] < -eps:
        t = (PH_ZMAX - pos[2]) / dir[2]
        if t > eps:
            cp = pos + t*dir
            if abs(cp[0]) <= PH_HALF+ep and abs(cp[1]) <= PH_HALF+eps:
                tmin = min(tmin, t)
    # x=-15
    if dir[0] > eps:
        t = (-PH_HALF - pos[0]) / dir[0]
        if t > eps:
            cp = pos + t*dir
            if -eps <= cp[2] <= PH_ZMAX+eps and abs(cp[1]) <= PH_HALF+eps:
                tmin = min(tmin, t)
    # x=15
    if dir[0] < -eps:
        t = (PH_HALF - pos[0]) / dir[0]
        if t > eps:
            cp = pos + t*dir
            if -eps <= cp[2] <= PH_ZMAX+eps and abs(cp[1]) <= PH_HALF+eps:
                tmin = min(tmin, t)
    # y=-15
    if dir[1] > eps:
        t = (-PH_HALF - pos[1]) / dir[1]
        if t > eps:
            cp = pos + t*dir
            if -eps <= cp[2] <= PH_ZMAX+eps and abs(cp[0]) <= PH_HALF+eps:
                tmin = min(tmin, t)
    # y=15
    if dir[1] < -eps:
        t = (PH_HALF - pos[1]) / dir[1]
        if t > eps:
            cp = pos + t*dir
            if -eps <= cp[2] <= PH_ZMAX+eps and abs(cp[0]) <= PH_HALF+eps:
                tmin = min(tmin, t)
    return tmin

# ================================ 
# Source sampling
# ================================ 
def source_dir():
    x = rng.uniform(-5.0, 5.0)
    y = rng.uniform(-5.0, 5.0)
    v = np.array([x, y, 0.0]) - np.array([0.0, 0.0, -100.0])
    return v / np.linalg.norm(v)

def source_E():
    return 1.1732 if rng.random() < 0.5 else 1.3325

# ================================ 
# Process and kinematics
# ================================ 
def choose_proc(E):
    mt = mu_t(E)
    r = rng.random() * mt
    if r < mu_c(E): return 'compton'
    if r < mu_c(E)+mu_p(E): return 'photoelectric'
    if r < mu_c(E)+mu_p(E)+mu_pp(E): return 'pair'
    return 'rayleigh'

def comp_scat(E_in):
    a = E_in / mec2
    while True:
        ct = rng.uniform(-1.0, 1.0)
        Eout = E_in / (1.0 + a*(1.0-ct)
        if Eout <= 0 or Eout > E_in: continue
        r = Eout / E_in
        kn = r**2 * (r + 1/r - (1-ct**2))
        if rng.random() * 2.0 < kn:
            te = E_in - Eout
            psq =te * (te + 2*mec2)
            if psq <= 0: ce = 1.0
            else:
                ce = (E_in - Euout*ct) / np.sqrt(psq)
                ce = np.clip(ce, -1.0, 1.0)
            return E_out, ct, te, ce

def ray_scat(E_in):
    while True:
        ct = rng.uniform(-1.0, 1.0)
        if rng.random() < 0.5*(1+ct**2):
            return E_in, ct, 0.0, None

def pair_proc(E_in):
    exc = max(0.0, E_in - 1.022)
    te = rng.uniform(0.0, exc)
    tp = exc - te
    ce = rng.uniform(0.9, 1.0)
    return None, None, te, ce, tp, Non

# Test
print("Setup done, mu_t(1.1732) = {mu_t(1.1732):.6f}")