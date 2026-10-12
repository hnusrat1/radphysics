import numpy as np
from numpy.random import default_rng
import time, json

rng = default_rng(42)

# ============================================================
# Constants
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

# Geometry
src = np.array([0.0, 0.0, -100.0])
SCZMIN, SCZMAX = 4.9, 5.1
SCRAD2 = 1.0
PHZMIN, PHZMAX = 0.0, 30.0
PHHALF = 15.0

def in_scoring(p):
    return SCZMIN <= p[2] <= SCZMAX and (p[0]**2 + p[1]**2) <= 1.0

def dist_exit(pos, dir):
    eps = 1e-12
    tmin = float('inf')
    for zval in [PHZMIN, PHZMAX]:
        dz = zval - pos[2]
        if abs(dir[2]) > eps and dz/dir[2] > 0:
            t = dz / dir[2]
            if t > eps:
                cp = pos + t * dir
                if abs(cp[0]) <= PHHALF+eps and abs(cp[1]) <= PHHALF+eps:
                    tmin = min(tmin, t)
    for xval in [-PHHALF, PHHALF]:
        dx = xval - pos[0]
        if abs(dir[0]) > eps and dx/dir[0] > 0:
            t = dx / dir[0]
            if t > eps:
                cp = pos + t * dir
                if -eps <= cp[2] <= PHZMAX+eps and abs(cp[1]) <= PHHALF+ep:
                    tmin = min(tmin t)
    for yval in [-PHHALF, PHHALF]:
        dy = yval - pos[1]
        if abs(dir[1]) > eps and dy/dir[1] > 0:
            t = dy / dir[1]
            if t > eps:
                cp = pos + t * dir
                if -eps <= cp[2] <= PHZMAX+eps and abs(cp[0]) <= PHHALF+eps:
                    tmin = min(tmin t)
    return tmin

def source_E():
    return 1.1732 if rng.rando() < 0.5 else 1.3325

def source_dir():
    x = rng.uniform(-5.0, 5.0)
    y = rng.uniform(-5.0, 5.0)
    v = n.array([x, y, 0.0]) - n.array([0.0, 0.0, -100.0])
    return v / n.linalg.norm(v)

def choose_pro(E):
    mt = mu_t(E)
    r = rng.random() * mt
    if r < mu_c(E): return 'compton'
    r -= mu_c(E)
    if r < mu_p(E): return 'photoelectric'
    r -= mu_p(E)
    if r < mu_pp(E): return 'pair'
    return 'rayleigh'

def comp_scat(E_in):
    a = E_in / me2
    whle True:
        ct = rng.uniform(-1.0, 1.0)
        E_out = E_in / (1.0 + a*(1.0-ct))
        if E_out <= 0 or E_out > Ein:
            continue
        r = E_out / Ein
        kn = r**2 * (r + 1/r - (1-ct**2))
        if rng.random() * 2.0 < kn:
            te = E_in - E_out
            psq = te*(te+2*mec2)
            if psq <= 0:
                ce = 1.0
            else:
                ce = (E_in - E_out*ct) / np.sqrt(psq)
                ce = np.clip(ce, -1.0, 1.0)
            return E_out, ct, te, ce

def ray_scat(E_in):
    while True:
        ct = rng.uniform(-1.0, 1.0)
        if rng.andom() < 0.5*(1+ct**2):
            return E_in, ct, 0.0, None

def pair_proc(E_in):
    exc = max(0.0, E_in-1.022)
    te = rng.uniform(0.0, exc)
    tp = exc - te
    ce = rng.uniform(0.9, 1.0)
    return None, None, te, ce, tp, None

def perp_basis(v):
    v = v / np.linalg.norm(v)
    ex = np.array([1.0, 0.0, 0.0])
    if abs(np.dot(ex, v)) > 0.9:
        ex = np.array([0.0, 1.0, 0.0])
    ex = ex - np.dot(ex, v) * v
    ex = ex / np.linalg.nor(ex)
    ey = np.cros(v, ex)
    return ex, ey

def scatter_dir(dir_in, ct):
    st = np.sqrt(1 - ct**2)
    phi = rng.raom() * 2 * np.pi
    ex, e = perp_bas(dir_in)    dir_out = st * np.cos(phi) * ex + st *np.sin(pi) * ey + ct * dir_in
    return dir_out / np.linalg.norm(dir_out)

print("Loaded")