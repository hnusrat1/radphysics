import numpy as np
from numpy.random import default_rng

rng = default_rng(42)

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
SCRAD = 1.0
SCRAD2 = SCRAD**2
PHZMIN, PHZMAX = 0.0, 30.0
PHHALF = 15.0

def in_scoring(p):
    return SCZMIN <= p[2] <= SCZMAX and (p[0]**2 + p[1]**2) <= SCRAD2

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
                if cp[2] >= -eps and cp[2] <= PHZMAX+eps and abs(cp[1]) <= PHHALF+eps:
                    tmin = min(tmin, t)
    for yval in [-PHHALF, PHHALF]:
        dy = yval - pos[1]
        if abs(dir[1]) > eps and dy/dir[1] > 0:
            t = dy / dir[1]
            if t > eps:
                cp = pos + t * dir
                if cp[2] >= -eps and cp[2] <= PHZMAX+eps and abs(cp[0]) <= PHHALF+eps:
                    tmin = min(tmin, t)
    return tmin

# Source
def source_E():
    return 1.1732 if rng.random() < 0.5 else 1.3325

def source_dir():
    x = rng.uniform(-5.0, 5.0)
    y = rng.uniform(-5.0, 5.0)
    v = np.array([x, y, 0.0]) - src
    return v / np.linalg.norm(v)

def choose_proc(E):
    mt = mu_t(E)
    r = rng.random() * mt
    if r < mu_c(E):
        return 'compton'
    r -= mu_c(E)
    if r < mu_p(E):
        return 'photoelectric'
    r -= mu_p(E)
    if r < mu_pp(E):
        return 'pair'
    return 'rayleigh'

def comp_scat(E_in):
    a = E_in / mec2
    while True:
        ct = rng.uniform(-1.0, 1.0)
        E_out = E_in / (1.0 + a * (1.0 - ct))
        if E_out <= 0 or E_out > E_in:
            continue
        r = E_out / E_in
        kn = r**2 * (r + 1.0/r - (1.0 - ct**2))
        if rng.random() * 2.0 < kn:
            t_e = E_in - E_out
            psq = t_e * (t_e + 2.0 * mec2)
            if psq <= 0:
                cos_e = 1.0
            else:
                cos_e = (E_in - E_out * ct) / np.sqrt(psq)
                cos_e = np.clip(cos_e, -1.0, 1.0)
            return E_out, ct, t_e, cos_e

def ray_scat(E_in):
    while True:
        ct = rng.uniform(-1.0, 1.0)
        if rng.random() < 0.5 * (1.0 + ct**2):
            return E_in, ct, 0.0, None

def pair_proc(E_in):
    exc = max(0.0, E_in - 1.022)
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
    ex = ex / np.linalg.norm(ex)
    ey = np.cross(v, ex)
    return ex, ey

def scatter_dir(dir_in, ct):
    st = np.sqrt(1.0 - ct**2)
    phi = rng.random() * 2 * np.pi
    ex, ey = perp_basis(dir_in)
    dir_out = st * np.cos(phi) * ex + st * np.sin(phi) * ey + ct * dir_in
    return dir_out / np.linalg.norm(dir_out)

print("All functions defined. Testing...")

# Test processes
for _ in range(3):
    proc = choose_proc(1.1732)
    print(f"Process: {proc}")
    
# Test Compton
eout, ct, te, ce = comp_scat(1.1732)
print(f"Compton: Eout={eout:.4f}, cos={ct:.4f}, te={te:.4f}, ce={ce:.4f}")

# Test Rayleigh
eout, ct, te, ce = ray_scat(1.1732)
print(f"Rayleigh: Eout={eout:.4f}, cos={ct:.4f}")

# Test pair
res = pair_proc(1.3325)
print(f"Pair: te={res[2]:.4f}, ce={res[3]:.4f}, tp={res[4]:.4f}")