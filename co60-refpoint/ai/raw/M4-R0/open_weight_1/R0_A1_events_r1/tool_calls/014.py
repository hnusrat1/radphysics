import numpy as np
from numpy.random import default_rng
import time

rng = default_rng(42)

# ============================================================
# Physics constants
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

# ============================================================
# Geometry: phantom 30x30x30 at z=0..30, x,y in [-15,15]
# Source at (0,0,-100)
# Scoring: cylinder radius 1.0, z=4.9..5.1
# ============================================================
src = np.array([0.0, 0.0, -100.0])
SCZMIN, SCZMAX = 4.9, 5.1
SCRAD = 1.0
PHZMIN, PHZMAX = 0.0, 30.0
PHHALF = 15.0

def in_scoring(p):
    return SCZMIN <= p[2] <= SCZMAX and (p[0]**2 + p[1]**2) <= SCRAD**2

def dist_exit(pos, dir):
    """Distance along dir from pos to exit phantom, or inf if no exit."""
    eps = 1e-12
    tmin = float('inf')
    # Check z boundaries
    for zval, znorm in [(PHZMIN, -1.0), (PHZMAX, 1.0)]:
        # Direction should point towards the boundary
        dz = zval - pos[2]
        if abs(dir[2]) > eps and dz/dir[2] > 0:
            t = dz / dir[2]
            if t > eps:
                cp = pos + t * dir
                if abs(cp[0]) <= PHHALF+eps and abs(cp[1]) <= PHHALF+eps:
                    tmin = min(tmin, t)
    # Check x boundaries
    for xval in [-PHHALF, PHHALF]:
        dx = xval - pos[0]
        if abs(dir[0]) > eps and dx/dir[0] > 0:
            t = dx / dir[0]
            if t > eps:
                cp = pos + t * dir
                if cp[2] >= -eps and cp[2] <= PHZMAX+eps and abs(cp[1]) <= PHHALF+eps:
                    tmin = min(tmin, t)
    # Check y boundaries
    for yval in [-PHHALF, PHHALF]:
        dy = yval - pos[1]
        if abs(dir[1]) > eps and dy/dir[1] > 0:
            t = dy / dir[1]
            if t > eps:
                cp = pos + t * dir
                if cp[2] >= -eps and cp[2] <= PHZMAX+eps and abs(cp[0]) <= PHHALF+eps:
                    tmin = min(tmin, t)
    return tmin

# Test geometry
pos_test = np.array([0.0, 0.0, 0.0])
dir_test = np.array([0.0, 0.0, 1.0])
print("Exit dist from surface:", dist_exit(pos_test, dir_test))
pos_test2 = np.array([0.0, 0.0, 5.0])
print("Exit dist from 5cm depth:", dist_exit(pos_test2, dir_test))
print("In scoring at (0,0,5):", in_scoring(np.array([0.0, 0.0, 5.0])))
print("In scoring at (0.5,0.5,5):", in_scoring(np.array([0.5, 0.5, 5.0])))
print("In scoring at (1.1,0,5):", in_scoring(np.array([1.1, 0.0, 5.0])))