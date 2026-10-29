import numpy as np
from numpy.random import default_rng
import time

rng = default_rng(42)

# ============================================================
# Physics
# ============================================================
mec2 = 0.5110034
re = 2.8179e-13
nel_cm3 = 6.022e23 * 10 / 18.015

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
def in_scoring(p):
    return 4.9 <= p[2] <= 5.1 and (p[0]**2 + p[1]**2) <= 1.0

def dist_exit(pos, di):
    """Distance from pos along di to exit 30x30x30 phantom at [0,30]x[-15,15]x[-15,15]"""
    e = 1e-12
    tmin = float('inf')
    for zval in [0.0, 30.0]:
        dz = zval - pos[2]
        if abs(di[2]) > e and dz/di[2] > 0:
            t = dz / di[2]
            if t > e:
                cp = pos + t * di
                if abs(cp[0]) <= 15+e and abs(cp[1]) <= 15+e:
                    tmin = min(tmin, t)
    for xval in [-15.0, 15.0]:
        dx = xval - pos[0]
        if abs(di[0]) > e and dx/di[0] > 0:
            t = dx / di[0]
            if t > e:
                cp = pos + t * di
                if -e <= cp[2] <= 30+e and abs(cp[1]) <= 15+e:
                    tmin = min(tmin, t)
    for yval in [-15.0, 15.0]:
        dy = yval - pos[1]
        if abs(di[1]) > e and dy/di[1] > 0:
            t = dy / di[1]
            if t > e:
                cp = pos + t * di
                if -e <= cp[2] <= 30+e and abs(cp[0]) <= 15+e:
                    tmin = min(tmin, t)
    return tmin

def source_E():
    return 1.1732 if rng.random() < 0.5 else 1.3325

def source_dir():
    x = rng.uniform(-5.0, 5.0)
    y = rng.uniform(-5.0, 5.0)
    di = np.array([x, y, 100.0])
    return di / np.linalg.norm(di)

def choos_proc(E):
    mt = mu_t(E)
    r = rng.random() * mt
    if r < mu_c(E): return 'compton'
    r -= mu_c(E)
    if r < mu_p(E): return 'photoelectric'
    r -= mu_p(E)
    if r < mu_pp(E): return 'pair'
    return 'rayleigh'

print("Basic functions defined")