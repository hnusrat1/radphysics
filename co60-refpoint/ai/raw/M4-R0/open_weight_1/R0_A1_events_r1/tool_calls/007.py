import numpy as np
from numpy.random import default_rng
import math

rng = default_rng(42)

# ============================================================
# Constants
# ============================================================
m_e_c2 = 0.5110034
r_e = 2.8179403262e-13
N_A = 6.02214076e23
H2O_M = 18.01528
n_e_cm3 = N_A * 10 / H2O_M  # electrons per cm^3 (rho=1)

# Geometry
SSD = 100.0
field_half = 5.0
src = np.array([0.0, 0.0, -SSD])

PHANTOM_ZMIN = 0.0
PHANTOM_ZMAX = 30.0
PHANTOM_HALF = 15.0

sc_z0 = 5.0
sc_rad = 1.0
sc_half = 0.1
sc_zmin = sc_z0 - sc_half
sc_zmax = sc_z0 + sc_half

# ============================================================
# Cross sections
# ============================================================
def sigma_KN(E):
    a = E / m_e_c2
    term = ((1+a)/a**2 * (2*(1+a)/(1+2*a) - np.log(1+2*a)/a) +
            np.log(1+2*a)/(2*a) - (1+3*a)/(1+2*a)**2)
    return 2 * np.pi * r_e**2 * term

def mu_comp(E):
    return n_e_cm3 * sigma_KN(E)

def mu_photo(E):
    return 0.00057 * (1.0 / E)**3.5

def mu_pair(E):
    if E <= 1.022:
        return 0.0
    return 0.00035 * (E - 1.022) / (1.3325 - 1.022)

def mu_rayleigh(E):
    return 0.00033 * (1.0 / E)**2

def mu_total(E):
    return mu_comp(E) + mu_photo(E) + mu_pair(E) + mu_rayleigh(E)

# ============================================================
# Geometry utilities
# ============================================================
def in_phantom(p):
    return (PHANTOM_ZMIN <= p[2] <= PHANTOM_ZMAX and
            abs(p[0]) <= PHANTOM_HALF and abs(p[1]) <= PHANTOM_HALF)

def in_scoring(p):
    if not (sc_zmin <= p[2] <= sc_zmax):
        return False
    return np.sqrt(p[0]**2 + p[1]**2) <= sc_rad

def distance_to_exit(pos, dir):
    """Distance along dir to exit phantom"""
    t_min = float('inf')
    eps = 1e-12
    # z=0 plane
    if dir[2] > eps:
        t = (0.0 - pos[2]) / dir[2]
        if t > eps:
            p = pos + t * dir
            if abs(p[0]) <= PHANTOM_HALF and abs(p[1]) <= PHANTOM_HALF:
                t_min = min(t_min, t)
    # z=30 plane
    if dir[2] < -eps:
        t = (PHANTOM_ZMAX - pos[2]) / dir[2]
        if t > eps:
            p = pos + t * dir
            if abs(p[0]) <= PHANTOM_HALF and abs(p[1]) <= PHANTOM_HALF:
                t_min = min(t_min, t)
    # x=+15
    if dir[0] > eps:
        t = (PHANTOM_HALF - pos[0]) / dir[0]
        if t > eps:
            p = pos + t * dir
            if PHANTOM_ZMIN <= p[2] <= PHANTOM_ZMAX and abs(p[1]) <= PHANTOM_HALF:
                t_min = min(t_min, t)
    # x=-15
    if dir[0] < -eps:
        t = (-PHANTOM_HALF - pos[0]) / dir[0]
        if t > eps:
            p = pos + t * dir
            if PHANTOM_ZMIN <= p[2] <= PHANTOM_ZMAX and abs(p[1]) <= PHANTOM_HALF:
                t_min = min(t_min, t)
    # y=+15
    if dir[1] > eps:
        t = (PHANTOM_HALF - pos[1]) / dir[1]
        if t > eps:
            p = pos + t * dir
            if PHANTOM_ZMIN <= p[2] <= PHANTOM_ZMAX and abs(p[0]) <= PHANTOM_HALF:
                t_min = min(t_min, t)
    # y=-15
    if dir[1] < -eps:
        t = (-PHANTOM_HALF - pos[1]) / dir[1]
        if t > eps:
            p = pos + t * dir
            if PHANTOM_ZMIN <= p[2] <= PHANTOM_ZMAX and abs(p[0]) <= PHANTOM_HALF:
                t_min = min(t_min, t)
    return t_min

# ============================================================
# Source sampling
# ============================================================
def sample_source_direction():
    x = rng.uniform(-field_half, field_half)
    y = rng.uniform(-field_half, field_half)
    v = np.array([x, y, 0.0]) - src
    return v / np.linalg.norm(v)

def sample_energy():
    return 1.1732 if rng.random() < 0.5 else 1.3325

# ============================================================
# Process selection
# ============================================================
def choose_process(E):
    mc = mu_comp(E)
    mp = mu_photo(E)
    mpr = mu_pair(E)
    mr = mu_rayleigh(E)
    mt = mc + mp + mpr + mr
    r = rng.random() * mt
    if r < mc:
        return 'compton'
    elif r < mc + mp:
        return 'photoelectric'
    elif r < mc + mp + mpr:
        return 'pair'
    else:
        return 'rayleigh'

# ============================================================
# Compton scattering
# ============================================================
def compton_scatter(E_in):
    alpha = E_in / m_e_c2
    max_val = 2.0
    while True:
        cosT = rng.uniform(-1.0, 1.0)
        denom = 1.0 + alpha * (1.0 - cosT)
        E_out = E_in / denom
        if E_out <= 0 or E_out > E_in:
            continue
        ratio = E_out / E_in
        sin2 = 1.0 - cosT**2
        kn_val = ratio**2 * (ratio + 1.0/ratio - sin2)
        if rng.random() * max_val < kn_val:
            t_e = E_in - E_out
            # Electron angle
            p_in = E_in
            p_out = E_out
            p_e_sq = t_e * (t_e + 2.0 * m_e_c2)
            if p_e_sq <= 0:
                cos_e = 1.0
            else:
                p_e = np.sqrt(p_e_sq)
                cos_e = (p_in - p_out * cosT) / p_e
                cos_e = np.clip(cos_e, -1.0, 1.0)
            return E_out, cosT, t_e, cos_e

# ============================================================
# Rayleigh scattering
# ============================================================
def rayleigh_scatter(E_in):
    while True:
        cosT = rng.uniform(-1.0, 1.0)
        prob = 0.5 * (1.0 + cosT**2)
        if rng.random() < prob:
            return E_in, cosT, 0.0, None

# ============================================================
# Pair production
# ============================================================
def pair_production(E_in):
    excess = E_in - 1.022
    if excess <= 0:
        return None, None, 0.0, 0.0, 0.0, 0.0
    t_e = rng.uniform(0.0, excess)
    t_pos = excess - t_e
    cos_e = rng.uniform(0.9, 1.0)
    cos_pos = rng.uniform(0.9, 1.0)
    return None, None, t_e, cos_e, t_pos cos_pos

# ===============================
# Test
print("Testing...")
print(f"mu_comp(1.1732) = {mu_comp(1.1732):.6f}")
print(f"mu_photo(1.1732) = {mu_photo(1.1732):.6f}")
print(f"mu_pair(1.1732) = {mu_pair(1.1732):.6f}")