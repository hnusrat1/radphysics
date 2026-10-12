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
H2O_rho = 1.0
n_e_cm3 = N_A * 10 * H2O_rho / H2O_M

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

def mu_comp(E): return n_e_cm3 * sigma_KN(E)

def mu_photo(E):
    # at 1 MeV: ~0.00057, scaling ~ E^-3.5 from ~0.05 MeV up
    return 0.00057 * (1.0 / E)**3.5

def mu_pair(E):
    if E <= 1.022: return 0.0
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
            -PHANTOM_HALF <= p[0] <= PHANTOM_HALF and
            -PHANTOM_HALF <= p[1] <= PHANTOM_HALF)

def in_scoring(p):
    if not (sc_zmin <= p[2] <= sc_zmax):
        return False
    return np.sqrt(p[0]**2 + p[1]**2) <= sc_rad

def distance_to_exit(pos, dir):
    """Distance along dir to exit phantom"""
    # Check all 6 planes
    t_min = floatt('inf')
    # z=0 plane (entrance)
    if dir[2] > 0:
        t = (0.0 - pos[2]) / dir[2]
        if t >1e-12:
            p = pos + t * dir
            if abs(p[0]) <= PHANTOM_HALF and abs(p[1]) <= PHANTOM_HALF:
                t_min = min(t_min, t)
    # z=30 plane
    if dir[2] < 0:
        t = (PHANTOM_ZMAX - pos[2]) / dir[2]
        if t > 1e-12:
            p = pos + t * dir
            if abs(p[0]) <= PHANTOM_HALF and abs(p[1]) <= PHANTOM_HALF:
                t_min = min(t_min, t)
    # x=+15
    if dir[0] > 0:
        t = (PHANTOM_HALF - pos[0]) / dir[0]
        if t > 1e-12:
            p = pos + t * dir
            if PHANTOM_ZMIN <= p[2] <= PHANTOM_ZMAX and abs(p[1]) <= PHANTOM_HALF:
                t_min = min(t_min, t)
    # x=-15
    if dir[0] < 0:
        t = (-PHANTOM_HALF - pos[0]) / dir[0]
        if t > 1e-12:
            p = pos + t * dir
            if PHANTOM_ZMIN <= p[2] <= PHANTOM_ZMAX and abs(p[1]) <= PHANTOM_HALF:
                t_min = min(t_min, t)
    # y=+15
    if dir[1] > 0:
        t = (PHANTOM_HALF - pos[1]) / dir[1]
        if t > 1e-12:
            p = pos + t * dir
            if PHANTOM_ZMIN <= p[2] <= PHANTOM_ZMAX and abs(p[0]) <= PHANTOM_HALF:
                t_min = min(t_min, t)
    # y=-15
    if dir[1] < 0:
        t = (-PHANTOM_HALF - pos[1]) / dir[1]
        if t > 1e-12:
            p = pos + t * dir
            if PHANTOM_ZMIN <= p[2] <= PHANTOM_ZMAX and abs(p[0]) <= PHANTOM_HALF:
                t_min = min(t_min, t)
    return t_min

# ============================================================
# Source sampling
# ============================================================
def sample_source_direction():
    """Returns direction unit vector from source towards field at z=0"""
    # Sample uniformly in solid angle within the pyramid
    # Uniform in area at phantom surface is accurate enough for small field
    x = rng.uniform(-field_half, field_half)
    y = rng.uniform(-field_half, field_half)
    v = np.array([x, y, 0.0]) - src  # src at (0,0,-100)
    return v / np.linalg.norm(v)

def sample_energy():
    """Return photon energy in MeV, equally likely 1.1732 or 1.3325"""
    return 1.1732 if rng.random() < 0.5 else 1.3325

# ============================================================
# Interaction sampling
# ============================================================
def choose_process(E):
    mc = mu_comp(E)
    mp = mu_photo(E)
    mpr = mu_pair(E)
    mr = mu_rayleigh(E)
    mt = mc + mp + mpr + mr
    r = rng.random() * mt
    if r < mc:
        return 'compton', mt
    elif r < mc + mp:
        return 'photoeletric', mt
    elif r < mc + mp + mpr:
        return 'pair', mt    else:
        return 'rayleigh', mt

# ============================================================ 
# Compton scattering
# ============================================================
def compton_scatter(E_in):
    """Returns (E_out, cos_scatter, t_e, cos_electron)"""
    alpha = E_in / m_e_c2
    # Sample cos(theta) using KN rejection
    max_val = 2.0  # max of f(cosT) at cosT=1
    while True:
        cosT = rng.uniform(-1.0, 1.0)
        E_out = E_in / (1.0 + alpha * (1.0 - cosT))
        if E_out <= 0 or E_out > E_in:
            continue        ratio = E_out / E_in
        sin2 = 1.0 - cosT**2
        kn_val = ratio**2 * (ratio + 1.0/ratio - sin2)
        if rng.random() * max_val < kn_val:
            t_e = E_in - E_out
            # Electron scattering angle
            p_in = E_in            p_out = E_out
            p_e_sq = t_e * (t_e + 2.0 * m_e_c2)
            if p_e_sq <= 0:
                cos_e = 1.0
            else:
                p_e = np.sqrt(p_e_sq)
                cos_e = (p_in - p_out * cosT) / p_e
                cos_e = np.clip(cos_e, -1.0, 1.0)
            return E_out, cosT, t_e, cos_e

# ============================================================
# Rayleigh scattering (elastic, forward-peaked)
# ============================================================
def rayleigh_scatter(E_in):
    """Returns (E_out, cos_scatter, t_e, cos_electron)
    For Rayleigh, E_out = E_in, no electron set in motion.
    Angular distribution proportional to (1+cos²θ) * F(q,Z)².
    For water at MeV energies, use approximate forward-peaked distribution.
    """
    # Approximate Rayleigh angular distribution ~ (1+cos²θ)
    # Sample using rejection
    while True:
        cosT = rng.uniform(-1.0, 1.0)
        # For Rayleigh, the form factor makes it very forward-peaked, but
        # at high energies it approaches (1+cos²θ)/2
        prob = 0.5 * (1.0 + cosT**2)
        if rng.random() < prob:
            return E_in, cosT, 0.0, None

# ============================================================
# Pair production
# ============================================================
def pair_production(E_in):
    """Returns (E_out, cos_scatter, t_e, cos_electron, t_positron, cos_positron)
    Actually for our schema: no outgoing photon, so E_out=none, cos_scatter=none.
    t_e = electron kinetic energy, t_positron = positron kinetic energy.
    cos_electron: electron direction relative to incident photon.
    For pair production at MeV energies, both particles go forward.
    """
    if E_in <= 1.022:
        return None  # not possible
    excess = E_in - 1.022
    # Approx: energy shared roughly equally, but slightly favoring electron
    # Actually for pair production, the energy distribution of e- and e+ is:
    # dσ/dE_e is roughly constant from 0 to excess
    # But we can just split equally for simplicity
    t_e = rng.uniform(0.0, excess)
    t_pos = excess - t_e
    
    # Both particles are emitted strongly forward at these energies
    # Approx: polar angle ~ m_e_c2 / E_particle
    # For simplicity, sample cos ~ uniform in [0.9, 1.0]
    cos_e = rng.uniform(0.9, 1.0)
    cos_pos = rng.uniform(0.9, 1.0)
    # Actually for this data schema, we only report t_e and cos_electron
    # t_positron_MeV is separate field
    return None, None, t_e, cos_e, t_pos, cos_pos

# ================== 
# Test run to see if sampling works
print("Testing cross sections at E=1.1732 MeV:")
print(f"  mu_comp={mu_comp(1.1732):.6f}")
print(f"  mu_photo={mu_photo(1.1732):.6f}")
print(f"  mu_pair={mu_pair(1.1732):.6f}")
print(f"  mu_rayleigh={mu_rayleigh(1.1732):.6f}")
print(f"  mu_total={mu_total(1.1732):.6f}")

# Test Compton sampling
print("\nCompton test:")
for _ in range(3):
    res = compton_scatter(1.1732)
    print(f"  E_out={res[0]:.4f}, cosT={res[1]:.4f}, t_e={res[2]:.4f}, cos_e={res[3]:.4f}")

# Estimate number of primaries needed
# Primary fluence at scoring vol = Phi0 * exp(-mu*5)
# prob a primary photon interacts in scoring vol:
# area fraction * exp(-mu*5) * (1 - exp(-mu*0.2))
# area_frac = pi*1^2 / (10.5^2) = 0.0285
area_frac = np.pi * sc_rad**2 / (10.5**2)
mu_avg = mu_total(1.25)
prob_per_primary = area_frac * np.exp(-mu_avg * sc_z0) * (1 - np.exp(-mu_avg * 2*sc_half))
print(f"\nEstimated prob of primary interaction per source photon: (prob_per_primary:.6f}")
print(f"Estimated source photons needed for 200 primaries: {200/prob_per_primary:.0f}")