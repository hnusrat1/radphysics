import numpy as np
from numpy.random import default_rng
import math

rng = default_rng(42)

# ============================================================
# Constants
# ============================================================
m_e_c2 = 0.5110034   # MeV
r_e = 2.8179403262e-13  # cm
N_A = 6.02214076e23
H2O_M = 18.01528      # g/mol
H2O_rho = 1.0         # g/cm^3
n_e_cm3 = N_A * 10 * H2O_rho / H2O_M

# Geometry
SSD = 100.0
field_half = 5.0      # half field size at surface

# Source at (0,0,-100)
src = np.array([0.0, 0.0, -SSD])

# Phantom: 30x30x30 from z=0 to z=30, x in [-15,15], y in [-15,15]
PHANTOM_ZMIN = 0.0
PHANTOM_ZMAX = 30.0
PHANTOM_HALF = 15.0

# Scoring volume: cylinder on axis
sc_z0 = 5.0
sc_rad = 1.0
sc_half = 0.1   # half thickness
sc_zmin = sc_z0 - sc_half
sc_zmax = sc_z0 + sc_half

# ============================================================
# Cross sections (cm^-1)
# ============================================================
def sigma_KN(E):
    a = E / m_e_c2
    term = ((1+a)/a**2 * (2*(1+a)/(1+2*a) - np.log(1+2*a)/a) +
            np.log(1+2*a)/(2*a) - (1+3*a)/(1+2*a)**2)
    return 2 * np.pi * r_e**2 * term

def mu_comp(E): return n_e_cm3 * sigma_KN(E)

def mu_photo(E): return 0.00057 * (1.0 / E)**3.5

def mu_pair(E):
    if E <= 1.022: return 0.0
    return 0.00035 * (E - 1.022) / (1.3325 - 1.022)

def mu_rayleigh(E): return 0.00033 * (1.0 / E)**2

def mu_total(E):
    return mu_comp(E) + mu_photo(E) + mu_pair(E) + mu_rayleigh(E)

# ============================================================
# Sampling interaction type
# ============================================================
def sample_interaction(E):
    """Return (process_string, mu_total)"""
    mc = mu_comp(E)
    mp = mu_photo(E)
    mpr = mu_pair(E)
    mr = mu_rayleigh(E)
    mt = mc + mp + mpr + mr
    r = rng.random() * mt
    if r < mc:
        return 'compton', mt
    elif r < mc + mp:
        return 'photoelectric', mt
    elif r < mc + mp + mpr:
        return 'pair', mt
    else:
        return 'rayleigh', mt

# ============================================================
# Sampling interaction position along a ray in medium
# ============================================================
def sample_interaction_distance(mu):
    """Sample distance to next interaction in medium (cm)"""
    if mu <= 0:
        return float('inf')
    return -np.log(rng.random()) / mu

# ============================================================
# Geometry helpers
# ============================================================
def point_in_phantom(p):
    return (PHANTOM_ZMIN <= p[2] <= PHANTOM_ZMAX and
            -PHANTOM_HALF <= p[0] <= PHANTOM_HALF and
            -PHANTOM_HALF <= p[1] <= PHANTOM_HALF)

def point_in_scoring(p):
    """Check if point is inside scoring cylinder"""
    if not (sc_zmin <= p[2] <= sc_zmax):
        return False
    r = np.sqrt(p[0]**2 + p[1]**2)
    return r <= sc_rad

def distance_to_phantom_exit(pos, dir_vec):
    """Distance from pos along dir to exit phantom"""
    dists = []
    # z = 0 plane
    if dir_vec[2] > 0:
        d = (0.0 - pos[2]) / dir_vec[2]
        if d > 0:
            p = pos + d * dir_vec
            if -PHANTOM_HALF <= p[0] <= PHANTOM_HALF and -PHANTOM_HALF <= p[1] <= PHANTOM_HALF:
                dists.append(d)
    if dir_vec[2] < 0:
        d = (PHANTOM_ZMAX - pos[2]) / dir_vec[2]
        if d > 0:
            p = pos + d * dir_vec
            if -PHANTOM_HALF <= p[0] <= PHANTOM_HALF and -PHANTOM_HALF <= p[1] <= PHANTOM_HALF:
                dists.append(d)
    # x walls
    if dir_vec[0] > 0:
        d = (PHANTOM_HALF - pos[0]) / dir_vec[0]
        if d > 0:
            p = pos + d * dir_vec
            if PHANTOM_ZMIN <= p[2] <= PHANTOM_ZMAX and -PHANTOM_HALF <= p[1] <= PHANTOM_HALF:
                dists.append(d)
    if dir_vec[0] < 0:
        d = (-PHANTOM_HALF - pos[0]) / dir_vec[0]
        if d > 0:
            p = pos + d * dir_vec
            if PHANTOM_ZMIN <= p[2] <= PHANTOM_ZMAX and -PHANTOM_HALF <= p[1] <= PHANTOM_HALF:
                dists.append(d)
    # y walls
    if dir_vec[1] > 0:
        d = (PHANTOM_HALF - pos[1]) / dir_vec[1]
        if d > 0:
            p = pos + d * dir_vec
            if PHANTOM_ZMIN <= p[2] <= PHANTOM_ZMAX and -PHANTOM_HALF <= p[0] <= PHANTOM_HALF:
                dists.append(d)
    if dir_vec[1] < 0:
        d = (-PHANTOM_HALF - pos[1]) / dir_vec[1]
        if d > 0:
            p = pos + d * dir_vec
            if PHANTOM_ZMIN <= p[2] <= PHANTOM_ZMAX and -PHANTOM_HALF <= p[0] <= PHANTOM_HALF:
                dists.append(d)
    
    if not dists:
        return float('inf')
    return min(d for d in dists if d > 1e-12)

# ============================================================
# Sample beam direction
# ============================================================
def sample_beam_direction():
    """Sample a direction from the source towards the 10x10 field at SSD.
    Returns direction unit vector."""
    # Uniform in solid angle within the pyramid.
    # The pyramid is defined by the field at z=0 (surface).
    # At z=0, x in [-5,5], y in [-5,5].
    # We sample uniformly in the quadrilateral on the surface,
    # then compute direction.
    x0 = rng.uniform(-field_half, field_half)
    y0 = rng.uniform(-field_half, field_half)
    # Direction from source to (x0, y0, 0)
    v = np.array([x0, y0, 0.0]) - src
    return v / np.linalg.norm(v)

# ============================================================
# Compton scattering kinematics
# ============================================================
def compton_scatter(E_in):
    """Sample Compton scattering angle and energies.
    Returns (E_out, cos_scatter, t_e, cos_electron)
    """
    # Sample cos(theta) from Klein-Nishina distribution
    # Use rejection sampling on cos_theta in [-1, 1]
    alpha = E_in / m_e_c2
    while True:
        cosT = rng.uniform(-1, 1)
        # Klein-Nishina differential cross section dsigma/dOmega
        # dsigma/dOmega = r_e^2/2 * (E_out/E_in)^2 * (E_out/E_in + E_in/E_out - sin^2(theta))
        # where E_out = E_in / (1 + alpha*(1-cosT))
        E_out = E_in / (1 + alpha * (1 - cosT))
        ratio = E_out / E_in
        if ratio <= 0: continue
        kn_diff = ratio**2 * (ratio + 1/ratio - (1 - cosT**2))
        # Maximum of kn_diff is ~ 1 + alpha^2/(1+2*alpha) ~ at cosT=1
        # Use bounding value
        max_val = 1.0 + 1.0/alpha**2  # rough upper bound
        if rng.random() * max_val < kn_diff:
            # Electron angle
            # cos(phi_e) = sqrt((E_in - E_out + m_e_c2)/(E_in + m_e_c2))... actually:
            # From conservation: cos(theta_e) = (E_in - E_out + m_e_c2)/sqrt(E_in^2 + (E_in + m_e_c2)^2 - 2*E_in*(E_in + m_e_c2)*cos(theta))
            # Actually let me use the formula:
            # tan(phi) = -cot(theta/2) * (1 + alpha)
            # where phi is the electron angle relative to incident direction
            # cot(theta/2) = (1+cosT)/sin(theta) = (1+cosT)/sqrt(1-cosT^2)
            # Actually simpler: cos(phi) = (E_in - E_out + m_e_c2) / np.sqrt(E_in**2 + m_e_c2**2 + 2*E_in*m_e_c2)
            # Better: from literature, the electron scattering angle phi satisfies:
            # cos(phi) = (E_in - E_out) / np.sqrt(E_out**2 + E_in**2 - 2*E_in*E_out*cosT)
            # Actually:
            # E_out = E_in - t_e, where t_e = E_in * alpha * (1-cosT) / (1 + alpha*(1-cosT))
            # The electron gets momentum p_e = sqrt(t_e*(t_e+2*m_e_c2))
            # cos(phi) = (p_in - p_out*cosT) / p_e
            # where p_in = E_in/c, p_out = E_out/c
            # So: cos_e = (E_in - E_out*cosT) / sqrt(E_in^2 + E_out^2 - 2*E_in*E_out*cosT)
            # Actually easier:
            t_e = E_in - E_out
            if t_e <= 0:
                cos_e = 0.0
            else:
                p_out = E_out
                p_in = E_in
                p_e_sq = t_e**2 + 2*t_e*m_e_c2  # since E = p*c for photons, and E^2 = p^2*c^2 + m^2*c^4 for electron
                # Actually for photons p = E/c, and we set c=1
                p_e = np.sqrt(p_e_sq)
                cos_e = (p_in - p_out*cosT) / p_e
                cos_e = np.clip(cos_e, -1.0, 1.0)
            return E_out, cosT, t_e, cos_e
    # Fallback
    return None

# Test Compton sampling
for _ in range(5):
    E_out, cosT, t_e, cos_e = compton_scatter(1.1732)
    print(f"E_out={E_out:.4f}, cosT={cosT:.4f}, t_e={t_e:.4f}, cos_e={cos_e:.4f}")