import numpy as np

# ============================================================
# Constants
# ============================================================
m_e = 0.510998950  # MeV
r_e = 2.8179403262e-13  # cm
NA = 6.02214076e23
e_per_cm3 = 10 * NA / 18.01528  # electrons/cm^3

# ============================================================
# Cross sections
# ============================================================
def mu_compton(E):
    alpha = E / m_e
    t1 = (1+alpha)/alpha**2 * (2*(1+alpha)/(1+2*alpha) - np.log(1+2*alpha)/alpha)
    t2 = np.log(1+2*alpha)/(2*alpha)
    t3 = (1+3*alpha)/(1+2*alpha)**2
    sigma = 2*np.pi*r_e**2*(t1 + t2 - t3)
    return sigma * e_per_cm3

def mu_photo(E):
    data_E = np.array([0.01, 0.015, 0.02, 0.03, 0.04, 0.05, 0.06, 0.08, 0.1, 0.15, 0.2, 0.3, 0.4, 0.5, 0.6, 0.8, 1.0, 1.1732, 1.25, 1.3325, 1.5, 2.0])
    data_mu = np.array([4.91, 1.67, 0.755, 0.216, 0.0881, 0.0442, 0.0255, 0.0113, 0.00642, 0.00254, 0.00152, 0.00154, 0.00228, 0.00406, 0.00555, 0.00450, 0.00377, 0.00300, 0.00258, 0.00230, 0.00179, 0.00100])
    return np.interp(E, data_E, data_mu) * 1.0

def mu_pair(E):
    if E < 2*m_e:
        return 0.0
    data_E = np.array([1.022, 1.1, 1.1732, 1.25, 1.3325, 1.5, 2.0, 3.0, 5.0])
    data_mu = np.array([0.0, 0.0002, 0.00035, 0.00071, 0.00100, 0.00146, 0.00383, 0.0085, 0.0175])
    return np.interp(E, data_E, data_mu) * 1.0

def mu_rayleigh(E):
    data_E = np.array([0.01, 0.015, 0.02, 0.03, 0.04, 0.05, 0.06, 0.08, 0.1, 0.15, 0.2, 0.3, 0.4, 0.5, 0.6, 0.8, 1.0, 1.1732, 1.25, 1.3325, 1.5, 2.0])
    data_mu = np.array([0.037, 0.034, 0.032, 0.028, 0.025, 0.022, 0.019, 0.015, 0.012, 0.0085, 0.0065, 0.0050, 0.0040, 0.0055, 0.0045, 0.0035, 0.00303, 0.00250, 0.00242, 0.00210, 0.00186, 0.00114])
    return np.interp(E, data_E, data_mu) * 1.0

def mu_total(E):
    return mu_compton(E) + mu_photo(E) + mu_pair(E) + mu_rayleigh(E)

# ============================================================
# Physics sampling
# ============================================================
def sample_interaction(E):
    """Sample interaction type. Returns string."""
    mc = mu_compton(E)
    mp = mu_photo(E)
    mpp = mu_pair(E)
    mr = mu_rayleigh(E)
    total = mc + mp + mpp + mr
    
    r = np.random.random() * total
    if r < mc:
        return 'compton'
    r -= mc
    if r < mp:
        return 'photoelectric'
    r -= mp
    if r < mpp:
        return 'pair'
    return 'rayleigh'

def sample_compton(E_in):
    """Returns (cos_theta, E_out, t_e)"""
    alpha = E_in / m_e
    while True:
        cos_theta = 2*np.random.random() - 1
        E_out = E_in / (1 + alpha * (1 - cos_theta))
        kn = (E_out/E_in)**2 * (E_out/E_in + E_in/E_out - (1 - cos_theta**2))
        if np.random.random() <= kn:
            return cos_theta, E_out, E_in - E_out

def sample_rayleigh(E_in):
    """Rayleigh scattering. Returns (cos_theta, E_out).
    Rayleigh is elastic (E_out = E_in).
    Angular distribution is forward-peaked, approximately 
    dσ/dΩ ∝ (1+cos²θ) * F²(x,Z) where F is the form factor.
    For simplicity, approximate as (1+cos²θ) * exp(-x²/2) type.
    At 1.17 MeV, Rayleigh is very forward-peaked.
    """
    # Simplified: sample from (1+cos²θ) * exp(-a*(1-cos_theta))
    # where a accounts for the form factor suppression
    # At 1.17 MeV, a ≈ 30 (very forward)
    # Let me use a simple rejection method
    
    # The Rayleigh DCS is proportional to (1+cos²θ) * F²(x,Z)
    # F is the atomic form factor, which falls off rapidly
    # For water at 1.17 MeV, the form factor restricts scattering 
    # to very small angles (cos_theta ≈ 1)
    
    # Simple model: sample from (1+cos²) but restrict to forward angles
    # Using a form factor approximation
    while True:
        cos_theta = 1.0 - 0.01 * np.random.exponential(1.0)
        if cos_theta < -1 or cos_theta > 1:
            continue
        # Weight by (1+cos²) * form_factor_squared
        ff = np.exp(-100.0 * (1 - cos_theta))
        w = (1 + cos_theta**2) * ff
        if np.random.random() <= w:
            return cos_theta, E_in

# ============================================================
# Geometry
# ============================================================
src_pos = np.array([0.0, 0.0, -100.0])

def sample_source_photon():
    """Sample a source photon. Returns (energy, position, direction)."""
    # Energy
    E = 1.1732 if np.random.random() < 0.5 else 1.3325
    
    # Direction: sample uniformly on the 10x10 field at z=0
    x = (np.random.random() - 0.5) * 10.0  # [-5, 5]
    y = (np.random.random() - 0.5) * 10.0  # [-5, 5]
    
    # Direction from source (0,0,-100) to (x,y,0)
    d = np.array([x, y, 100.0])
    d = d / np.linalg.norm(d)
    
    # Starting position: at the phantom surface (z=0)
    # The ray from src goes to (x,y,0), so entry point is (x,y,0)
    pos = np.array([x, y, 0.0])
    
    return E, pos, d

def distance_to_boundary(pos, dir):
    """Distance along dir to phantom boundary (or beyond)."""
    d_max = np.inf
    
    # z boundaries
    for z_bound in [0.0, 30.0]:
        if abs(dir[2]) > 1e-10:
            d = (z_bound - pos[2]) / dir[2]
            if d > 1e-10 and d < d_max:
                d_max = d
    
    # x boundaries
    for x_bound in [-15.0, 15.0]:
        if abs(dir[0]) > 1e-10:
            d = (x_bound - pos[0]) / dir[0]
            if d > 1e-10 and d < d_max:
                d_max = d
    
    # y boundaries
    for y_bound in [-15.0, 15.0]:
        if abs(dir[1]) > 1e-10:
            d = (y_bound - pos[1]) / dir[1]
            if d > 1e-10 and d < d_max:
                d_max = d
    
    return d_max

def in_scoring_volume(pos):
    x, y, z = pos
    return (4.9 <= z <= 5.1) and (x**2 + y**2 <= 1.0**2)

def in_phantom(pos):
    x, y, z = pos
    return (0 <= z <= 30) and (-15 <= x <= 15) and (-15 <= y <= 15)

# ============================================================
# Record keeping
# ============================================================
events = []

def record_event(process, e_in, e_out, cos_scatter, t_e, cos_electron, t_positron):
    events.append({
        'process': process,
        'e_in_MeV': float(e_in),
        'e_out_MeV': float(e_out) if e_out is not None else None,
        'cos_scatter': float(cos_scatter) if cos_scatter is not None else None,
        't_e_MeV': float(t_e),
        'cos_electron': float(cos_electron) if cos_electron is not None else None,
        't_positron_MeV': float(t_positron) if t_positron is not None else None
    })

# ============================================================
# Photon transport
# ============================================================
# Let me check the syntax errors first