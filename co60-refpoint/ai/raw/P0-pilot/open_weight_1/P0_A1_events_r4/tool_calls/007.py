import numpy as np

# ---- Physical constants ----
m_e = 0.510998950  # MeV
r_e = 2.8179403262e-13  # cm
N_A = 6.02214076e23
N_e_per_g = N_A * 10 / 18.01528

# ---- Geometry ----
Z_SOURCE = 0.0
Z_SURFACE = 100.0  # phantom surface
Z_EXIT = 130.0     # phantom exit
FIELD_SIZE = 10.0  # cm side at surface

# Scoring volume
SV_CENTER_Z = 110.0  # depth 10 cm
SV_HALF_Z = 0.1      # half-thickness 0.1 cm (total 0.2 cm)
SV_RADIUS = 1.0      # cm
SV_Z_MIN = SV_CENTER_Z - SV_HALF_Z
SV_Z_MAX = SV_CENTER_Z + SV_HALF_Z

# ---- Cross sections (cached) ----
def compute_mu(E):
    """Return (mu_total, mu_compton, mu_pe, mu_pair, mu_rayleigh) for energy E (MeV)"""
    mu_c = 0.0
    mu_p = 0.0
    mu_r = 0.0
    mu_pp = 0.0
    
    # Compton
    if E > 0:
        alpha = E / m_e
        if alpha > 0:
            term1 = (1+alpha)/alpha**2 * (2*(1+alpha)/(1+2*alpha) - np.log(1+2*alpha)/alpha)
            term2 = np.log(1+2*alpha) / (2*alpha)
            term3 = (1+3*alpha) / (1+2*alpha)**2
            sigma = 2*np.pi * r_e**2 * (term1 + term2 - term3)
            mu_c = sigma * N_e_per_g
    
    # Photoelectric
    if E > 0:
        if E < 0.01:
            mu_p = 4.89 * (E/0.01)**(-3.5)
        elif E <= 10.0:
            # Use interpolation table
            pass
        else:
            mu_p = 1e-10
    # We'll use precomputed values for efficiency
    
    return mu_c, mu_p, mu_pp, mu_r

# Actually let's precompute cross section tables for efficiency
# Create a fine grid
E_grid = np.logspace(-2, 1, 500)  # 0.01 to 10 MeV
mu_C_grid = np.array([mu_compton(E) for E in E_grid])
mu_PE_grid = np.array([mu_pe(E) for E in E_grid])
mu_PP_grid = np.array([mu_pair(E) for E in E_grid])
mu_R_grid = np.array([mu_rayleigh(E) for E in E_grid])
mu_T_grid = mu_C_grid + mu_PE_grid + mu_PP_grid + mu_R_grid

# Interpolation functions
from scipy.interpolate import interp1d
f_muC = interp1d(E_grid, mu_C_grid, kind='linear', bounds_error=False, fill_value=0.0)
f_muPE = interp1d(E_grid, mu_PE_grid, kind='linear', bounds_error=False, fill_value=0.0)
f_muPP = interp1d(E_grid, mu_PP_grid, kind='linear', bounds_error=False, fill_value=0.0)
f_muR = interp1d(E_grid, mu_R_grid, kind='linear', bounds_error=False, fill_value=0.0)
f_muT = interp1d(E_grid, mu_T_grid, kind='linear', bounds_error=False, fill_value=0.0)

def get_mu(E):
    """Return (mu_total, mu_compton, mu_pe, mu_pair, mu_rayleigh)"""
    if E <= 0:
        return (0.0, 0.0, 0.0, 0.0, 0.0)
    mu_t = float(f_muT(E))
    mu_c = float(f_muC(E))
    mu_pe = float(f_muPE(E))
    mu_pp = float(f_muPP(E))
    mu_r = float(f_muR(E))
    return mu_t, mu_c, mu_pe, mu_pp, mu_r

# ---- Sampling functions ----
def sample_compton(energy, rng):
    E = energy
    alpha = E / m_e
    while True:
        xi1 = rng.uniform()
        xi2 = rng.uniform()
        mu = 2.0 * xi1 - 1.0
        ratio = 1.0 / (1.0 + alpha * (1.0 - mu))
        f_kn = ratio**2 * (ratio + 1.0/ratio - (1.0 - mu**2))
        if xi2 <= f_kn / 2.0:
            e_out = E * ratio
            cos_scatter = mu
            break
    t_e = E - e_out
    denom = np.sqrt(E**2 + e_out**2 - 2.0 * E * e_out * cos_scatter)
    cos_electron = (E - e_out * cos_scatter) / denom if denom > 0 else 1.0
    return e_out, cos_scatter, t_e, cos_electron

def sample_rayleigh(energy, rng):
    """Simplified Rayleigh: sample cosθ from forward-peaked distribution."""
    # Use a simple model: cosθ = 1 - 2 * (rng.exponential(scale=theta0^2/2))
    # where theta0 is characteristic angle
    # theta0 ~ 0.02 * Z^(-1/3) / E (in radians)
    Z_eff = 10.0
    theta0 = 0.02 / (max(energy, 0.01) * Z_eff**(1/3))  # radians
    # Sample theta from Rayleigh distribution with scale = theta0
    # Using transformation: theta = sqrt(-2*ln(1-u))*theta0? Actually Rayleigh(σ) has PDF r/σ^2 exp(-r^2/(2σ^2))
    # We can sample theta = sigma * sqrt(-2*ln(1-u))
    u = rng.uniform()
    theta = theta0 * np.sqrt(-2.0 * np.log(1.0 - u))
    cos_scatter = np.cos(theta)
    return energy, cos_scatter  # e_out = energy (no loss)

def sample_photoelectric(energy, rng):
    t_e = energy  # approximately
    cos_electron = rng.uniform(-1.0, 1.0)  # isotropic
    return t_e, cos_electron

def sample_pair(energy, rng):
    T_total = energy - 2*m_e
    if T_total <= 0:
        return 0.0, 1.0, 0.0
    t_e = rng.uniform(0.0, T_total)
    t_pos = T_total - t_e
    # Electron angle: approximate forward peaking
    theta_avg = m_e / energy
    if theta_avg > 0:
        theta = theta_avg * np.sqrt(-2.0 * np.log(1.0 - rng.uniform()))
    else:
        theta = 0.0
    cos_electron = np.cos(theta)
    return t_e, cos_electron, t_pos

# ---- Vector utilities ----
def normalize(v):
    n = np.linalg.norm(v)
    if n == 0:
        return v
    return v / n

def dot(a, b):
    return a[0]*b[0] + a[1]*b[1] + a[2]*b[2]

def cross(a, b):
    return np.array([a[1]*b[2] - a[2]*b[1],
                     a[2]*b[0] - a[0]*b[2],
                     a[0]*b[1] - a[1]*b[0]])

def rotate_direction(d, cos_theta, azimuth):
    """Rotate direction vector d by angle theta (cos_theta) and azimuth.
    Returns new direction vector."""
    # Build orthonormal basis: d is z-axis, x' and y' are perpendicular
    z = d
    # Find a vector not parallel to z
    if abs(z[2]) < 0.9:
        x = normalize(cross(np.array([0.0, 0.0, 1.0]), z))
    else:
        x = normalize(cross(np.array([1.0, 0.0, 0.0]), z))
    y = cross(z, x)
    # New direction in local coordinates
    sin_theta = np.sqrt(1.0 - cos_theta**2)
    new_dir = (sin_theta * np.cos(azimuth) * x + 
               sin_theta * np.sin(azimuth) * y +
               cos_theta * z)
    return new_dir

# ---- Main simulation ----
def simulate_photon(rng, record_events=True):
    """
    Simulate one photon history.
    Returns list of interaction events in scoring volume (each dict).
    """
    events = []
    
    # Sample initial direction
    # Uniform in square at z=100
    u = rng.uniform(-FIELD_SIZE/2, FIELD_SIZE/2)
    v = rng.uniform(-FIELD_SIZE/2, FIELD_SIZE/2)
    dir_vec = np.array([u, v, Z_SURFACE - Z_SOURCE])
    dir_vec = normalize(dir_vec)
    
    pos = np.array([0.0, 0.0, Z_SOURCE])
    energy = 2.0  # MeV
    dir = dir_vec
    
    # Distance to phantom surface
    # Path: pos + t * dir, find t such that z = Z_SURFACE
    if dir[2] > 0:
        t_surf = (Z_SURFACE - pos[2]) / dir[2]
        pos = pos + t_surf * dir
    else:
        # Photon misses phantom (shouldn't happen for field)
        return events
    
    # Track inside phantom
    max_steps = 1000
    for _ in range(max_steps):
        # Get cross sections at current energy
        mu_t, mu_c, mu_pe, mu_pp, mu_r = get_mu(energy)
        if mu_t <= 0:
            break  # no interaction possible
        
        # Mean free path (cm) = 1/(mu_t * rho), rho=1
        mfp = 1.0 / mu_t
        
        # Sample distance to next interaction
        s = -mfp * np.log(1.0 - rng.uniform())
        
        # New position after step
        new_pos = pos + s * dir
        
        # Check if beyond phantom exit
        if new_pos[2] >= Z_EXIT:
            break
        
        # Check if inside phantom (should be, but check)
        if new_pos[2] < Z_SURFACE:
            break
        
        # Now we have an interaction at new_pos
        # Check if in scoring volume
        z = new_pos[2]
        r = np.sqrt(new_pos[0]**2 + new_pos[1]**2)
        
        in_scoring = (z >= SV_Z_MIN and z <= SV_Z_MAX and r <= SV_RADIUS)
        
        # Sample interaction type
        xi = rng.uniform()
        # Probabilities
        p_c = mu_c / mu_t
        p_pe = mu_pe / mu_t
        p_pp = mu_pp / mu_t
        p_r = mu_r / mu_t
        
        process = None
        e_in = energy
        e_out = None
        cos_scatter = None
        t_e = 0.0
        cos_electron = None
        t_pos = None
        
        if xi < p_c:
            # Compton
            process = 'compton'
            e_out, cos_scatter, t_e, cos_electron = sample_compton(energy, rng)
            # Update energy and direction for scattered photon
            azimuth = rng.uniform(0, 2*np.pi)
            new_dir = rotate_direction(dir, cos_scatter, azimuth)
            energy = e_out
            dir = new_dir
        elif xi < p_c + p_pe:
            # Photoelectric
            process = 'photoelectric'
            t_e, cos_electron = sample_photoelectric(energy, rng)
            # Photon absorbed, stop tracking
            energy = 0.0
        elif xi < p_c + p_pe + p_pp:
            # Pair production
            process = 'pair'
            t_e, cos_electron, t_pos = sample_pair(energy, rng)
            # Photon absorbed
            energy = 0.0
        else:
            # Rayleigh
            process = 'rayleigh'
            e_out, cos_scatter = sample_rayleigh(energy, rng)
            # No energy loss
            azimuth = rng.uniform(0, 2*np.pi)
            new_dir = rotate_direction(dir, cos_scatter, azimuth)
            dir = new_dir
            # Rayleigh does not set electron in motion
            cos_electron = None
            t_e = 0.0
        
        # Record if in scoring volume
        if in_scoring and record_events:
            event = {
                'process': process,
                'e_in_MeV': float(e_in),
                'e_out_MeV': float(e_out) if e_out is not None else None,
                'cos_scatter': float(cos_scatter) if cos_scatter is not None else None,
                't_e_MeV': float(t_e),
                'cos_electron': float(cos_electron) if cos_electron is not None else None,
                't_positron_MeV': float(t_pos) if t_pos is not None else None
            }
            events.append(event)
        
        # Update position
        pos = new_pos
        
        # If photon absorbed (or energy too low), stop
        if process in ('photoelectric', 'pair') or energy < 0.001:
            break
    
    return events

# Test with a few histories
rng = np.random.RandomState(42)
total_events = []
n_histories = 50000
for i in range(n_histories):
    evts = simulate_photon(rng)
    total_events.extend(evts)
    if len(total_events) >= 200:
        break

print(f"Generated {len(total_events)} events in {i+1} histories")
if len(total_events) > 0:
    # Count processes
    from collections import Counter
    cnt = Counter(e['process'] for e in total_events)
    print("Process counts:", dict(cnt))