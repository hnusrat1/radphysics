import numpy as np
from scipy import interpolate
from collections import Counter

# ---- Physical constants ----
m_e = 0.510998950  # MeV
r_e = 2.8179403262e-13  # cm
N_A = 6.02214076e23
N_e_per_g = N_A * 10 / 18.01528

# ---- Cross section functions ----
def klein_nishina_sigma(E_MeV):
    alpha = E_MeV / m_e
    if alpha <= 0:
        return 0.0
    term1 = (1+alpha)/alpha**2 * (2*(1+alpha)/(1+2*alpha) - np.log(1+2*alpha)/alpha)
    term2 = np.log(1+2*alpha) / (2*alpha)
    term3 = (1+3*alpha) / (1+2*alpha)**2
    sigma = 2*np.pi * r_e**2 * (term1 + term2 - term3)
    return sigma

def mu_compton(E):
    return klein_nishina_sigma(E) * N_e_per_g

# Photoelectric interpolation
pe_energies = np.array([0.01, 0.015, 0.02, 0.03, 0.04, 0.05, 0.06, 0.08, 0.1, 0.15, 0.2, 0.3, 0.4, 0.5, 0.6, 0.8, 1.0, 1.5, 2.0, 3.0, 5.0, 10.0])
pe_murho = np.array([4.89, 1.23, 0.359, 0.0559, 0.0150, 0.0050, 0.0019, 0.00038, 0.000163, 0.000044, 0.000020, 0.000008, 0.000004, 0.0000025, 0.0000018, 0.0000010, 0.0000007, 0.0000003, 0.0000002, 0.0000001, 0.00000003, 0.00000001])
f_pe_log = interpolate.interp1d(np.log(pe_energies[1:]), np.log(np.maximum(pe_murho[1:], 1e-20)), kind='linear', bounds_error=False, fill_value='extrapolate')
def mu_pe(E):
    if E <= 0:
        return 0.0
    if E < 0.01:
        return 4.89 * (E/0.01)**(-3.5)
    if E > 10.0:
        return 1e-10
    return max(0.0, float(np.exp(f_pe_log(np.log(E)))))

pair_energies = np.array([1.022, 1.5, 2.0, 3.0, 5.0, 10.0, 20.0])
pair_murho = np.array([0.0, 0.00004, 0.00013, 0.00046, 0.0015, 0.0048, 0.0077])
f_pair = interpolate.interp1d(pair_energies, pair_murho, kind='linear', bounds_error=False, fill_value=0.0)
def mu_pair(E):
    if E <= 2*m_e:
        return 0.0
    if E > 20.0:
        return 0.0077 + 0.00002 * (E-20)
    return float(f_pair(E))

def mu_rayleigh(E):
    if E < 0.01:
        return 0.16
    elif E < 0.1:
        return 0.16 * (E/0.01)**(-3.5)
    elif E < 0.5:
        return 0.0002 * (E/0.1)**(-4)
    else:
        return 0.0

def mu_total(E):
    return mu_compton(E) + mu_pe(E) + mu_pair(E) + mu_rayleigh(E)

# ---- Precompute cross section tables ----
E_grid = np.logspace(-2, 1, 500)  # 0.01 to 10 MeV
mu_C_grid = np.array([mu_compton(E) for E in E_grid])
mu_PE_grid = np.array([mu_pe(E) for E in E_grid])
mu_PP_grid = np.array([mu_pair(E) for E in E_grid])
mu_R_grid = np.array([mu_rayleigh(E) for E in E_grid])
mu_T_grid = mu_C_grid + mu_PE_grid + mu_PP_grid + mu_R_grid

f_muC = interpolate.interp1d(E_grid, mu_C_grid, kind='linear', bounds_error=False, fill_value=0.0)
f_muPE = interpolate.interp1d(E_grid, mu_PE_grid, kind='linear', bounds_error=False, fill_value=0.0)
f_muPP = interpolate.interp1d(E_grid, mu_PP_grid, kind='linear', bounds_error=False, fill_value=0.0)
f_muR = interpolate.interp1d(E_grid, mu_R_grid, kind='linear', bounds_error=False, fill_value=0.0)
f_muT = interpolate.interp1d(E_grid, mu_T_grid, kind='linear', bounds_error=False, fill_value=0.0)

def get_mu(E):
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
    Z_eff = 10.0
    theta0 = 0.02 / (max(energy, 0.01) * Z_eff**(1/3))
    u = rng.uniform()
    theta = theta0 * np.sqrt(-2.0 * np.log(1.0 - u))
    cos_scatter = np.cos(theta)
    return energy, cos_scatter

def sample_photoelectric(energy, rng):
    t_e = energy
    cos_electron = rng.uniform(-1.0, 1.0)
    return t_e, cos_electron

def sample_pair(energy, rng):
    T_total = energy - 2*m_e
    if T_total <= 0:
        return 0.0, 1.0, 0.0
    t_e = rng.uniform(0.0, T_total)
    t_pos = T_total - t_e
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

def rotate_direction(d, cos_theta, azimuth):
    """Rotate direction d by polar angle theta (given cos_theta) and azimuth."""
    z = d
    if abs(z[2]) < 0.9:
        x = normalize(np.cross(np.array([0.0, 0.0, 1.0]), z))
    else:
        x = normalize(np.cross(np.array([1.0, 0.0, 0.0]), z))
    y = np.cross(z, x)
    sin_theta = np.sqrt(max(0.0, 1.0 - cos_theta**2))
    new_dir = (sin_theta * np.cos(azimuth) * x + 
               sin_theta * np.sin(azimuth) * y +
               cos_theta * z)
    return new_dir

# ---- Geometry ----
Z_SOURCE = 0.0
Z_SURFACE = 100.0
Z_EXIT = 130.0
FIELD_SIZE = 10.0

SV_CENTER_Z = 110.0
SV_HALF_Z = 0.1
SV_RADIUS = 1.0
SV_Z_MIN = SV_CENTER_Z - SV_HALF_Z
SV_Z_MAX = SV_CENTER_Z + SV_HALF_Z

# ---- Main simulation ----
def simulate_photon(rng, record_events=True):
    events = []
    
    # Sample initial direction
    u = rng.uniform(-FIELD_SIZE/2, FIELD_SIZE/2)
    v = rng.uniform(-FIELD_SIZE/2, FIELD_SIZE/2)
    dir_vec = np.array([u, v, Z_SURFACE - Z_SOURCE])
    dir_vec = normalize(dir_vec)
    
    pos = np.array([0.0, 0.0, Z_SOURCE])
    energy = 2.0
    dir = dir_vec
    
    # Distance to phantom surface
    if dir[2] > 0:
        t_surf = (Z_SURFACE - pos[2]) / dir[2]
        pos = pos + t_surf * dir
    else:
        return events
    
    # Track inside phantom
    for _ in range(1000):
        mu_t, mu_c, mu_pe, mu_pp, mu_r = get_mu(energy)
        if mu_t <= 0:
            break
        
        mfp = 1.0 / mu_t
        s = -mfp * np.log(1.0 - rng.uniform())
        new_pos = pos + s * dir
        
        if new_pos[2] >= Z_EXIT:
            break
        if new_pos[2] < Z_SURFACE:
            break
        
        # Interaction at new_pos
        z = new_pos[2]
        r = np.sqrt(new_pos[0]**2 + new_pos[1]**2)
        in_scoring = (z >= SV_Z_MIN and z <= SV_Z_MAX and r <= SV_RADIUS)
        
        xi = rng.uniform()
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
            process = 'compton'
            e_out, cos_scatter, t_e, cos_electron = sample_compton(energy, rng)
            azimuth = rng.uniform(0, 2*np.pi)
            dir = rotate_direction(dir, cos_scatter, azimuth)
            energy = e_out
        elif xi < p_c + p_pe:
            process = 'photoelectric'
            t_e, cos_electron = sample_photoelectric(energy, rng)
            energy = 0.0
        elif xi < p_c + p_pe + p_pp:
            process = 'pair'
            t_e, cos_electron, t_pos = sample_pair(energy, rng)
            energy = 0.0
        else:
            process = 'rayleigh'
            e_out, cos_scatter = sample_rayleigh(energy, rng)
            azimuth = rng.uniform(0, 2*np.pi)
            dir = rotate_direction(dir, cos_scatter, azimuth)
            cos_electron = None
            t_e = 0.0
        
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
        
        pos = new_pos
        
        if process in ('photoelectric', 'pair') or energy < 0.001:
            break
    
    return events

# Run a quick test
rng = np.random.RandomState(12345)
total_events = []
n_histories = 50000
for i in range(n_histories):
    evts = simulate_photon(rng)
    total_events.extend(evts)
    if len(total_events) >= 200:
        break

print(f"Generated {len(total_events)} events in {i+1} histories")
if len(total_events) > 0:
    cnt = Counter(e['process'] for e in total_events)
    print("Process counts:", dict(cnt))
    # Show a few events
    for e in total_events[:5]:
        print(e)