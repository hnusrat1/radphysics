import numpy as np
from scipy import interpolate
import time

# Constants
m_e = 0.510998950
r_e = 2.8179403262e-13
N_A = 6.02214076e23
N_e_per_g = N_A * 10 / 18.01528

# Precompute cross section tables
E_grid = np.logspace(-2, 1, 500)  # 0.01 to 10 MeV

def klein_nishina_sigma(E):
    alpha = E / m_e
    term1 = (1+alpha)/alpha**2 * (2*(1+alpha)/(1+2*alpha) - np.log(1+2*alpha)/alpha)
    term2 = np.log(1+2*alpha) / (2*alpha)
    term3 = (1+3*alpha) / (1+2*alpha)**2
    return 2*np.pi * r_e**2 * (term1 + term2 - term3)

mu_C_grid = np.array([klein_nishina_sigma(E) * N_e_per_g for E in E_grid])

# Photoelectric
pe_energies = np.array([0.01, 0.015, 0.02, 0.03, 0.04, 0.05, 0.06, 0.08, 0.1, 0.15, 0.2, 0.3, 0.4, 0.5, 0.6, 0.8, 1.0, 1.5, 2.0, 3.0, 5.0, 10.0])
pe_murho = np.array([4.89, 1.23, 0.359, 0.0559, 0.0150, 0.0050, 0.0019, 0.00038, 0.000163, 0.000044, 0.000020, 0.000008, 0.000004, 0.0000025, 0.0000018, 0.0000010, 0.0000007, 0.0000003, 0.0000002, 0.0000001, 0.00000003, 0.00000001])
f_pe_log = interpolate.interp1d(np.log(pe_energies[1:]), np.log(np.maximum(pe_murho[1:], 1e-20)), kind='linear', bounds_error=False, fill_value='extrapolate')
mu_PE_grid = np.array([max(0.0, float(np.exp(f_pe_log(np.log(E))))) if 0.01 <= E <= 10.0 else (4.89*(E/0.01)**(-3.5) if E < 0.01 else 1e-10) for E in E_grid])

# Pair production
pair_energies = np.array([1.022, 1.5, 2.0, 3.0, 5.0, 10.0, 20.0])
pair_murho = np.array([0.0, 0.00004, 0.00013, 0.00046, 0.0015, 0.0048, 0.0077])
f_pair = interpolate.interp1d(pair_energies, pair_murho, kind='linear', bounds_error=False, fill_value=0.0)
mu_PP_grid = np.array([(0.0 if E <= 2*m_e else (float(f_pair(E)) if E <= 20.0 else 0.0077+0.00002*(E-20))) for E in E_grid])

# Rayleigh
mu_R_grid = np.array([(0.16*(E/0.01)**(-3.5) if E < 0.1 else (0.0002*(E/0.1)**(-4) if E < 0.5 else 0.0)) for E in E_grid])

mu_T_grid = mu_C_grid + mu_PE_grid + mu_PP_grid + mu_R_grid

f_muT = interpolate.interp1d(E_grid, mu_T_grid, kind='linear', bounds_error=False, fill_value=0.0)
f_muC = interpolate.interp1d(E_grid, mu_C_grid, kind='linear', bounds_error=False, fill_value=0.0)
f_muPE = interpolate.interp1d(E_grid, mu_PE_grid, kind='linear', bounds_error=False, fill_value=0.0)
f_muPP = interpolate.interp1d(E_grid, mu_PP_grid, kind='linear', bounds_error=False, fill_value=0.0)
f_muR = interpolate.interp1d(E_grid, mu_R_grid, kind='linear', bounds_error=False, fill_value=0.0)

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
def sample_compton(E, rng):
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

def sample_rayleigh(E, rng):
    Z_eff = 10.0
    theta0 = 0.02 / (max(E, 0.01) * Z_eff**(1/3))
    u = rng.uniform()
    theta = theta0 * np.sqrt(-2.0 * np.log(1.0 - u))
    return E, np.cos(theta)

def sample_photoelectric(E, rng):
    return E, rng.uniform(-1.0, 1.0)

def sample_pair(E, rng):
    T_total = E - 2*m_e
    if T_total <= 0:
        return 0.0, 1.0, 0.0
    t_e = rng.uniform(0.0, T_total)
    t_pos = T_total - t_e
    theta_avg = m_e / E
    if theta_avg > 0:
        theta = theta_avg * np.sqrt(-2.0 * np.log(1.0 - rng.uniform()))
    else:
        theta = 0.0
    return t_e, np.cos(theta), t_pos

# ---- Vector utilities ----
def normalize(v):
    n = np.sqrt(v[0]**2 + v[1]**2 + v[2]**2)
    if n == 0:
        return v
    return v / n

def rotate_direction(d, cos_theta, azimuth):
    z = d
    if abs(z[2]) < 0.9:
        x = np.cross(np.array([0.0, 0.0, 1.0]), z)
        x = x / np.sqrt(x[0]**2 + x[1]**2 + x[2]**2)
    else:
        x = np.cross(np.array([1.0, 0.0, 0.0]), z)
        x = x / np.sqrt(x[0]**2 + x[1]**2 + x[2]**2)
    y = np.cross(z, x)
    sin_theta = np.sqrt(max(0.0, 1.0 - cos_theta**2))
    new_dir = np.array([
        sin_theta * np.cos(azimuth) * x[0] + sin_theta * np.sin(azimuth) * y[0] + cos_theta * z[0],
        sin_theta * np.cos(azimuth) * x[1] + sin_theta * np.sin(azimuth) * y[1] + cos_theta * z[1],
        sin_theta * np.cos(azimuth) * x[2] + sin_theta * np.sin(azimuth) * y[2] + cos_theta * z[2]
    ])
    return new_dir

# ---- Perform interaction (returns dict with outcomes) ----
def perform_interaction(energy, dir, record_events, rng):
    """Perform an interaction at the current position.
    Returns (absorbed, new_energy, new_dir, event_dict)
    """
    mu_t, mu_c, mu_pe, mu_pp, mu_r = get_mu(energy)
    if mu_t <= 0:
        return True, 0.0, dir, None
    
    xi = rng.uniform()
    p_c = mu_c / mu_t
    p_pe = mu_pe / mu_t
    p_pp = mu_pp / mu_t
    
    process = None
    e_in = energy
    e_out = None
    cos_scatter = None
    t_e = 0.0
    cos_electron = None
    t_pos = None
    new_energy = energy
    new_dir = dir
    
    if xi < p_c:
        process = 'compton'
        e_out, cos_scatter, t_e, cos_electron = sample_compton(energy, rng)
        azimuth = rng.uniform(0, 2*np.pi)
        new_dir = rotate_direction(dir, cos_scatter, azimuth)
        new_energy = e_out
    elif xi < p_c + p_pe:
        process = 'photoelectric'
        t_e, cos_electron = sample_photoelectric(energy, rng)
        new_energy = 0.0
        cos_electron = np.float64(cos_electron)
    elif xi < p_c + p_pe + p_pp:
        process = 'pair'
        t_e, cos_electron, t_pos = sample_pair(energy, rng)
        new_energy = 0.0
    else:
        process = 'rayleigh'
        e_out, cos_scatter = sample_rayleigh(energy, rng)
        azimuth = rng.uniform(0, 2*np.pi)
        new_dir = rotate_direction(dir, cos_scatter, azimuth)
        cos_electron = None
        t_e = 0.0
    
    absorbed = (process in ('photoelectric', 'pair')) or new_energy < 0.001
    
    event = None
    if record_events:
        event = {
            'process': process,
            'e_in_MeV': float(e_in),
            'e_out_MeV': float(e_out) if e_out is not None else None,
            'cos_scatter': float(cos_scatter) if cos_scatter is not None else None,
            't_e_MeV': float(t_e),
            'cos_electron': float(cos_electron) if cos_electron is not None else None,
            't_positron_MeV': float(t_pos) if t_pos is not None else None
        }
    
    return absorbed, new_energy, new_dir, event

# ---- Geometry ----
Z_SURFACE = 100.0
Z_EXIT = 130.0
FIELD_HALF = 5.0

SV_Z_MIN = 109.9
SV_Z_MAX = 110.1
SV_RADIUS_SQ = 1.0  # radius 1 cm

# ---- Main simulation ----
def simulate_one(rng):
    events = []
    
    # Sample initial direction
    u = rng.uniform(-FIELD_HALF, FIELD_HALF)
    v = rng.uniform(-FIELD_HALF, FIELD_HALF)
    dir_norm = np.sqrt(u*u + v*v + 10000.0)
    dir = np.array([u/dir_norm, v/dir_norm, 100.0/dir_norm])
    
    pos = np.array([u, v, Z_SURFACE])  # position at surface
    energy = 2.0
    
    for _ in range(200):
        mu_t, _, _, _, _ = get_mu(energy)
        if mu_t <= 0:
            break
        
        mfp = 1.0 / mu_t
        s = -mfp * np.log(1.0 - rng.uniform())
        
        new_x = pos[0] + s * dir[0]
        new_y = pos[1] + s * dir[1]
        new_z = pos[2] + s * dir[2]
        
        if new_z >= Z_EXIT or new_z < Z_SURFACE:
            break
        
        # Check if in scoring volume
        in_scoring = (new_z >= SV_Z_MIN and new_z <= SV_Z_MAX and 
                      new_x*new_x + new_y*new_y <= SV_RADIUS_SQ)
        
        # Perform interaction
        absorbed, energy, dir, event = perform_interaction(energy, dir, in_scoring, rng)
        
        if in_scoring and event is not None:
            events.append(event)
        
        pos = np.array([new_x, new_y, new_z])
        
        if absorbed:
            break
    
    return events

# Run benchmark
rng = np.random.RandomState(12345)
t0 = time.time()
total_events = []
for i in range(50000):
    evts = simulate_one(rng)
    total_events.extend(evts)
    if len(total_events) >= 200:
        break
t1 = time.time()
print(f"Generated {len(total_events)} events in {i+1} histories, time={t1-t0:.2f}s")
if len(total_events) > 0:
    from collections import Counter
    cnt = Counter(e['process'] for e in total_events)
    print("Process counts:", dict(cnt))
    for e in total_events[:3]:
        print(e)