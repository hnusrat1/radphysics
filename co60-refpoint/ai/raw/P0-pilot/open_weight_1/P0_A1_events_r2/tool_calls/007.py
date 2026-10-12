import numpy as np
import math
import random
import time

# Re-initialize everything (to avoid any caching issues)
# Constants
m_e = 0.51099895
r_e_sq = 7.9408e-26
n_e = 3.3428e23

PHANTOM_HALF = 15.0
PHANTOM_Z_MIN = 0.0
PHANTOM_Z_MAX = 30.0
SSD = 100.0
SOURCE = np.array([0.0, 0.0, -SSD])
FIELD_HALF = 5.0

SCORE_Z_CENTER = 10.0
SCORE_Z_HALF = 0.1
SCORE_RADIUS = 1.0
SCORE_RADIUS_SQ = SCORE_RADIUS**2

# Cross sections
def mu_Compton(E):
    alpha = E / m_e
    t1 = (1+alpha)/alpha**2 * (2*(1+alpha)/(1+2*alpha) - math.log(1+2*alpha)/alpha)
    t2 = math.log(1+2*alpha) / (2*alpha)
    t3 = (1+3*alpha) / (1+2*alpha)**2
    sigma = 2*math.pi * r_e_sq * (t1 + t2 - t3)
    return n_e * sigma

def mu_Rayleigh(E):
    if E <= 0:
        return 0.0
    return 0.0008 * (1.0 / max(E, 0.01))**2

def mu_Photoelectric(E):
    if E <= 0:
        return 0.0
    return 1.9e-5 * (1.0 / max(E, 0.01))**3.5

def mu_Pair(E):
    if E <= 1.022:
        return 0.0
    if E < 2.0:
        return 0.0004 * (E - 1.022) / (2.0 - 1.022)
    elif E < 3.0:
        return 0.0004 + (0.0007 - 0.0004) * (E - 2.0)
    elif E < 5.0:
        return 0.0007 + (0.0020 - 0.0007) * (E - 3.0) / 2.0
    else:
        return 0.0020 * (E / 5.0)**0.5

def mu_Total(E):
    return mu_Compton(E) + mu_Rayleigh(E) + mu_Photoelectric(E) + mu_Pair(E)

# Compton sampling tables
compton_cache = {}

def _build_compton_cdf(alpha):
    n_pts = 2000
    x = np.linspace(0, 2, n_pts)
    term1 = 1.0 / (1 + alpha * x)
    term2 = 1.0 / (1 + alpha * x)**3
    term3 = (2 * x - x**2) / (1 + alpha * x)**2
    pdf = term1 + term2 - term3
    pdf = np.maximum(pdf, 0.0)
    cdf = np.cumsum(pdf)
    cdf = cdf / cdf[-1]
    return x, cdf

def get_compton_sampler(alpha):
    if alpha not in compton_cache:
        compton_cache[alpha] = _build_compton_cdf(alpha)
    return compton_cache[alpha]

def sample_compton_x(alpha):
    x_vals, cdf = get_compton_sampler(alpha)
    r = random.random()
    idx = np.searchsorted(cdf, r)
    if idx == 0:
        return x_vals[0]
    if idx >= len(cdf):
        return x_vals[-1]
    x0 = x_vals[idx-1]
    x1 = x_vals[idx]
    y0 = cdf[idx-1]
    y1 = cdf[idx]
    return x0 + (x1 - x0) * (r - y0) / (y1 - y0)

def compton_electron_angle(alpha, cos_theta):
    theta = math.acos(cos_theta)
    tan_half = math.tan(theta / 2.0)
    cot_phi = (1 + alpha) * tan_half
    phi = math.atan2(1.0, cot_phi)
    return math.cos(phi)

# Geometry
def inside_phantom(pos):
    x, y, z = pos
    return (abs(x) < PHANTOM_HALF and abs(y) < PHANTOM_HALF and 
            PHANTOM_Z_MIN <= z <= PHANTOM_Z_MAX)

def in_scoring(pos):
    x, y, z = pos
    return (abs(z - SCORE_Z_CENTER) <= SCORE_Z_HALF and 
            x*x + y*y <= SCORE_RADIUS_SQ)

# Rotate direction
def rotate_direction(dir_vec, cos_theta, phi):
    theta = math.acos(cos_theta)
    u = dir_vec
    if abs(u[0]) < 0.9:
        v = np.array([1.0, 0.0, 0.0])
    else:
        v = np.array([0.0, 1.0, 0.0])
    v = v - np.dot(v, u) * u
    v = v / np.linalg.norm(v)
    w = np.cross(u, v)
    new_dir = (math.cos(phi) * math.sin(theta) * v + 
               math.sin(phi) * math.sin(theta) * w + 
               math.cos(theta) * u)
    return new_dir / np.linalg.norm(new_dir)

# Source
def source_photon():
    x0 = random.uniform(-FIELD_HALF, FIELD_HALF)
    y0 = random.uniform(-FIELD_HALF, FIELD_HALF)
    dx = x0 - SOURCE[0]
    dy = y0 - SOURCE[1]
    dz = 0.0 - SOURCE[2]
    norm = math.sqrt(dx*dx + dy*dy + dz*dz)
    dir_vec = np.array([dx, dy, dz]) / norm
    pos = np.array([x0, y0, 0.0])
    return pos, dir_vec

# Main simulation
def simulate(num_events=200, max_histories=5000000):
    events = []
    n_hist = 0
    n_prim_score = 0
    n_sec_score = 0
    n_escaped = 0
    n_absorbed = 0
    
    while len(events) < num_events and n_hist < max_histories:
        pos, dir_vec = source_photon()
        n_hist += 1
        energy = 2.0
        primary = True
        
        while True:
            if not inside_phantom(pos):
                n_escaped += 1
                break
            
            mu_t = mu_Total(energy)
            if mu_t <= 0:
                break
            mfp = 1.0 / mu_t
            ds = -math.log(random.random()) * mfp
            new_pos = pos + ds * dir_vec
            
            if not inside_phantom(new_pos):
                n_escaped += 1
                break
            
            # This is an interaction point
            mu_c = mu_Compton(energy)
            mu_r = mu_Rayleigh(energy)
            mu_p = mu_Photoelectric(energy)
            mu_pp = mu_Pair(energy)
            
            r = random.random() * mu_t
            
            if r < mu_c:
                # Compton
                alpha = energy / m_e
                x = sample_compton_x(alpha)
                cos_theta = 1.0 - x
                e_out = energy / (1 + alpha * x)
                t_e = energy - e_out
                cos_electron = compton_electron_angle(alpha, cos_theta)
                phi = random.random() * 2 * math.pi
                new_dir = rotate_direction(dir_vec, cos_theta, phi)
                if in_scoring(new_pos):
                    events.append({
                        "process": "compton",
                        "e_in_MeV": energy,
                        "e_out_MeV": e_out,
                        "cos_scatter": cos_theta,
                        "t_e_MeV": t_e,
                        "cos_electron": cos_electron,
                        "t_positron_MeV": None
                    })
                    if primary:
                        n_prim_score += 1
                    else:
                        n_sec_score += 1
                    if len(events) >= num_events:
                        return events, n_hist, n_prim_score, n_sec_score, n_escaped, n_absorbed
                energy = e_out
                dir_vec = new_dir
                pos = new_pos
                primary = False
                
            elif r < mu_c + mu_r:
                # Rayleigh
                theta = random.random() * 0.1
                cos_theta = math.cos(theta)
                e_out = energy
                phi = random.random() * 2 * math.pi
                new_dir = rotate_direction(dir_vec, cos_theta, phi)
                if in_scoring(new_pos):
                    events.append({
                        "process": "rayleigh",
                        "e_in_MeV": energy,
                        "e_out_MeV": e_out,
                        "cos_scatter": cos_theta,
                        "t_e_MeV": 0.0,
                        "cos_electron": None,
                        "t_positron_MeV": None
                    })
                    if primary:
                        n_prim_score += 1
                    else:
                        n_sec_score += 1
                    if len(events) >= num_events:
                        return events, n_hist, n_prim_score, n_sec_score, n_escaped, n_absorbed
                dir_vec = new_dir
                pos = new_pos
                primary = False
                
            elif r < mu_c + mu_r + mu_p:
                # Photoelectric
                t_e = energy
                r_ang = random.random()
                theta = math.acos(1 - 0.5 * r_ang**2)
                cos_electron = math.cos(theta)
                if in_scoring(new_pos):
                    events.append({
                        "process": "photoelectric",
                        "e_in_MeV": energy,
                        "e_out_MeV": None,
                        "cos_scatter": None,
                        "t_e_MeV": t_e,
                        "cos_electron": cos_electron,
                        "t_positron_MeV": None
                    })
                    if primary:
                        n_prim_score += 1
                    else:
                        n_sec_score += 1
                    if len(events) >= num_events:
                        return events, n_hist, n_prim_score, n_sec_score, n_escaped, n_absorbed
                n_absorbed += 1
                break
                
            else:
                # Pair production
                total_ke = energy - 1.022
                f = random.random()
                t_e = f * total_ke
                t_positron = (1 - f) * total_ke
                theta_e = random.random() * 0.2
                cos_electron = math.cos(theta_e)
                if in_scoring(new_pos):
                    events.append({
                        "process": "pair",
                        "e_in_MeV": energy,
                        "e_out_MeV": None,
                        "cos_scatter": None,
                        "t_e_MeV": t_e,
                        "cos_electron": cos_electron,
                        "t_positron_MeV": t_positron
                    })
                    if primary:
                        n_prim_score += 1
                    else:
                        n_sec_score += 1
                    if len(events) >= num_events:
                        return events, n_hist, n_prim_score, n_sec_score, n_escaped, n_absorbed
                n_absorbed += 1
                break
                
    return events, n_hist, n_prim_score, n_sec_score, n_escaped, n_absorbed

# Run
random.seed(98765)
print("Starting simulation for 200 events...")
t0 = time.time()
evts, n_hist, n_prim, n_sec, n_esc, n_abs = simulate(200, 5000000)
t1 = time.time()
print(f"Time: {t1-t0:.2f} s")
print(f"Histories: {n_hist}, events: {len(evts)}, primary: {n_prim}, scattered: {n_sec}")
print(f"Escaped: {n_esc}, Absorbed: {n_abs}")

# Statistics
process_counts = {}
for e in evts:
    p = e['process']
    process_counts[p] = process_counts.get(p, 0) + 1
print("Process counts:", process_counts)

# Check energies
print("\nFirst 10 events:")
for e in evts[:10]:
    print(f"  {e['process']:15s} E_in={e['e_in_MeV']:.3f} E_out={str(e['e_out_MeV'] if e['e_out_MeV'] is not None else 'null'):8s} cos_scat={str(e['cos_scatter'] if e['cos_scatter'] is not None else 'null'):8s} t_e={e['t_e_MeV']:.4f}")