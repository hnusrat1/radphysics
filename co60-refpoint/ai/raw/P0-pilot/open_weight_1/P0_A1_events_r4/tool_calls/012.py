import numpy as np
import random
import time
from math import sqrt, log, exp, cos, sin, pi

# ---- Constants ----
m_e = 0.510998950
r_e = 2.8179403262e-13
N_A = 6.02214076e23
N_e_per_g = N_A * 10 / 18.01528

# Precompute cross sections at a fine grid for fast lookup
# We'll use direct computation at each energy using analytic formulas
# to avoid interpolation overhead.

# But actually, let's just compute mu_total analytically each time.
# The Klein-Nishina formula is simple enough.

def mu_total_fast(E):
    """Fast computation of total mu and component fractions."""
    alpha = E / m_e
    if alpha <= 0:
        return 0.0, 0.0, 0.0, 0.0, 0.0
    
    # Compton
    term1 = (1+alpha)/alpha**2 * (2*(1+alpha)/(1+2*alpha) - log(1+2*alpha)/alpha)
    term2 = log(1+2*alpha) / (2*alpha)
    term3 = (1+3*alpha) / (1+2*alpha)**2
    sigma = 2*pi * r_e**2 * (term1 + term2 - term3)
    mu_c = sigma * N_e_per_g
    
    # Photoelectric: use power law approximation
    if E < 0.01:
        mu_pe = 4.89 * (E/0.01)**(-3.5)
    elif E < 0.1:
        mu_pe = 4.89 * (E/0.01)**(-3.5)  # same power law
    elif E < 1.0:
        mu_pe = 0.0017 * (E/0.1)**(-3.2)
    else:
        mu_pe = 7e-6 * (E/1.0)**(-3.5)
    
    # Pair production
    if E <= 2*m_e:
        mu_pp = 0.0
    elif E < 2.0:
        mu_pp = 0.00013 * (E - 2*m_e) / (2.0 - 2*m_e)  # linear rise
    elif E < 5.0:
        mu_pp = 0.00013 + (0.0015-0.00013)*(E-2)/3
    elif E < 10.0:
        mu_pp = 0.0015 + (0.0048-0.0015)*(E-5)/5
    else:
        mu_pp = 0.0048 + (0.0077-0.0048)*(E-10)/10
    
    # Rayleigh
    if E < 0.01:
        mu_r = 0.16
    elif E < 0.1:
        mu_r = 0.16 * (E/0.01)**(-3.5)
    elif E < 0.5:
        mu_r = 0.0002 * (E/0.1)**(-4)
    else:
        mu_r = 0.0
    
    mu_t = mu_c + mu_pe + mu_pp + mu_r
    return mu_t, mu_c, mu_pe, mu_pp, mu_r

# Fast Compton sampling
def sample_compton_fast(E, rng):
    alpha = E / m_e
    while True:
        xi1 = rng.random()
        xi2 = rng.random()
        mu = 2.0 * xi1 - 1.0
        ratio = 1.0 / (1.0 + alpha * (1.0 - mu))
        f_kn = ratio**2 * (ratio + 1.0/ratio - (1.0 - mu**2))
        if xi2 <= f_kn * 0.5:
            e_out = E * ratio
            cos_scatter = mu
            break
    t_e = E - e_out
    # Electron direction
    denom = sqrt(E**2 + e_out**2 - 2.0 * E * e_out * cos_scatter)
    if denom > 0:
        cos_electron = (E - e_out * cos_scatter) / denom
    else:
        cos_electron = 1.0
    return e_out, cos_scatter, t_e, cos_electron

# Fast rotation
def rotate_dir_fast(dx, dy, dz, cos_theta, azimuth):
    """Rotate direction (dx,dy,dz) by polar angle theta (cos_theta) and azimuth.
    Returns (new_dx, new_dy, new_dz)."""
    # Build orthonormal basis
    if abs(dz) < 0.9:
        # x' = normalize(z_hat cross d)
        norm = sqrt(dx*dx + dy*dy)  # d cross z_hat = (-dy, dx, 0)
        ux = -dy / norm
        uy = dx / norm
        uz = 0.0
    else:
        norm = sqrt(dy*dy + dz*dz)
        ux = 0.0
        uy = -dz / norm
        uz = dy / norm
    
    # y' = d cross x'
    vx = dy*uz - dz*uy
    vy = dz*ux - dx*uz
    vz = dx*uy - dy*ux
    
    sin_theta = sqrt(max(0.0, 1.0 - cos_theta*cos_theta))
    ca = cos(azimuth)
    sa = sin(azimuth)
    
    new_dx = sin_theta * ca * ux + sin_theta * sa * vx + cos_theta * dx
    new_dy = sin_theta * ca * uy + sin_theta * sa * vy + cos_theta * dy
    new_dz = sin_theta * ca * uz + sin_theta * sa * vz + cos_theta * dz
    
    return new_dx, new_dy, new_dz

# ---- Geometry ----
Z_SURFACE = 100.0
Z_EXIT = 130.0
FIELD_HALF = 5.0

SV_Z_MIN = 109.9
SV_Z_MAX = 110.1
SV_RADIUS_SQ = 1.0

def simulate_one_fast(rng):
    """Simulate one photon history, return list of events in scoring volume."""
    events = []
    
    # Sample initial direction: uniform in 10x10 cm field at z=100
    u = rng.uniform(-FIELD_HALF, FIELD_HALF)
    v = rng.uniform(-FIELD_HALF, FIELD_HALF)
    # Direction normalization
    inv_norm = 1.0 / sqrt(u*u + v*v + 10000.0)
    dx = u * inv_norm
    dy = v * inv_norm
    dz = 100.0 * inv_norm
    
    # Initial position at surface
    px = u
    py = v
    pz = Z_SURFACE
    energy = 2.0
    
    for _ in range(200):
        mu_t, mu_c, mu_pe, mu_pp, mu_r = mu_total_fast(energy)
        if mu_t <= 0:
            break
        
        mfp = 1.0 / mu_t
        s = -mfp * log(1.0 - rng.random())
        
        new_px = px + s * dx
        new_py = py + s * dy
        new_pz = pz + s * dz
        
        if new_pz >= Z_EXIT or new_pz < Z_SURFACE:
            break
        
        # Check scoring volume
        in_scoring = (new_pz >= SV_Z_MIN and new_pz <= SV_Z_MAX and 
                      new_px*new_px + new_py*new_py <= SV_RADIUS_SQ)
        
        # Sample interaction type
        xi = rng.random()
        p_c = mu_c / mu_t
        p_pe = mu_pe / mu_t
        p_pp = mu_pp / mu_t
        
        e_in = energy
        e_out = None
        cos_scatter = None
        t_e = 0.0
        cos_electron = None
        t_pos = None
        absorbed = False
        
        if xi < p_c:
            process = 'compton'
            e_out, cos_scatter, t_e, cos_electron = sample_compton_fast(energy, rng)
            azimuth = rng.random() * 2*pi
            dx, dy, dz = rotate_dir_fast(dx, dy, dz, cos_scatter, azimuth)
            energy = e_out
        elif xi < p_c + p_pe:
            process = 'photoelectric'
            t_e = energy
            cos_electron = rng.uniform(-1.0, 1.0)
            energy = 0.0
            absorbed = True
        elif xi < p_c + p_pe + p_pp:
            process = 'pair'
            T_total = energy - 2*m_e
            if T_total > 0:
                t_e = rng.uniform(0.0, T_total)
                t_pos = T_total - t_e
            else:
                t_e = 0.0
                t_pos = 0.0
            theta_avg = m_e / energy if energy > 0 else 0
            if theta_avg > 0:
                theta = theta_avg * sqrt(-2.0 * log(1.0 - rng.random()))
                cos_electron = cos(theta)
            else:
                cos_electron = 1.0
            energy = 0.0
            absorbed = True
        else:
            process = 'rayleigh'
            Z_eff = 10.0
            theta0 = 0.02 / (max(energy, 0.01) * Z_eff**(1/3))
            theta = theta0 * sqrt(-2.0 * log(1.0 - rng.random()))
            cos_scatter = cos(theta)
            e_out = energy
            azimuth = rng.random() * 2*pi
            dx, dy, dz = rotate_dir_fast(dx, dy, dz, cos_scatter, azimuth)
            cos_electron = None
            t_e = 0.0
        
        if in_scoring:
            events.append({
                'process': process,
                'e_in_MeV': float(e_in),
                'e_out_MeV': float(e_out) if e_out is not None else None,
                'cos_scatter': float(cos_scatter) if cos_scatter is not None else None,
                't_e_MeV': float(t_e),
                'cos_electron': float(cos_electron) if cos_electron is not None else None,
                't_positron_MeV': float(t_pos) if t_pos is not None else None
            })
        
        px, py, pz = new_px, new_py, new_pz
        
        if absorbed:
            break
    
    return events

# Benchmark
rng = random.Random(42)
t0 = time.time()
total_events = []
for i in range(100000):
    evts = simulate_one_fast(rng)
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