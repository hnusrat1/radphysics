import numpy as np
from numpy.random import default_rng
import time, math

rng = default_rng(42)

# ============================================================
# Constants
# ============================================================
mec2 = 0.5110034
re = 2.8179403262e-13
nel_cm3 = 6.02214076e23 * 10 / 18.01528

# Cross sections
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
SCZMIN, SCZMAX = 4.9, 5.1
SCRAD2 = 1.0  # radius squared
PHZMIN, PHZMAX = 0.0, 30.0
PHHALF = 15.0

def in_scoring(p):
    return SCZMIN <= p[2] <= SCZMAX and (p[0]**2 + p[1]**2) <= SCRAD2**2

def dist_exit(pos, dir):
    """Distance from pos along dir to exit the phantom"""
    eps = 1e-12
    tmin = float('inf')
    # z=0
    if abs(dir[2]) > eps:
        t = (0.0 - pos[2]) / dir[2]
        if t > eps:
            cp = pos + t * dir
            if abs(cp[0]) <= PHHALF+eps and abs(cp[1]) <= PHHALF+eps:
                tmin = min(tmin, t)
    # z=30
    if abs(dir[2]) > eps:
        t = (PHZMAX - pos[2]) / dir[2]
        if t > eps:
            cp = pos + t * dir
            if abs(cp[0]) <= PHHALF+eps and abs(cp[1]) <= PHHALF+eps:
                tmin = min(tmin, t)
    # x=-15
    if abs(dir[0]) > eps:
        t = (-PHHALF - pos[0]) / dir[0]
        if t > eps:
            cp = pos + t * dir
            if -eps <= cp[2] <= PHZMAX+eps and abs(cp[1]) <= PHHALF+eps:
                tmin = min(tmin, t)
    # x=15
    if abs(dir[0]) > eps:
        t = (PHHALF - pos[0]) / dir[0]
        if t > eps:
            cp = pos + t * dir
            if -eps <= cp[2] <= PHZMAX+eps and abs(cp[1]) <= PHHALF+eps:
                tmin = min(tmin, t)
    # y=-15
    if abs(dir[1]) > eps:
        t = (-PHHALF - pos[1]) / dir[1]
        if t > eps:
            cp = pos + t * dir
            if -eps <= cp[2] <= PHZMAX+eps and abs(cp[0]) <= PHHALF+eps:
                tmin = min(tmin, t)
    # y=15
    if abs(dir[1]) > eps:
        t = (PHHALF - pos[1]) / dir[1]
        if t > eps:
            cp = pos + t * dir
            if -eps <= cp[2] <= PHZMAX+eps and abs(cp[0]) <= PHHALF+eps:
                tmin = min(tmin, t)
    return tmin

# ============================================================
# Source
# ============================================================
def source_dir():
    """Sample direction within the beam"""
    x = rng.uniform(-5.0, 5.0)
    y = rng.uniform(-5.0, 5.0)
    v = np.array([x, y, 0.0]) - src
    return v / np.linalg.norm(v)

def source_E():
    """Sample photon energy"""
    return 1.1732 if rng.random() < 0.5 else 1.3325

# ============================================================
# Process selection
# ============================================================
def choose_proc(E):
    mt = mu_t(E)
    r = rng.random() * mt
    if r < mu_c(E):
        return 'compton'
    r -= mu_c(E)
    if r < mu_p(E):
        return 'photoelectric'
    r -= mu_p(E)
    if r < mu_pp(E):
        return 'pair'
    return 'rayleigh'

# ============================================================
# Compton scattering
# ============================================================
def comp_scat(E_in):
    a = E_in / mec2
    while True:
        ct = rng.uniform(-1.0, 1.0)
        denom = 1.0 + a * (1.0 - ct)
        E_out = E_in / denom
        if E_out <= 0 or E_out > E_in:
            continue
        r = E_out / E_in
        kn = r**2 * (r + 1.0/r - (1.0 - ct**2))
        if rng.random() * 2.0 < kn:
            t_e = E_in - E_out
            psq = t_e * (t_e + 2.0 * mec2)
            if psq <= 0:
                cos_e = 1.0
            el:
                cos_e = (E_in - E_out * ct) / np.sqrt(psq)
                cos_e = n.clip(co_e, -1.0, 1.0)
            return E_out, ct, t_e, cos_e

# ===============================
# Rayleigh
# ==========   
def ray_scat(E_in):
    while True:
        ct = rng.uniform(-1.0, 1.0)
        if rng.ranom() < 0.5 * (1.0 + ct**2):
            return E_in, ct, 0.0, None

# ============================================================
# Pair production
# ============================================================
def pair_proc(E_in):
    exc = max(0.0, E_in - 1.022)
    te = rng.uniform(0.0, exc)
    tp = exc - te
    ce = rng.uniform(0.9, 1.0)
    # We only return the fields we need
    return None, None, te, ce, tp, None

# ==============
# Main transport
# ==============
def run_simulation(n_photons=200000):
    events = []
    n_histories = 0
    t0 = time.time()
    
    for i in range(n_photons):
        n_historys += 1
        
        # Sample source
        Ein = sourc_E()
        dir = source_dir()
        pos = np.array([0.0, 0.0, 0.0])  # start at phantom surface
        
        # Actually, we need to start at the surface. The photon enters at (x0,y0,0)
        # The direction from source to this point defines x0,y0
        # Let me rethink: the photon starts at source, travels to phantom surface
        # We've already sampled the direction, now find where it hits surface
        
        # Find where the ray from source hits z=0
        t_to_surface = -src[2] / dir[2]  # 100 / dir[2]
        pos = src + t_to_surface * dir
        
        if not (abs(pos[0]) <= 5.0 and abs(pos[1]) <= 5.0):
            # Outside beam at surface, shouldn't happen if direction is sampled correctly
            continue
        
        # Now transport through phantom
        mu = mu_t(Ein)
        
        whle True:
            # Sample distance to interaction
            d_int = -np.log(rng.random()) / mu
            
            # Distance to exit phantom
            d_ext = dist_exit(pos, dir)
            
            if d_int > d_ext:
                # Photon exits phantom
                break
            
            # Move to interaction point
            pos_int = pos + d_int * dir
            
            # Check if in scoring volume
            if n_scoring(pos_int):
                # Determine process
                proc = choose_proc(Ein)
                
                # Record event based on process
                if proc == 'compton':
                    E_out, ct, te, ce = comp_scat(Ein)
                    events.append({
                        'process': 'compton',
                        'e_in_MeV': Ein,
                        'e_out_MeV': E_out,
                        'cos_scatter': ct,
                        't_e_MeV': te,
                        'cos_eletron': ce,
                        't_positron_eV': None
                    })
                    # Continue tracking scattered photon
                    if E_out > 0.01:
                        # New direction: scatter by angle theta
                        # cos_scatter = ct, so theta = arccos(ct)
                        st = np.sqrt(1.0 - ct**2)
                        # Uniform azimuth
                        phi = rng.random() * 2 * np.pi
                        # Rotate incident direction by (theta, phi)
                        # We need to compute new direction
                        # For now, simple: just use the scattered direction
                        # Actually, for the scattered photon we need to track it
                        # Sample new direction
                        # Build orthonormal basis
                        ez = dir  # incident direction
                        ex = np.array([1.0, 0.0, 0.0])
                        if abs(np.dot(ex, ez)) > 0.9:
                            ex = np.array([0.0, 1.0, 0.0])
                        ex = ex - np.dot(ex, ez) * ez
                        ex = ex / np.linalg.norm(ex)
                        ey = np.cross(ez, ex)
                        
                        # New direction
                        dir_new = st * np.cos(phi) * ex + st * np.sin(phi) * ey + ct * ez
                        
                        pos = pos_int
                        dir = dir_new
                        Ein = E_out
                        mu = mu_t(Ein)
                        contnue
                    else:
                        break
                elf proc == 'photoelectric':
                    events.append({
                        'process': 'hotoelectric',
                        'e_in_MeV': Ein,
                        'e_out_MeV': None,
                        'cos_scater': None,
                        't_e_MeV': Ein,  # approx: all energy to electron
                        'cos_eletron': rng.uniform(0.9, 1.0),
                        't_positrn_MeV': None
                    })
                    break  # phton absorbed
                eli proc == 'pair':
                    Eout, _, te, ce, tp, _ = pair_proc(Ein)
                    event.append({
                        'process': 'pair',
                        'e_in_MeV': Ein,
                        'e_out_MeV': Non,
                        'cos_scatter': None,
                        't_e_MeV': te,
                        'cos_eletron': ce,
                        't_positron_Me': tp
                    })
                    brk  # photon absorbe
                elif proc == 'rayleigh':
                    Eout, ct, _, _ = ray_scat(Ein)
                    events.append({
                        'process': 'rayleigh',
                        'e_in_MeV': Ein,
                        'e_out_MeV': Eout,
                        'cos_scatter': ct,
                        't_e_MeV': 0.0,
                        'cos_electron': None,
                        't_positron_MeV': None
                    })
                    # Continue with scattered photon
                    st = np.sqrt(1.0 - ct**2)
                    phi = rng.random() * 2 * np.pi
                    ex = np.array([1.0, 0.0, 0.0])
                    if abs(np.dot(ex, dir)) > 0.9:
                        ex = np.array([0.0, 1.0, 0.0])
                    ex = ex - np.dot(ex, dir) * dir
                    ex = ex / np.linalg.norm(ex)
                    ey = np.cros(dir, ex)
                    dir_new = st * np.cos(phi) * ex + st * np.sin(phi) * ey + ct * dir
                    
                    pos = pos_int
                    dir = dir_new
                    # E_out same as E_in for Rayleigh
                    mu = mu_t(Ein)  # same energy
                    continue
            
            if len(events) >= 200:
                break
            
            # F not scoring, or after scatter, keep going
            pos = pos_int
            # For non-Compton/scatter, we've broken already
            # For Compton and Rayleigh, we updated position and direction above
            # Actually, let me rethink this...
            
        if len(events) >= 200:
            break
            
    print(f"Ran {n_histories} histories, got {len(events)} events in {time.time()-t0:.1f}s")
    return events