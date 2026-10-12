import numpy as np
import math
import random
import time

# -----------------------------
# Constants
# -----------------------------
m_e = 0.51099895  # MeV
r_e_sq = 7.9408e-26  # cm^2
n_e = 3.3428e23  # electrons/cm^3 for water

PHANTOM_SIZE = 15.0  # half size in x and y
PHANTOM_Z_MAX = 30.0
PHANTOM_Z_MIN = 0.0
SSD = 100.0
SOURCE_POS = np.array([0.0, 0.0, -SSD])  # z = -100
FIELD_HALF = 5.0  # half of 10 cm field at surface (z=0)

# Scoring volume
SCORE_Z_CENTER = 10.0
SCORE_Z_HALF = 0.1  # 0.1 cm half-thickness, total 0.2 cm
SCORE_RADIUS = 1.0  # cm
SCORE_RADIUS_SQ = SCORE_RADIUS**2

# -----------------------------
# Cross sections
# -----------------------------
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
    # Power law: ~ E^-2
    return 0.0008 * (1.0 / max(E, 0.01))**2

def mu_Photoelectric(E):
    if E <= 0:
        return 0.0
    # ~ E^-3.5
    return 1.9e-5 * (1.0 / max(E, 0.01))**3.5

def mu_Pair(E):
    if E <= 1.022:
        return 0.0
    # Piecewise linear approximation
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

# -----------------------------
# Compton sampling tables (cached)
# -----------------------------
# We'll cache CDF tables for different alpha values
compton_cache = {}

def get_compton_cdf(alpha):
    if alpha not in compton_cache:
        x_vals, cdf = _build_compton_cdf(alpha)
        compton_cache[alpha] = (x_vals, cdf)
    return compton_cache[alpha]

def _build_compton_cdf(alpha):
    n_pts = 2000
    x_vals = np.linspace(0, 2, n_pts)
    term1 = 1.0 / (1 + alpha * x_vals)
    term2 = 1.0 / (1 + alpha * x_vals)**3
    term3 = (2 * x_vals - x_vals**2) / (1 + alpha * x_vals)**2
    pdf = term1 + term2 - term3
    pdf = np.maximum(pdf, 0.0)
    cdf = np.cumsum(pdf)
    cdf = cdf / cdf[-1]
    return x_vals, cdf

def sample_compton(alpha):
    x_vals, cdf = get_compton_cdf(alpha)
    r = random.random()
    idx = np.searchsorted(cdf, r)
    if idx == 0:
        x = x_vals[0]
    elif idx >= len(cdf):
        x = x_vals[-1]
    else:
        x0 = x_vals[idx-1]
        x1 = x_vals[idx]
        y0 = cdf[idx-1]
        y1 = cdf[idx]
        x = x0 + (x1 - x0) * (r - y0) / (y1 - y0)
    return x  # 1 - cosθ

# -----------------------------
# Electron angle from Compton kinematics
# -----------------------------
def compton_electron_angle(alpha, cos_theta):
    """Return cos(phi) where phi is the electron recoil angle relative to incident photon."""
    # cot(phi) = (1 + alpha) * tan(theta/2)
    theta = math.acos(cos_theta)
    tan_half_theta = math.tan(theta / 2.0)
    cot_phi = (1 + alpha) * tan_half_theta
    phi = math.atan2(1.0, cot_phi)  # cot phi = cot, so phi = arctan(1/cot_phi)
    return math.cos(phi)

# -----------------------------
# Geometry functions
# -----------------------------
def is_inside_phantom(pos):
    x, y, z = pos
    return (abs(x) < PHANTOM_SIZE and abs(y) < PHANTOM_SIZE and 
            PHANTOM_Z_MIN <= z <= PHANTOM_Z_MAX)

def is_in_scoring(pos):
    x, y, z = pos
    if abs(z - SCORE_Z_CENTER) > SCORE_Z_HALF:
        return False
    if x*x + y*y > SCORE_RADIUS_SQ:
        return False
    return True

# -----------------------------
# Source sampling
# -----------------------------
def sample_source_photon():
    """Return initial position (at surface) and direction (unit vector)."""
    # Uniformly sample a point on the field at z=0
    x0 = random.uniform(-FIELD_HALF, FIELD_HALF)
    y0 = random.uniform(-FIELD_HALF, FIELD_HALF)
    # Direction from source to that point
    dx = x0 - SOURCE_POS[0]
    dy = y0 - SOURCE_POS[1]
    dz = 0.0 - SOURCE_POS[2]  # from z=-100 to z=0 is 100
    norm = math.sqrt(dx*dx + dy*dy + dz*dz)
    dir_vec = np.array([dx, dy, dz]) / norm
    pos = np.array([x0, y0, 0.0])
    return pos, dir_vec

# -----------------------------
# Main simulation
# -----------------------------
def simulate_events(n_events_needed=200, max_histories=2000000):
    events = []
    n_histories = 0
    n_interactions_total = 0
    
    # Stats
    n_primary_in_scoring = 0
    n_scattered_in_scoring = 0
    n_escaped = 0
    n_absorbed = 0
    
    while len(events) < n_events_needed and n_histories < max_histories:
        # Sample source photon
        pos, dir_vec = sample_source_photon()
        n_histories += 1
        energy = 2.0  # MeV
        primary = True  # has not interacted yet
        
        # Transport
        while is_inside_phantom(pos):
            mu = mu_Total(energy)
            if mu <= 0:
                # Photon cannot interact (e.g., very low energy with no cross section?)
                # Should not happen
                break
            mfp = 1.0 / mu
            
            # Sample distance to next interaction
            ds = -math.log(random.random()) * mfp
            
            # Move
            new_pos = pos + ds * dir_vec
            
            # Check if new position is outside phantom
            if not is_inside_phantom(new_pos):
                # Photon escapes
                n_escaped += 1
                break
            
            # Check if interaction occurs within scoring volume
            if is_in_scoring(new_pos):
                # Record the event
                # Determine process
                mu_c = mu_Compton(energy)
                mu_r = mu_Rayleigh(energy)
                mu_p = mu_Photoelectric(energy)
                mu_pp = mu_Pair(energy)
                mu_t = mu_c + mu_r + mu_p + mu_pp
                r = random.random() * mu_t
                
                # Initialize event fields
                process = None
                e_in = energy
                e_out = None
                cos_scatter = None
                t_e = 0.0
                cos_electron = None
                t_positron = None
                
                if r < mu_c:
                    process = "compton"
                    # Compton scattering
                    alpha = energy / m_e
                    x = sample_compton(alpha)  # 1 - cosθ
                    cos_theta = 1.0 - x
                    # Scattered photon energy
                    e_out = energy / (1 + alpha * x)
                    cos_scatter = cos_theta
                    # Electron kinetic energy
                    t_e = energy - e_out  # binding energy negligible
                    # Electron angle
                    cos_electron = compton_electron_angle(alpha, cos_theta)
                    # Update photon for continuation
                    # Compute new direction
                    # Scattering angle theta, but also need azimuthal angle phi
                    phi = random.random() * 2 * math.pi
                    # Rotate direction vector
                    # We need to rotate the current direction by angle theta around a perpendicular axis
                    # Compute two orthogonal vectors
                    u = dir_vec
                    # Find a vector perpendicular to u
                    if abs(u[0]) < 0.9:
                        v = np.array([1.0, 0.0, 0.0])
                    else:
                        v = np.array([0.0, 1.0, 0.0])
                    v = v - np.dot(v, u) * u
                    v = v / np.linalg.norm(v)
                    w = np.cross(u, v)
                    # Rotated direction
                    new_dir = (math.cos(phi) * math.sin(theta) * v + 
                                math.sin(phi) * math.sin(theta) * w + 
                                math.cos(theta) * u)
                    dir_vec = new_dir / np.linalg.norm(new_dir)
                    energy = e_out
                    pos = new_pos
                    primary = False
                    
                elif r < mu_c + mu_r:
                    process = "rayleigh"
                    # Rayleigh scattering: coherent, same energy
                    e_out = energy
                    # Scattering angle: very small. Use form factor.
                    # Approximate: sample cosθ from Rayleigh distribution using a simple model.
                    # For water at these energies, the Rayleigh scattering is strongly forward.
                    # We'll approximate by sampling from a Gaussian with sigma ~ 0.01 rad?
                    # Actually, for 2 MeV, the Rayleigh scattering angle is very small (< 0.1 rad).
                    # Let's use a simple approximation: cosθ ~ 1 - 0.5 * (θ)^2, with θ small.
                    # We'll sample from a uniform distribution of cosθ between cos(5°)~0.996 and 1.0.
                    theta = random.random() * 0.1  # 0 to 0.1 rad (~5.7°)
                    cos_theta = math.cos(theta)
                    cos_scatter = cos_theta
                    t_e = 0.0  # no electron
                    cos_electron = None
                    # Update photon direction (same energy)
                    phi = random.random() * 2 * math.pi
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
                    dir_vec = new_dir / np.linalg.norm(new_dir)
                    # energy unchanged
                    pos = new_pos
                    primary = False
                    
                elif r < mu_c + mu_r + mu_p:
                    process = "photoelectric"
                    # Photon absorbed
                    e_out = None
                    cos_scatter = None
                    t_e = energy  # kinetic energy of electron (minus binding energy, negligible)
                    # Electron angular distribution: forward peaked. 
                    # At high energies, photoelectron is emitted in forward direction (θ ≈ 0).
                    # Use an approximate distribution: cosθ ~ uniform in [cos(30°), 1]? Actually,
                    # for Z=8 (oxygen), the photoelectron angular distribution is complex.
                    # We'll sample from: cosθ = 1 - 0.5 * (abs(r)^{1/3})? 
                    # Simple: sample cosθ from distribution proportional to sin^2θ? No.
                    # Let's use a simple approximation: cosθ = 1 - a * r^2, where a is small.
                    # For simplicity, assume isotropic? But physical distribution is forward.
                    # We'll use: cosθ = 1 - 0.5 * (random.random())^2 (forward peaked)
                    r_angle = random.random()
                    theta = math.acos(1 - 0.5 * r_angle**2)  # forward peaked
                    cos_electron = math.cos(theta)
                    t_positron = None
                    # Photon absorbed, stop tracking
                    pos = new_pos
                    # Record event and break inner loop (photon gone)
                    # We'll record after the else
                    
                else:
                    process = "pair"
                    # Pair production
                    e_out = None
                    cos_scatter = None
                    # Energy sharing: the total kinetic energy of e+e- pair is E_in - 1.022 MeV
                    total_ke = energy - 1.022
                    # Assume uniform sharing: sample f uniformly in [0,1] for fraction to electron
                    f = random.random()
                    t_e = f * total_ke  # electron kinetic energy
                    t_positron = (1 - f) * total_ke  # positron kinetic energy
                    # Angular distribution: both forward peaked.
                    # Sample electron and positron angles around the incident direction.
                    # For small angles, cosθ ≈ 1 - θ^2/2
                    # Sample θ_e from distribution P(θ) ∝ θ^2? Actually, pair production angular distribution is complex.
                    # We'll approximate: electron and positron are emitted at small angles.
                    theta_e = random.random() * 0.2  # max ~0.2 rad (~11°)
                    cos_electron = math.cos(theta_e)
                    # For positron, we also need its angle. But the question only asks for cos_electron (electron angle).
                    # The positron angle is not requested.
                    # Photon absorbed, stop tracking
                    pos = new_pos
                    
                # Record event
                events.append({
                    "process": process,
                    "e_in_MeV": e_in,
                    "e_out_MeV": e_out,
                    "cos_scatter": cos_scatter,
                    "t_e_MeV": t_e,
                    "cos_electron": cos_electron,
                    "t_positron_MeV": t_positron
                })
                
                if primary:
                    n_primary_in_scoring += 1
                else:
                    n_scattered_in_scoring += 1
                
                if len(events) >= n_events_needed:
                    break
                
                # If photon is not absorbed, continue tracking; else break
                if process in ("photoelectric", "pair"):
                    break
                # else (Compton, Rayleigh) continue with new pos and dir
                
            else:
                # Not in scoring volume, just update position
                pos = new_pos
                # Continue loop (no interaction here, but we sampled an interaction point)
                # Actually, we are at the interaction point. We must now determine the interaction
                # and update the photon state.
                # Wait, the standard MC approach: when we sample a path length and it ends at an interaction point,
                # we must process that interaction regardless of location. But we only want to record events
                # that occur in the scoring volume. For interactions outside, we still need to process them
                # because they affect the photon state (scattering, absorption).
                
                # So we need to process the interaction even if not in scoring volume.
                # Let's restructure.
        else:
            # Photon escaped or was absorbed before reaching scoring volume
            pass
        
        # If we have enough events, stop
        if len(events) >= n_events_needed:
            break
    
    return events, n_histories

# Run a test
random.seed(12345)
evts, n_hist = simulate_events(10, 200000)
print(f"Got {len(evts)} events from {n_hist} histories")
for e in evts:
    print(e)