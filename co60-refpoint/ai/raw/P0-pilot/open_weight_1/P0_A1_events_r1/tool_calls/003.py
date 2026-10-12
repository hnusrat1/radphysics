import numpy as np
import math
from numpy import random

# Constants
m_e = 0.511  # MeV
r_e = 2.818e-13  # cm, classical electron radius

# Cross section functions (more precise)
def compton_cs_per_electron(E):
    """Total Compton cross section per electron (cm^2)"""
    if E <= 0:
        return 0
    epsilon = E / m_e
    k = epsilon
    # Klein-Nishina total cross section formula
    sigma = 2 * np.pi * r_e**2 * (
        (1 + k) / k**2 * (2 * (1 + k) / (1 + 2*k) - np.log(1 + 2*k) / k) +
        np.log(1 + 2*k) / (2*k) - (1 + 3*k) / (1 + 2*k)**2
    )
    return sigma

def mu_comp(E):
    """Compton linear attenuation coefficient for water (cm^-1)"""
    if E <= 0:
        return 0
    return compton_cs_per_electron(E) * 3.34277e23  # electrons/cm^3

def mu_photo(E):
    """Photoelectric linear attenuation coefficient for water (cm^-1)"""
    if E <= 0:
        return 0
    # Approximate fit for water using NIST data
    # Parameters from fitting
    if E < 0.01:
        return 10.0 * (0.01/E)**3.2
    elif E < 0.1:
        return 10.0 * (0.01/E)**2.9
    elif E < 0.5:
        return 0.04 * (0.1/E)**2.7
    else:
        return 0.0004 * (0.5/E)**3.0

def mu_pair(E):
    """Pair production linear attenuation coefficient for water (cm^-1)"""
    if E < 1.022:
        return 0.0
    # Approximate: linear from threshold to ~3 MeV, then logarithmic
    # At 2 MeV: ~0.0078
    # At 3 MeV: ~0.012
    # At 5 MeV: ~0.015
    # At 10 MeV: ~0.020
    if E < 3.0:
        return 0.0078 * (E - 1.022) / (2.0 - 1.022)
    else:
        return 0.012 * (1 + 0.15 * np.log(E/3))

def mu_rayleigh(E):
    """Coherent (Rayleigh) scattering linear attenuation coefficient for water (cm^-1)"""
    if E <= 0:
        return 0
    # Simple fit
    if E < 0.01:
        return 8.0 * (0.01/E)**2.6
    elif E < 0.1:
        return 8.0 * (0.01/E)**2.3
    elif E < 1.0:
        return 0.06 * (0.1/E)**2.3
    else:
        return 0.0003 * (1.0/E)**2.5

def mu_total(E):
    """Total linear attenuation coefficient for water (cm^-1)"""
    return mu_comp(E) + mu_photo(E) + mu_pair(E) + mu_rayleigh(E)

#=============================================================================
# Sampling functions
#=============================================================================

def sample_interaction_process(E):
    """Sample which process occurs, return process string"""
    mc = mu_comp(E)
    mp = mu_photo(E)
    mpp = mu_pair(E)
    mr = mu_rayleigh(E)
    mt = mc + mp + mpp + mr
    
    if mt == 0:
        return None
    
    r = random.random()
    cum = 0
    cum += mc/mt
    if r < cum:
        return 'compton'
    cum += mp/mt
    if r < cum:
        return 'photoelectric'
    cum += mpp/mt
    if r < cum:
        return 'pair'
    return 'rayleigh'

def sample_interaction_length(mu):
    """Sample distance to next interaction (cm)"""
    if mu <= 0:
        return float('inf')
    return -np.log(1 - random.random()) / mu

def sample_compton_angle(E_in):
    """Sample cos(theta) from Klein-Nishina distribution"""
    k = E_in / m_e
    max_val = 4.0  # (k'/k)^2 * (k/k' + k'/k - 1 + mu^2) has max near 2 for small k, near 2 for large k too... let me compute
    # For safety, find max numerically
    n_test = 200
    mus = np.linspace(-1, 1, n_test)
    vals = []
    for mu in mus:
        kp = k / (1 + k * (1 - mu))
        if kp > 0:
            val = (kp/k)**2 * (k/kp + kp/k - 1 + mu**2)
            vals.append(val)
        else:
            vals.append(0)
    max_val = max(vals) * 1.01
    
    # Rejection sampling
    while True:
        mu = random.uniform(-1, 1)
        kp = k / (1 + k * (1 - mu))
        if kp <= 0:
            continue
        val = (kp/k)**2 * (k/kp + kp/k - 1 + mu**2)
        if random.random() * max_val <= val:
            return mu

def compton_kinematics(E_in, costheta):
    """Return (E_out, cos_electron, t_e)"""
    E_out = E_in / (1 + (E_in/m_e) * (1 - costheta))
    t_e = E_in - E_out
    
    # Electron angle
    if E_in > 0 and E_out > 0:
        p_in = math.sqrt(E_in**2 + 2*E_in*m_e)
        p_out = math.sqrt(E_out**2 + 2*E_out*m_e)
        cos_e = (p_in - p_out * costheta) / math.sqrt(p_in**2 + p_out**2 - 2*p_in*p_out*costheta)
        cos_e = max(-1.0, min(1.0, cos_e))
    else:
        cos_e = 0.0
    
    return E_out, cos_e, t_e

def sample_pair_energies(E_in):
    """Sample electron and positron kinetic energies for pair production.
    Returns (t_e, t_positron, cos_electron, cos_positron)
    """
    # Total energy available: E_in - 2*m_e = E_in - 1.022
    # For simplicity, assume equal sharing (this is approximately correct for the screening regime)
    # The electron and positron share the available energy
    
    # In reality, the energy distribution is not exactly equal, but for our simplified model:
    available = E_in - 2 * m_e
    
    # Sample electron kinetic energy (roughly uniform between 0 and available)
    # Actually, for high-energy pair production, the distribution is approximately
    # symmetric, with the electron and positron sharing roughly equally
    # Let's sample with some variation
    f = random.uniform(0.3, 0.7)  # fraction for electron
    t_e = available * f
    t_pos = available * (1 - f)
    
    # Angle - positron and electron are emitted at small angles relative to incident photon
    # For our purposes, approximate with small forward angles
    # Mean angle ~ m_e/E for each particle
    mean_angle = m_e / (E_in/2)  # approximate
    theta_e = random.exponential(mean_angle)
    theta_pos = random.exponential(mean_angle)
    
    cos_e = math.cos(theta_e)
    cos_pos = math.cos(theta_pos)
    
    return t_e, t_pos, cos_e, cos_pos

def rayleigh_angle(E_in):
    """Sample Rayleigh scattering angle (very forward peaked)"""
    # Rayleigh scattering at high energies is strongly forward peaked
    # For simplicity, sample from a very narrow forward distribution
    # The angular distribution goes as ~ (1+cosθ)^2 * F(q,Z)^2
    # For water at 2 MeV, the typical angle is very small (fraction of a degree)
    sigma_theta = 0.001  # ~0.05 degrees
    costheta = math.cos(random.normal(0, sigma_theta))
    costheta = max(-1.0, min(1.0, costheta))
    return costheta

#=============================================================================
# Geometry
#=============================================================================

# Source at (0, 0, -100) if phantom surface at z=0
# Phantom: x in [-15, 15], y in [-15, 15], z in [0, 30]
# Field at surface (z=0): 10x10 cm, centered at (0,0)
# So beam pyramid: from (0,0,-100) to square [-5,5]x[-5,5] at z=0

def sample_beam_direction():
    """Sample a photon direction uniformly within the beam pyramid"""
    # Sample a point uniformly in the 10x10 cm square at z=0
    x0 = random.uniform(-5, 5)
    y0 = random.uniform(-5, 5)
    
    # Direction from source to this point
    dx = x0 - 0
    dy = y0 - 0
    dz = 0 - (-100)  # 100
    
    norm = math.sqrt(dx*dx + dy*dy + dz*dz)
    return np.array([dx/norm, dy/norm, dz/norm])

def point_in_phantom(pos):
    """Check if point is inside water phantom"""
    x, y, z = pos
    return abs(x) <= 15 and abs(y) <= 15 and 0 <= z <= 30

def distance_to_phantom_boundary(pos, dir_vec):
    """Distance along direction from pos to phantom boundary.
    dir_vec is a unit vector.
    Returns distance to exit phantom, or inf if outside.
    """
    x, y, z = pos
    dx, dy, dz = dir_vec
    
    # Check if inside phantom
    if not point_in_phantom(pos):
        # Find entry point
        # t_z = (0 - z)/dz if moving forward
        t = float('inf')
        for boundary, normal_sign in [(-15, -1), (15, 1), (-15, -1), (15, 1), (0, -1), (30, 1)]:
            if normal_sign == -1:  # lower boundary
                if dir_vec[0] != 0 and abs(x) <= 15:
                    pass  # need proper check
            pass
        return float('inf')
    
    # Inside phantom - find distance to exit
    t_min = float('inf')
    
    # x boundaries
    if dx > 0:
        t = (15 - x) / dx
        if t > 0:
            t_min = min(t_min, t)
    elif dx < 0:
        t = (-15 - x) / dx
        if t > 0:
            t_min = min(t_min, t)
    
    # y boundaries
    if dy > 0:
        t = (15 - y) / dy
        if t > 0:
            t_min = min(t_min, t)
    elif dy < 0:
        t = (-15 - y) / dy
        if t > 0:
            t_min = min(t_min, t)
    
    # z boundaries
    if dz > 0:
        t = (30 - z) / dz
        if t > 0:
            t_min = min(t_min, t)
    elif dz < 0:
        t = (0 - z) / dz
        if t > 0:
            t_min = min(t_min, t)
    
    return t_min

#=============================================================================
# Scoring volume
#=============================================================================

# Cylinder: centered at (0, 0, 10), radius 1 cm, axis along z, thickness 0.2 cm
# So z from 9.9 to 10.1, radius <= 1.0

def in_scoring_volume(pos):
    """Check if point is inside the scoring volume cylinder"""
    x, y, z = pos
    r = math.sqrt(x*x + y*y)
    return r <= 1.0 and 9.9 <= z <= 10.1

#=============================================================================
# Monte Carlo simulation
#=============================================================================

def run_simulation(num_events_to_score=200, max_histories=500000):
    """Run simplified Monte Carlo to score photon interactions in scoring volume"""
    
    scored_events = []
    
    history_count = 0
    while len(scored_events) < num_events_to_score and history_count < max_histories:
        history_count += 1
        
        # Start a primary photon
        pos = np.array([0.0, 0.0, -100.0])
        dir_vec = sample_beam_direction()
        E = 2.0  # MeV
        photon_active = True
        
        # Track the photon through the phantom
        while photon_active:
            # Current position check
            # Does the photon go into the phantom?
            if not point_in_phantom(pos):
                # Check if it will enter the phantom
                # Distance to phantom surface along direction
                if dir_vec[2] <= 0:
                    # Moving away from phantom or sideways
                    # Check if already past
                    if pos[2] < 0 and dir_vec[2] > 0:
                        # Moving toward phantom
                        t_surf = -pos[2] / dir_vec[2]
                        pos = pos + dir_vec * t_surf
                        if not point_in_phantom(pos):
                            photon_active = False
                            break
                    else:
                        photon_active = False
                        break
            
            # Now inside phantom (or at surface)
            # Compute distance to next interaction
            mu = mu_total(E)
            if mu <= 0:
                photon_active = False
                break
            
            d = sample_interaction_length(mu)
            
            # Distance to phantom boundary
            d_exit = distance_to_phantom_boundary(pos, dir_vec)
            
            if d > d_exit:
                # Photon leaves phantom
                pos = pos + dir_vec * d_exit
                photon_active = False
                break
            
            # Move to interaction point
            pos = pos + dir_vec * d
            
            # Check if interaction point is in scoring volume
            if in_scoring_volume(pos):
                # Sample interaction process
                process = sample_interaction_process(E)
                
                if process == 'compton':
                    costheta = sample_compton_angle(E)
                    E_out, cos_e, t_e = compton_kinematics(E, costheta)
                    
                    event = {
                        'process': 'compton',
                        'e_in_MeV': E,
                        'e_out_MeV': E_out,
                        'cos_scatter': costheta,
                        't_e_MeV': t_e,
                        'cos_electron': cos_e,
                        't_positron_MeV': None
                    }
                    scored_events.append(event)
                    
                    # Continue tracking scattered photon if it's still in phantom
                    # Update energy and direction
                    # Compton scatter direction
                    # For the scattered photon, we need its new direction
                    # The scattering angle is θ (cosθ = costheta)
                    # We need to rotate the direction vector
                    # This is complex - for now, let's simplify
                    # For the purpose of this simulation, after a scatter we can just
                    # continue tracking the scattered photon
                    E = E_out
                    # The new direction is rotated by angle θ
                    # We need to randomly sample the azimuthal angle φ
                    phi = random.uniform(0, 2*math.pi)
                    # Rotate direction vector by θ around an axis perpendicular to dir_vec
                    # This is a standard 3D rotation
                    costheta_s = costheta
                    sintheta_s = math.sqrt(1 - costheta_s**2)
                    
                    # Find a perpendicular vector
                    # Take cross product with z-axis if not parallel
                    if abs(dir_vec[2]) < 0.999:
                        perp = np.cross(dir_vec, np.array([0, 0, 1]))
                    else:
                        perp = np.cross(dir_vec, np.array([1, 0, 0]))
                    perp = perp / np.linalg.norm(perp)
                    
                    # Another perpendicular vector
                    perp2 = np.cross(dir_vec, perp)
                    perp2 = perp2 / np.linalg.norm(perp2)
                    
                    # New direction
                    new_dir = (costheta_s * dir_vec + 
                               sintheta_s * (math.cos(phi) * perp + math.sin(phi) * perp2))
                    dir_vec = new_dir / np.linalg.norm(new_dir)
                    
                elif process == 'photoelectric':
                    event = {
                        'process': 'photoelectric',
                        'e_in_MeV': E,
                        'e_out_MeV': None,
                        'cos_scatter': None,
                        't_e_MeV': E,  # all energy to electron (minus binding energy, negligible for water at 2 MeV)
                        'cos_electron': 1.0,  # approximately forward for high energy
                        't_positron_MeV': None
                    }
                    scored_events.append(event)
                    photon_active = False  # photon absorbed
                    
                elif process == 'pair':
                    t_e, t_pos, cos_e, cos_pos = sample_pair_energies(E)
                    event = {
                        'process': 'pair',
                        'e_in_MeV': E,
                        'e_out_MeV': None,
                        'cos_scatter': None,
                        't_e_MeV': t_e,
                        'cos_electron': cos_e,
                        't_positron_MeV': t_pos
                    }
                    scored_events.append(event)
                    photon_active = False  # photon converted to e-/e+
                    
                elif process == 'rayleigh':
                    # Coherent scattering - no energy loss
                    costheta = rayleigh_angle(E)
                    # Rotate direction
                    phi = random.uniform(0, 2*math.pi)
                    costheta_s = costheta
                    sintheta_s = math.sqrt(max(0, 1 - costheta_s**2))
                    
                    if abs(dir_vec[2]) < 0.999:
                        perp = np.cross(dir_vec, np.array([0, 0, 1]))
                    else:
                        perp = np.cross(dir_vec, np.array([1, 0, 0]))
                    perp = perp / np.linalg.norm(perp)
                    perp2 = np.cross(dir_vec, perp)
                    perp2 = perp2 / np.linalg.norm(perp2)
                    
                    new_dir = (costheta_s * dir_vec + 
                               sintheta_s * (math.cos(phi) * perp + math.sin(phi) * perp2))
                    dir_vec = new_dir / np.linalg.norm(new_dir)
                    
                    event = {
                        'process': 'rayleigh',
                        'e_in_MeV': E,
                        'e_out_MeV': E,
                        'cos_scatter': costheta,
                        't_e_MeV': 0.0,
                        'cos_electron': None,
                        't_positron_MeV': None
                    }
                    scored_events.append(event)
                    # Continue tracking same photon (same energy, new direction)
                
            else:
                # Interaction happened but not in scoring volume
                # Sample process to determine what happens
                process = sample_interaction_process(E)
                
                if process == 'compton':
                    costheta = sample_compton_angle(E)
                    E_out, cos_e, t_e = compton_kinematics(E, costheta)
                    # Update energy and direction
                    E = E_out
                    phi = random.uniform(0, 2*math.pi)
                    costheta_s = costheta
                    sintheta_s = math.sqrt(max(0, 1 - costheta_s**2))
                    
                    if abs(dir_vec[2]) < 0.999:
                        perp = np.cross(dir_vec, np.array([0, 0, 1]))
                    else:
                        perp = np.cross(dir_vec, np.array([1, 0, 0]))
                    perp = perp / np.linalg.norm(perp)
                    perp2 = np.cross(dir_vec, perp)
                    perp2 = perp2 / np.linalg.norm(perp2)
                    
                    new_dir = (costheta_s * dir_vec + 
                               sintheta_s * (math.cos(phi) * perp + math.sin(phi) * perp2))
                    dir_vec = new_dir / np.linalg.norm(new_dir)
                    
                elif process == 'photoelectric':
                    photon_active = False
                elif process == 'pair':
                    photon_active = False
                elif process == 'rayleigh':
                    costheta = rayleigh_angle(E)
                    phi = random.uniform(0, 2*math.pi)
                    costheta_s = costheta
                    sintheta_s = math.sqrt(max(0, 1 - costheta_s**2))
                    
                    if abs(dir_vec[2]) < 0.999:
                        perp = np.cross(dir_vec, np.array([0, 0, 1]))
                    else:
                        perp = np.cross(dir_vec, np.array([1, 0, 0]))
                    perp = perp / np.linalg.norm(perp)
                    perp2 = np.cross(dir_vec, perp)
                    perp2 = perp2 / np.linalg.norm(perp2)
                    
                    new_dir = (costheta_s * dir_vec + 
                               sintheta_s * (math.cos(phi) * perp + math.sin(phi) * perp2))
                    dir_vec = new_dir / np.linalg.norm(new_dir)
                    # E unchanged
        
        if len(scored_events) % 50 == 0 and len(scored_events) > 0:
            print(f"Scored {len(scored_events)} events after {history_count} histories...")
    
    return scored_events

# Run simulation
print("Starting simulation...")
events = run_simulation(200, max_histories=500000)
print(f"Scored {len(events)} events")

# Print summary
processes = [e['process'] for e in events]
for p in ['compton', 'photoelectric', 'pair', 'rayleigh']:
    count = processes.count(p)
    print(f"  {p}: {count} ({count/len(events)*100:.1f}%)")

# Check energies
e_in = [e['e_in_MeV'] for e in events]
print(f"E_in range: {min(e_in):.3f} to {max(e_in):.3f} MeV")