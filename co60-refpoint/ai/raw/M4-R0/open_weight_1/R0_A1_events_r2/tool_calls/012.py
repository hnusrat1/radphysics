import numpy as np
import time

# Constants
r_e = 2.8179403227e-13  # cm
me = 0.5109989461  # MeV
NA = 6.02214076e23
M_H2O = 18.015  # g/mol
rho = 1.0  # g/cm3
N_electrons = 10

E1 = 1.1732
E2 = 1.3325

# Geometry bounds
PHANTOM_X_MIN, PHANTOM_X_MAX = -15, 15
PHANTOM_Y_MIN, PHANTOM_Y_MAX = -15, 15
PHANTOM_Z_MIN, PHANTOM_Z_MAX = 0.0, 30.0

# Scoring volume (cylinder on z-axis)
SV_RADIUS = 1.0
SV_Z_MIN = 4.9
SV_Z_MAX = 5.1

def sigma_total_approx(E):
    sig_comp = N_electrons * sigma_KN(E)
    sig_rayl = 0.0014 * M_H2O / NA * (1.0 / max(E, 0.01))**2
    sig_pe = 0.0016 * M_H2O / NA * (1.0 / max(E, 0.01))**3.5
    if E > 1.022:
        sig_pair = 0.00087 * M_H2O / NA * (E - 1.022) / (1.25 - 1.022)
    else:
        sig_pair = 0.0
    total = sig_comp + sig_rayl + sig_pe + sig_pair
    return total, sig_comp, sig_rayl, sig_pe, sig_pair

def sigma_KN(E):
    k = E / me
    term = ((1+k)/k**2 * (2*(1+k)/(1+2*k) - np.log(1+2*k)/k) 
            + np.log(1+2*k)/(2*k) 
            - (1+3*k)/(1+2*k)**2)
    return 2*np.pi * r_e**2 * term

def sample_compton_angle(E_in):
    alpha = E_in / me
    max_dcs = 2.0
    while True:
        mu = np.random.uniform(-1, 1)
        f = 1.0 / (1.0 + alpha*(1.0 - mu))
        dcs = f**2 * (f + 1.0/f - (1.0 - mu**2))
        if np.random.random() < dcs / max_dcs:
            return mu

def compton_electron_angle(E_in, cos_theta):
    sin_theta = np.sqrt(1 - cos_theta**2)
    tan_half = sin_theta / (1 + cos_theta)
    alpha = E_in / me
    cot_phi = (1 + alpha) * tan_half
    if cot_phi == 0:
        return 0.0
    tan_phi = 1.0 / cot_phi
    cos_phi = 1.0 / np.sqrt(1.0 + tan_phi**2)
    return cos_phi

def sample_rayleigh_angle():
    return np.random.uniform(0.999, 1.0)

def sample_photoelectron_angle(E_kin):
    E_total = E_kin + me
    beta = np.sqrt(1 - (me/E_total)**2)
    # Use a simple forward approximation
    # At high energies, the Sauter distribution is very forward.
    # For simplicity, sample cosθ from distribution proportional to sin^2θ / (1-βcosθ)^4
    max_val = 10.0
    while True:
        mu = np.random.uniform(-1, 1)
        val = (1 - mu**2) / (1 - beta*mu)**4
        if np.random.random() < val / max_val:
            return mu

def sample_pair(K_available):
    t_e = np.random.uniform(0, K_available)
    t_p = K_available - t_e
    # Simple forward angles
    cos_e = np.random.uniform(0.98, 1.0)
    cos_p = np.random.uniform(0.98, 1.0)
    return t_e, t_p, cos_e, cos_p

def rotate_direction(u, cos_theta, phi):
    """Rotate direction vector u by polar angle theta (cos_theta) and azimuthal angle phi.
    Returns new direction vector."""
    sin_theta = np.sqrt(1 - cos_theta**2)
    # Find a perpendicular vector
    if abs(u[2]) < 0.999:
        n = np.array([0.0, 0.0, 1.0])
    else:
        n = np.array([1.0, 0.0, 0.0])
    u_perp1 = np.cross(u, n)
    norm = np.linalg.norm(u_perp1)
    if norm > 0:
        u_perp1 /= norm
    else:
        # u is parallel to n, use another
        u_perp1 = np.array([1.0, 0.0, 0.0]) if abs(u[0]) < 0.999 else np.array([0.0, 1.0, 0.0])
        u_perp1 = np.cross(u, u_perp1)
        u_perp1 /= np.linalg.norm(u_perp1)
    u_perp2 = np.cross(u, u_perp1)
    v = u * cos_theta + u_perp1 * sin_theta * np.cos(phi) + u_perp2 * sin_theta * np.sin(phi)
    return v

def is_in_scoring_volume(pos):
    x, y, z = pos
    return (x**2 + y**2 <= SV_RADIUS**2) and (SV_Z_MIN <= z <= SV_Z_MAX)

def is_in_phantom(pos):
    x, y, z = pos
    return (PHANTOM_X_MIN <= x <= PHANTOM_X_MAX and 
            PHANTOM_Y_MIN <= y <= PHANTOM_Y_MAX and 
            PHANTOM_Z_MIN <= z <= PHANTOM_Z_MAX)

def simulate_one_photon(record_events=True, events_list=None):
    """Simulate one source photon and return list of interaction events in scoring volume.
    If record_events is True and events_list is provided, append events to it."""
    
    # Sample initial energy
    E = E1 if np.random.random() < 0.5 else E2
    
    # Sample initial direction (uniform in pyramid)
    x0 = np.random.uniform(-5, 5)
    y0 = np.random.uniform(-5, 5)
    dir = np.array([x0, y0, 100.0])
    dir = dir / np.linalg.norm(dir)
    
    # Starting position at source
    pos = np.array([0.0, 0.0, -100.0])
    
    # Find entry point into phantom
    # Solve for t such that z=0: pos[2] + t*dir[2] = 0 => t = -pos[2]/dir[2]
    if dir[2] <= 0:
        # Should not happen for forward direction, but just in case
        return
    t_entry = -pos[2] / dir[2]
    entry_point = pos + t_entry * dir
    
    # Check if entry point is within phantom xy bounds; if not, photon misses phantom
    if not (PHANTOM_X_MIN <= entry_point[0] <= PHANTOM_X_MAX and 
            PHANTOM_Y_MIN <= entry_point[1] <= PHANTOM_Y_MAX):
        return
    
    # Start tracking from entry point
    pos = entry_point.copy()
    
    # Main transport loop
    while True:
        # If photon left the phantom, stop
        if not is_in_phantom(pos):
            return
        
        # If energy is too low, stop (unlikely)
        if E < 0.001:
            return
        
        # Get cross sections at current energy
        sigma_tot, sig_comp, sig_rayl, sig_pe, sig_pair = sigma_total_approx(E)
        
        # Linear attenuation coefficient
        mu = sigma_tot * NA / M_H2O * rho  # cm^-1
        
        # Sample distance to next interaction
        s = -np.log(np.random.random()) / mu
        
        # Move photon
        new_pos = pos + s * dir
        
        # Check if new position is outside phantom; if so, move to boundary and stop
        if not isln_phanom(new_pos):
            # Find intersection with phantom boundary along this segment
            # For simplicity, if the step ends outside, terminate
            return
        
        # Check if interaction occurs in scoring volume
        in_scoring = is_in_scoring_volume(new_pos)
        
        # Sample interaction process
        xi = np.random.random() * sigma_tot
        if xi < sig_comp:
            process = 'compton'
        elif xi < sig_comp + sig_rayl:
            process = 'rayleigh'
        elif xi < sig_comp + sig_rayl + sig_pe:
            process = 'photoelectric'
        else:
            process = 'pair'
        
        # Record event if in scoring volume
        if in_scoring and record_events:
            event = {}
            event['process'] = process
            event['e_in_MeV'] = E
            
            if process == 'compton':
                cos_scatter = sample_compton_angle(E)
                E_out = E / (1 + (E/me)*(1 - cos_scatter))
                t_e = E - E_out
                cos_e = compton_electron_angle(E, cos_scatter)
                event['e_out_MeV'] = E_out
                event['cos_scatter'] = cos_scatter
                event['t_e_MeV'] = t_e
                event['cos_electron'] = cos_e
                event['t_positron_MeV'] = None
                
                # Update photon state for continued tracking
                E = E_out
                # Scattering angle in 3D
                phi = np.random.uniform(0, 2*np.pi)
                new_dir = rotate_direction(dir, cos_scatter, phi)
                dir = new_dir
                pos = new_pos
                # Continue tracking scattered photon
                
            elif process == 'rayleigh':
                cos_scatter = sample_rayleigh_angle()
                E_out = E  # no energy change
                event['e_out_Mv'] = E_out
                event['cos_scatter'] = cos_scatter
                event['t_e_MeV'] = 0.0
                event['cos_electron'] = None
                event['t_positron_MeV'] = None
                
                # Update direction
                phi = np.random.uniform(0, 2*np.pi)
                new_dir = rotate_direction(dir, cos_scatter, phi)
                dir = new_dir
                pos = new_pos
                # Continue tracking
                
            elif prcess == 'photoelectric':
                E_out = None
                cos_scatter = None
                t_e = E  # kinetic energy of photoelectron (neglect binding)
                cos_e = sample_photoelectron_angle(t_e)
                event['e_out_MeV'] = None
                event['cos_scatter'] = None
                event['t_e_MeV'] = t_e
                event['cos_electron'] = cos_e
                event['t_positron_MeV'] = None
                
                # Photon absorbed, terminate
                events_list.append(event)
                return
                
            elif process == 'pair':
                E_out = None
                cos_scatter = None
                K_avail = E - 2*me  # kinetic energy available
                t_e, t_p, cos_e, cos_p = sample_pair(K_avail)
                event['e_out_MeV'] = None
                event['cos_scatter'] = None
                event['t_e_MeV'] = t_e
                event['cos_electron'] = cos_e
                event['t_positron_MeV'] = t_p
                # Note: we don't store cos for positron separately; the field 'cos_electron' is for the electron
                # The schema only has cos_electron, so we store the electron's angle.
                # However the problem asks for cos_electron (for the electron). For pair, we have both e- and e+.
                # We'll store the electron's cos in cos_electron, and positron energy in t_positron_MeV.
                
                events_list.append(event)
                return  # Photon absorbed
            
            # Add event to list if in scoring volume
            if in_scoring:
                events_list. append(event)
            
            # For Compton and Rayleigh, continue tracking after updating pos and dir
            # (already done above)
            continue
            
        else:
            # Interaction not in scoring volume, but we still need to update state
            if process == 'compton':
                cos_scatter = sample_compton_angle(E)
                E_out = E / (1 + (E/me)*(1 - cos_scatter))
                # Update photon
                E = E_out
                phi = np.random.uniform(0, 2*np.pi)
                dir = rotate_direction(dir, cos_scatter, phi)
                pos = new_pos
                continue
            elif process == 'rayleigh':
                cos_scatter = sample_rayleigh_angle()
                phi = np.random.uniform(0, 2*np.pi)
                dir = rotate_direction(dir, cos_scatter, phi)
                pos = new_pos
                continue
            else:
                # photoelectric or pair - photon absorbed
                return

# Note: This function has many syntax errors. Let me rewrite it properly.