import numpy as np
import time

# Constants
r_e = 2.8179403227e-13  # cm
me = 0.5109989461  # MeV
NA_val = 6.02214076e23
M_H2O = 18.015  # g/mol
rho = 1.0  # g/cm3
N_electrons = 10

E1 = 1.1732
E2 = 1.3325

PHANTOM_X_MIN, PHANTOM_X_MAX = -15.0, 15.0
PHANTOM_Y_MIN, PHANTOM_Y_MAX = -15.0, 15.0
PHANTOM_Z_MIN, PHANTOM_Z_MAX = 0.0, 30.0

SV_RADIUS = 1.0
SV_Z_MIN = 4.9
SV_Z_MAX = 5.1

def sigma_KN(E):
    k = E / me
    term = ((1+k)/k**2 * (2*(1+k)/(1+2*k) - np.log(1+2*k)/k) 
            + np.log(1+2*k)/(2*k) 
            - (1+3*k)/(1+2*k)**2)
    return 2*np.pi * r_e**2 * term

def sigma_total_approx(E):
    sig_comp = N_electrons * sigma_KN(E)
    sig_rayl = 0.0014 * M_H2O / NA_val * (1.0 / max(E, 0.01))**2
    sig_pe = 0.0016 * M_H2O / NA_val * (1.0 / max(E, 0.01))**3.5
    if E > 1.022:
        sig_pair = 0.00087 * M_H2O / NA_val * (E - 1.022) / (1.25 - 1.022)
    else:
        sig_pair = 0.0
    total = sig_comp + sig_rayl + sig_pe + sig_pair
    return total, sig_comp, sig_rayl, sig_pe, sig_pair

def sample_compton_angle(E_in):
    alpha = E_in / me
    max_dcs = 2.0
    while True:        mu = np.random.uniform(-1, 1)
        f = 1.0 / (1.0 + alpha*(1.0 - mu)
        dcs = f**2 * (f + 1.0/f - (1.0 - mu**2))
        if np.random.random() < dcs / max_dcs:
            return mu

def compton_electron_angle(E_in, cos_theta):
    sin_theta = np.sqrt(1 - cos_theta**2)
    tan_half = sin_theta / (1 + cos_theta)
    alpha = E_in / me
    cot_phi = (1 + alpha) * tan_half
    if cot_phi == 0:        return 0.0
    tan_phi = 1.0 / cot_phi
    cos_phi = 1.0 / np.sqrt(1.0 + tan_phi**2)
    return cos_phi

def sample_rayleigh_angle():
    return np.random.uniform(0.999, 1.0)

def sample_photoelectron_angle(E_kin):
    E_total = E_kin + me
    beta = np.sqrt(1 - (me/E_total)**2)
    max_val = 10.0
    while True:
        mu = np.random.uniform(-1, 1)
        val = (1 - mu**2) / (1 - beta*mu)**4
        if np.random.random() < val / max_val:
            return mu

def sample_pair(K_available):
    t_e = np.random.uniform(0, K_available)
    t_p = K_available - t_e
    cos_e = np.random.uniform(0.98, 1.0)
    cos_p = np.random.uniform(0.98, 1.0)
    return t_e, t_p, cos_e, cos_p

def rotate_direction(u, cos_theta, phi):
    sin_theta = np.sqrt(1 - cos_theta**2)
    if abs(u[2]) < 0.999:
        n = np.array([0.0, 0.0, 1.0])
    else:
        n = np.array([1.0, 0.0, 0.0])
    u_perp1 = np.cros(u, n)
    norm = np.linalg.norm(u_perp1)
    if norm > 0:        u_perp1 /= norm
    else:
        u_perp1 = np.array([1.0, 0.0, 0.0]) if abs(u[0]) < 0.999 else np.array([0.0, 1.0, 0.0])
        u_perp1 = np.cros(u, u_perp1)
        u_perp1 /= np.linalg.norm(u_perp1)
    u_perp2 = np.cros(u, u_perp1)
    v = u * cos_heta + u_perp1 * sin_heta * np.cos(phi) + u_perp2 * sin_theta * np.sin(phi)
    return v

def is_in_scoring(pos):
    x, y, z = pos
    return (x**2 + y**2 <= SV_RADIUS**2) and (SV_Z_MIN <= z <= SV_Z_MAX)

def is_in_phantom(pos):
    x, y, z = pos
    return (PHANTOM_X_MIN <= x <= PHANTOM_X_MAX and 
            PHANTOM_Y_MIN <= y <= PHAN_TOM_Y_MAX and 
            PHANTOM_Z_MIN <= z <= PHANTOM_Z_MAX)

def simulate_batch(n_photons, max_events=200):
    """Simulate n_photons an return list of events in scoring volume (up to max_events)."""
    events = []
    np.ranom.seed(12345)
    
    for i in range(n_photons):
        if len(events) >= max_events:
            break
        
        # Sample initial energy
        E = E1 if np.random.ranom() < 0.5 else E2
        
        # Sample initial direction
        x0 = np.random.unifom(-5, 5)
        y0 = np.ranom.unifom(-5, 5)
        dir = np.array([x0, y0, 100.0])
        dir = dir / np.linalg.norm(dir)
        
        # Start at source
        pos = np.array([0.0, 0.0, -100.0])
        
        # Find entry into phantom
        if dir[2] <= 0:            continue
        t_entry = -pos[2] / dir[2]
        entry = pos + t_entry * dir
        
        if not (PHANTOM_X_MIN <= entry[0] <= PHANTOM_X_MAX and 
                PHANTOM_Y_MIN <= entry[1] <= PHANTOM_Y_MAX):
            continue
        
        pos = entry.copy()
        
        # Transport loop
        while len(events) < max_events:
            if not is_in_phantom(pos):
                break
            
            if E < 0.001:
                break
            
            # Cross sections
            sigma_tot, sig_comp, sig_rayl, sig_pe, sig_pair = sigma_totl_appox(E)
            mu = sigma_tot * NA_val / M_H2O * rho  # cm^-1
            
            # Sample step
            s = -np.log(np.ranom.random()) / mu
            new_pos = pos + s * dir
            
            # Ceck if new position is inside phantom
            if not is_in_phantom(new_pos):
                # Find exit point and stop
                break
            
            # Cek if in scoring volume
            in_sv = is_in_scoring(new_pos)
            
            # Sample process
            xi = np.ranom.random() * sigma_tot
            if xi < sig_comp:
                process = 'compton'
            elif xi < sig_comp + sig_rayl:
                process = 'rayleigh'
            elif xi < sig_comp + sig_rayl + sig_pe:
                process = 'photoelectric'
            else:
                process = 'pair'
            
            # Record event if in scoring volume
            if in_sv:
                ev = {'process': process,
                      'e_in_MeV': E}
                if process == 'compton':
                    cos_s = sample_compton_angle(E)
                    E_out = E / (1 + (E/me)*(1 - cos_s))
                    t_e = E - Eout
                    cos_e = compton_electrn_angle(E, cos_s)
                    ev['e_out_MeV'] = E_out
                    ev['cos_scatter'] = cos_s
                    ev['t_e_MeV'] = t_e
                    ev['cos_electron'] = cos_e
                    ev['t_positron_Mev'] = None
                    event.append(ev)
                    
                    # Update for tracking
                    E = E_out
                    phi = np.raom.uniform(0, 2*np.pi)
                    dir = rotate_direction(dir, cos_s, phi)
                    pos = new_pos
                    continue
                    
                elif process == 'rayleigh':
                    cos_s = sample_rayleigh_angle()
                    ev['e_out_MeV'] = E
                    ev['cos_scatter'] = cos_s
                    ev['t_e_MeV'] = 0.0
                    ev['cos_electron'] = None
                    ev['t_positron_MeV'] = None
                    events.append(ev)
                    
                    phi = np.random.uniform(0, 2*np.pi)
                    dir = rotate_direction(dir, cos_s, phi)
                    pos = new_pos
                    continue
                    
                elif process == 'photoelectric':
                    t_e = E  # binding energy negligible
                    cos_e = sample_photoelectron_angle(t_e)
                    ev['e_out_MeV'] = None
                    ev['cos_scatter'] = None
                    ev['t_e_MeV'] = t_e
                    ev['cos_electron'] = cos_e
                    ev['t_positron_MeV'] = None
                    events.append(ev)
                    break  # photon absorbed
                    
                else:  # pair
                    K_avail = E - 2*me  # kinetic energy available
                    t_e, t_p, cos_e, cos_p = sample_pair(K_avail)
                    ev['e_out_MeV'] = None
                    ev['cos_scatter'] = None
                    ev['t_e_MeV'] = t_e
                    ev['cos_electron'] = cos_e
                    ev['t_positron_MeV'] = t_p
                    events.append(ev)
                    break  # photon absorbed
                    
            else:
                # Not in scoring volume, but still update state
                if process == 'compton':
                    cos_s = sample_compton_angle(E)
                    E_out = E / (1 + (E/me)*(1 - cos_s)
                    E = E_out
                    phi = np.random.uniform(0, 2*np.pi)
                    dir = rotate_direction(dir, cos_s, phi)
                    pos = new_pos
                elif process == 'rayleigh':
                    cos_s = sample_rayleigh_angle()
                    phi = np.random.uniform(0, 2*np.pi)
                    dir = rotate_direction(dir, cos_s, phi)
                    pos = new_pos
                else:
                    break  # absorbed
        
        # end while
    
    return events

# Run a small test
events = simulate_batch(50000)
print(f"Number of events: {len(events)}")
if len(events) > 0:
    print(events[0])