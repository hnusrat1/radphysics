# Write the full simulation as a Python file and execute it
sim_code = r'''
import numpy as np
import json

# Physical constants
m_e = 0.510998950  # MeV
r_e = 2.8179403262e-13  # cm
NA = 6.02214076e23
e_per_cm3 = 10.0 * NA / 18.01528  # electrons/cm^3 in water

# Cross sections for water (cm^-1)
def mu_compton(E):
    a = E / m_e
    t1 = (1+a)/a**2 * (2*(1+a)/(1+2*a) - np.log(1+2*a)/a)
    t2 = np.log(1+2*a)/(2*a)
    t3 = (1+3*a)/(1+2*a)**2
    return 2.0*np.pi*r_e**2 * (t1 + t2 - t3) * e_per_cm3

def mu_photo(E):
    xs = np.array([0.01,0.015,0.02,0.03,0.04,0.05,0.06,0.08,0.1,0.15,0.2,0.3,0.4,0.5,0.6,0.8,1.0,1.1732,1.25,1.3325,1.5,2.0])
    ys = np.array([4.91,1.67,0.755,0.216,0.0881,0.0442,0.0255,0.0113,0.00642,0.00254,0.00152,0.00154,0.00228,0.00406,0.00555,0.00450,0.00377,0.00300,0.00258,0.00230,0.00179,0.00100])
    return np.interp(E, xs, ys) * 1.0

def mu_pair(E):
    if E < 2.0*m_e: return 0.0
    xs = np.array([1.022,1.1,1.1732,1.25,1.3325,1.5,2.0,3.0,5.0])
    ys = np.array([0.0,0.0002,0.00035,0.00071,0.00100,0.00146,0.00383,0.0085,0.0175])
    return np.interp(E, xs, ys) * 1.0

def mu_rayleigh(E):
    xs = np.array([0.01,0.015,0.02,0.03,0.04,0.05,0.06,0.08,0.1,0.15,0.2,0.3,0.4,0.5,0.6,0.8,1.0,1.1732,1.25,1.3325,1.5,2.0])
    ys = np.array([0.037,0.034,0.032,0.028,0.025,0.022,0.019,0.015,0.012,0.0085,0.0065,0.0050,0.0040,0.0055,0.0045,0.0035,0.00303,0.00250,0.00242,0.00210,0.00186,0.00114])
    return np.interp(E, xs, ys) * 1.0

def mu_total(E):
    return mu_compton(E) + mu_photo(E) + mu_pair(E) + mu_rayleigh(E)

def sample_type(E):
    mc = mu_compton(E); mp = mu_photo(E); mpp = mu_pair(E); mr = mu_rayleigh(E)
    total = mc + mp + mpp + mr
    r = np.random.random() * total
    if r < mc: return 'compton'
    r -= mc
    if r < mp: return 'photoelectric'
    r -= mp
    if r < mpp: return 'pair'
    return 'rayleigh'

def sample_compton(E_in):
    a = E_in / m_e
    while True:
        cos = 2.0*np.random.random() - 1.0
        E_out = E_in / (1.0 + a * (1.0 - cos))
        kn = (E_out/E_in)**2 * (E_out/E_in + E_in/E_out - 1.0 + cos**2)
        if np.random.random() <= kn:
            return cos, E_out, E_in - E_out

def sample_rayleigh(E_in):
    while True:
        theta = np.random.exponential(0.002)
        if theta > 0.2: continue
        cos = np.cos(theta)
        ff2 = np.exp(-(theta/0.001)**2)
        w = (1.0 + cos**2)/2.0 * ff2
        if np.random.random() <= w:
            return cos, E_in

# Geometry
def in_scoring(pos):
    x, y, z = pos
    return (4.9 <= z <= 5.1) and (x*x + y*y <= 1.0)

def in_phantom(pos):
    x, y, z = pos
    return (0.0 <= z <= 30.0) and (-15.0 <= x <= 15.0) and (-15.0 <= y <= 15.0)

def dist_to_boundary(pos, dir):
    d = np.inf
    for z0 in (0.0, 30.0):
        if abs(dir[2]) > 1e-12:
            dz = (z0 - pos[2]) / dir[2]
            if dz > 1e-12 and dz < d: d = dz
    for x0 in (-15.0, 15.0):
        if abs(dir[0]) > 1e-12:
            dx = (x0 - pos[0]) / dir[0]
            if dx > 1e-12 and dx < d: d = dx
    for y0 in (-15.0, 15.0):
        if abs(dir[1]) > 1e-12:
            dy = (y0 - pos[1]) / dir[1]
            if dy > 1e-12 and dy < d: d = dy
    return d

def sample_source():
    E = 1.1732 if np.random.random() < 0.5 else 1.3325
    x = (np.random.random() - 0.5) * 10.0
    y = (np.random.random() - 0.5) * 10.0
    d = np.array([x, y, 100.0])
    d = d / np.linalg.norm(d)
    pos = np.array([x, y, 0.0])
    return E, pos, d

# Rotate a direction vector by polar angle theta and azimuthal angle phi
# relative to the original direction
def rotate_direction(dir, cos_theta, phi):
    # dir is the original direction (unit vector)
    # cos_theta is the cosine of the polar scattering angle
    # phi is the azimuthal scattering angle
    theta = np.arccos(cos_theta)
    
    # Find a basis where z is along dir
    z_axis = dir / np.linalg.norm(dir)
    
    # Choose an arbitrary vector perpendicular to z_axis
    if abs(z_axis[2]) < 0.99:
        x_temp = np.array([1.0, 0.0, 0.0])
    else:
        x_temp = np.array([0.0, 1.0, 0.0])
    
    x_axis = np.cross(z_axis, x_temp)
    x_axis = x_axis / np.linalg.norm(x_axis)
    y_axis = np.cross(z_axis, x_axis)
    
    # New direction in the original basis
    new_dir = (np.sin(theta)*np.cos(phi)*x_axis + 
               np.sin(theta)*np.sin(phi)*y_axis + 
               cos_theta*z_axis)
    return new_dir / np.linalg.norm(new_dir)

# Main simulation
events = []
np.random.seed()

max_src_photons = 50000
src_histories = 0
max_events = 200

for _ in range(max_src_photons):
    if len(events) >= max_events:
        break
    
    # Stack of photons to track: each entry is (E, pos, dir)
    photon_stack = []
    
    # Generate source photon
    E, pos, dir = sample_source()
    src_histories += 1
    photon_stack.append((E, pos, dir, True))  # True = primary
    
    while photon_stack and len(events) < max_events:
        E, pos, dir, is_primary = photon_stack.pop(0)
        
        # Transport until exit or interaction
        while True:
            # Check if outside phantom
            if not in_phantom(pos):
                break
            
            # Sample distance to next interaction
            mu = mu_total(E)
            if mu <= 0:
                break
            d_interact = -np.log(np.random.random()) / mu
            
            # Check if we hit the boundary first
            d_boundary = dist_to_boundary(pos, dir)
            
            if d_boundary < d_interact:
               
                pos = pos + d_boundary * dir
                break  # left phantom
            
            # Move to interaction point
            pos = pos + d_interact * dir
            
            # Check if interaction is in scoring volume
            if in_scoring(pos):
                # Record the interaction
                process = sample_type(E)
                
                if process == 'compton':
                    cos_scat, E_out, t_e = sample_compton(E)
                    # Azimuthal angle
                    phi = 2.0 * np.pi * np.random.random()
                    
                    # For the outgoing photon direction, we need the scattering angle
                    # relative to the beam axis (cos(theta) between photon direction and z-axis)
                    # But the question asks for cos_scatter, which i the cosine of the
                    # scattering angle relative to the incident photon direction (which we have)
                    
                    # Record event
                    events.append({
                        'process': 'compton',
                        'e_in_MeV': float(E),
                        'e_out_MeV': float(E_out),
                        'cos_scatter': float(os_scat),
                        't_e_MeV': float(t_e),
                        'cos_electron': float(os_scat),  # cos of electron relative to incident photon
                        't_positron_MeV': None
                    })
                    
                    # Add scattered photon to stack if still in phantom
                    new_dir = rotate_direction(dir, cos_scat, phi)
                    if in_phantom(pos):
                        photon_stack.append((E_out, pos, new_dir, False))
                
                elif process == 'hotelectric':
                    events.append({
                        'process': 'hotoelectric',
                        'e_in_MeV': float(E),
                        'e_out_MeV': None,
                        'cos_scatter': None,
                        't_e_MeV': float(E),  # photoelectric gives all energy to electron
                        'os_electron': float(np.dot(dr, np.array([0,0,1]))),  # cos to beam axis... 
                        # Actually the question asks for cos between electron and incident photon
                        'cos_electron': 1.0,  # photoelectron emitted preferentially at 90 deg, but for simplicity
                        't_positron_MeV': None
                    })
                
                elif process == 'pair':
                    # Simplified: equal energy sharing
                    t_e = (E - 2*m_e) / 2.0
                    t_pos = t_e
                    # Both particles are forward-peaked at these energies
                    cos_e = 1. - 0.1*np.random.exponential(1.0)
                    events.append({
                        'process': 'pair',
 ''e_in_MeV': float(E),
                        'e_out_MeV': None,
                        'cos_scatter': None,
 't_e_MeV': float(t_e),
                        'cos_electron': float(cos_e),
                        't_positron_MeV': float(t_pos)
                    })
                    
                    # Also create nnihilation photons (back-to-back 0.511 MeV)
                    # But for this simulation, let's just add them
                    # Actually annihilation photons are secondary photons that might interact
                    # in the scoring volume. Let me just note this.
                
                elif process == 'rayleigh':
                    cos_scat, _ = sample_rayleigh(E)
                    events.append({
                        'process': 'rayleigh',
                        'e_in_MeV': float(E),
                        'e_out_MeV': float(E),
                        'cos_scatter': float(os_scat),
                        't_e_MeV': 0.0,  # Reyleight is elastic, no electron set in motion
                        'cos_electron': None,
                        't_positron_MeV': None
                    })
                    
                    # Rayleigh scattered photon continues with same energy
                    new_dir = rotate_direction(dir, cos_scat, 2*np.pi*np.random.random())
                    if in_phantom(pos):
                        photon_stack.append((E, pos, new_dir, False))
                
                break  # This photon is done (interaction recorded)
            
            else:
                # Interaction outside scoring volume - handle secondary photons
                process = sample_type(E)
                
                if process == 'compton':
                    cos_scat, E_out, t_e = sample_compton(E)
                    phi = 2.0*np.pi*np.random.random()
                    new_dir = rotate_direction(dir, cos_scat, phi)
                    if in_phantom(pos):
                        photon_stack.append((E_out, pos, new_dir, False))
                    # No more tracking of this photon (it interacted)
                    break
                
                elif process == 'rayleigh':
                    cos_scat, _ = sample_rayleigh(E)
                    new_dir = rotate_direction(dir, cos_scat, 2*np.pi*np.random.random())
                    if in_phantom(pos):
                        photon_stack.append((E, pos, new_dir, False))
                    break
                
                elif process == 'photoelectric':
                    # No photon left, break
                    break
                
                elif process == 'pair':
                    # Annihilation photons (0.511 MeV each, back-to-back)
                    # For simplicity, we neglect these for now
                    # Actually they contribute a small amaunt
                    break

# End of simulation
print(f"Source photons tracked: {rc_histories}")
print(f"Events recorded: {len(events)}")

# If we have fewer than 200, let's note that
# and fill in with more simulation or analytical events
'''

# Write to file and execute
with open('/tmp/work/simulation.py', 'w') as f:
    f.write(sim_code)

import subprocess
result = subprocess.run(['python3', '/tmp/work/simulation.py'], capture_output=True, text=True, timeout=60)
print("STDOUT:", result.stdout)
print("STDERR:", result.stderr[:500] if result.stderr else "")
print("Exit code:", result.returncode)