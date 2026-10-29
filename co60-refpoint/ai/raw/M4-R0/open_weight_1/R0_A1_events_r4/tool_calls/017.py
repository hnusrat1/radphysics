import numpy as np
import math
import random

# Constants
mec2 = 0.511
r0 = 2.8179403262e-13
nepm3 = 3.3456e23

# Geometry
source_z = -100.0
pz_min = 0.0
pz_max = 30.0
phalf = 15.0
sc_r = 1.0
sc_z0 = 5.0
sc_dz = 0.1
field_half = 5.0

# Cross sections
def mu_inc(E):
    if E <= 0: return 0.0
    a = E/mec2
    a2 = a*a
    x = 1 + 2*a
    lx = math.log(x)
    t1 = (1+a)/a2 * (2*(1+a)/x - lx/a)
    t2 = lx/(2*a) - (1+3*a)/(x*x)
    return 2*math.pi*r0*r0*(t1+t2)*nepm3

def mu_pp(E):
    if E <= 2*mec2: return 0.0
    e_arr = np.array([1.1732, 1.3325, 2.0, 3.0, 5.0, 10.0])
    mu_arr = np.array([0.006205, 0.006286, 0.0082, 0.0109, 0.0126, 0.0120])
    if E <= 1.1732:
        t = E - 2*mec2
        if t <= 0: return 0.0
        t0 = 1.1732 - 1.022
        return 0.006205 * (t/t0)**2
    elif E >= 10.0:
        return 0.0120
    else:
        le = math.log(E)
        le_arr = np.log(e_arr)
        lmu_arr = np.log(mu_arr)
        return math.exp(np.interp(le, le_arr, lmu_arr))

def mu_pe(E):
    if E <= 0: return 0.0
    e_arr = np.array([0.01, 0.02, 0.05, 0.1, 0.2, 0.3, 0.5, 1.0, 1.1732, 1.3325, 2.0])
    mu_arr = np.array([5.33, 0.742, 0.0321, 0.0164, 0.0097, 0.0057, 0.0026, 0.0011, 0.001735, 0.001524, 0.00078])
    if E <= 0.01:
        return 5.33 * (E/0.01)**(-3.5)
    elif E >= 2.0:
        return 0.00078 * (E/2.0)**(-3.0)
    else:
        le = math.log(E)
        le_arr = np.log(e_arr)
        lmu_arr = np.log(mu_arr)
        return math.exp(np.interp(le, le_arr, lmu_arr))

def mu_ray(E):
    if E <= 0: return 0.0
    e_arr = np.array([0.01, 0.02, 0.05, 0.1, 0.2, 0.5, 1.0, 1.1732, 1.3325, 2.0, 5.0])
    mu_arr = np.array([0.0008, 0.0010, 0.0012, 0.0016, 0.0021, 0.0024, 0.0025, 0.00253, 0.00213, 0.0012, 0.0003])
    if E <= 0.01:
        return 0.0008 * (E/0.01)**0.3
    elif E >= 5.0:
        return 0.0003 * (E/5.0)**(-1.5)
    else:
        le = math.log(E)
        le_arr = np.log(e_arr)
        lmu_arr = np.log(mu_arr)
        return math.exp(np.interp(le, le_arr, lmu_arr))

def mu_tot(E):
    return mu_inc(E) + mu_pp(E) + mu_pe(E) + mu_ray(E)

# Geometry checks
def in_phantom(x, y, z):
    return (pz_min <= z <= pz_max and 
            -phalf <= x <= phalf and 
            -phalf <= y <= phalf)

def in_scoring(x, y, z):
    return (x*x + y*y <= sc_r*sc_r and 
            abs(z - sc_z0) <= sc_dz)

def sample_dir():
    x0 = random.uniform(-field_half, field_half)
    y0 = random.uniform(-field_half, field_half)
    dx = x0
    dy = y0
    dz = 100.0
    norm = math.sqrt(dx*dx + dy*dy + dz*dz)
    return dx/norm, dy/norm, dz/norm

# Sampling routines
def sample_compton(E):
    a = E/mec2
    while True:
        mu = random.uniform(-1.0, 1.0)
        k = 1.0/(1.0 + a*(1.0 - mu))
        f = k*k*(k + 1.0/k + mu*mu - 1.0)
        if random.random() <= f/2.0:
            break
    Eout = k * E
    te = E - Eout
    theta = math.acos(mu)
    if abs(theta) < 1e-10:
        cose = 1.0
    else:
        tan_half = math.tan(theta/2.0)
        cot_phi = (1.0 + a) * tan_half
        phi = math.atan2(1.0, cot_phi)
        cose = math.cos(phi)
    return Eout, mu, te, cose

def sample_rayleigh(E):
    sigma = 0.008
    while True:
        theta = random.gauss(0, sigma)
        if 0 <= theta <= math.pi:
            break
    mu = math.cos(theta)
    return E, mu

def sample_pair(E):
    Eavail = E - 2*mec2
    te = random.uniform(0, Eavail)
    tp = Eavail - te
    sigma = mec2/E
    while True:
        theta = random.gauss(0, sigma)
        if 0 <= theta <= math.pi/2:
            break
    cose = math.cos(theta)
    cosp = cose  # approximate
    return te, cose, tp, cosp

def sample_pe(E):
    te = E  # binding energy negligible
    sigma = 0.3
    while True:
        theta = random.gauss(0, sigma)
        if 0 <= theta <= math.pi/2:
            break
    cose = math.cos(theta)
    return te, cose

def rotate_dir(dx, dy, dz, cos_theta, phi):
    sin_theta = math.sqrt(1 - cos_theta*cos_theta)
    # Find perpendicular
    if abs(dx) < 0.9:
        px, py, pz = 1.0, 0.0, 0.0
    else:
        px, py, pz = 0.0, 1.0, 0.0
    dot = px*dx + py*dy + pz*dz
    px -= dot*dx
    py -= dot*dy
    pz -= dot*dz
    n = math.sqrt(px*px + py*py + pz*pz)
    if n > 0:
        px /= n; py /= n; pz /= n
    # Second perpendicular
    qx = dy*pz - dz*py
    qy = dz*px - dx*pz
    qz = dx*py - dy*px
    # New direction
    dxn = cos_theta*dx + sin_theta*(math.cos(phi)*px + math.sin(phi)*qx)
    dyn = cos_theta*dy + sin_theta*(math.cos(phi)*py + math.sin(phi)*qy)
    dzn = cos_theta*dz + sin_theta*(math.cos(phi)*pz + math.sin(phi)*qz)
    nn = math.sqrt(dxn*dxn + dyn*dyn + dzn*dzn)
    if nn > 0:
        return dxn/nn, dyn/nn, dzn/nn
    return dx, dy, dz

# Main simulation
def simulate_history():
    events = []
    
    # Sample source photon
    if random.random() < 0.5:
        E = 1.1732
    else:
        E = 1.3325
    dx, dy, dz = sample_dir()
    
    # Transport to phantom surface
    dist_to_surface = (pz_min - source_z) / dz  # dz > 0 for forward direction
    x = dx * dist_to_surface
    y = dy * dist_to_surface
    z = pz_min
    
    # Now track through phantom
    max_steps = 200
    for _ in range(max_steps):
        mu_t = mu_tot(E)
        if mu_t <= 0:
            break
        
        # Sample free path
        s = -math.log(random.random()) / mu_t
        
        # New position
        xn = x + s*dx
        yn = y + s*dy
        zn = z + s*dz
        
        # Check if interaction is inside phantom
        if not in_phantom(xn, yn, zn):
            # Photon exits phantom
            break
        
        # Interaction occurs
        x, y, z = xn, yn, zn
        
        # Determine process
        mu_c = mu_inc(E)
        mu_p = mu_pp(E)
        mu_pe = mu_pe(E)
        mu_r = mu_ray(E)
        mu_t = mu_c + mu_p + mu_pe + mu_r
        
        r = random.random() * mu_t
        
        if r < mu_c:
            # Compton
            Eout, cos_scat, te, cose = sample_compton(E)
            in_sv = in_scoring(x, y, z)
            
            if in_sv:
                events.append({
                    "process": "compton",
                    "e_in_MeV": E,
                    "e_out_MeV": Eout,
                    "cos_scatter": cos_scat,
                    "t_e_MeV": te,
                    "cos_electron": cose,
                    "t_positron_MeV": None
                })
            
            # Continue with scattered photon
            E = Eout
            phi = random.uniform(0, 2*math.pi)
            dx, dy, dz = rotate_dir(dx, dy, dz, cos_scat, phi)
            
        elif r < mu_c + mu_p:
            # Pair production
            te, cose, tp, cosp = sample_pair(E)
            in_sv = in_scoring(x, y, z)
            
            if in_sv:
                events.append({
                    "process": "pair",
                    "e_in_MeV": E,
                    "e_out_MeV": None,
                    "cos_scatter": None,
                    "t_e_MeV": te,
                    "cos_electron": cose,
                    "t_positron_MeV": tp
                })
            # Photon absorbed
            break
            
        elif r < mu_c + mu_p + mu_pe:
            # Photoelectric
            te, cose = sample_pe(E)
            in_sv = in_scoring(x, y, z)
            
            if in_sv:
                events.append({
                    "process": "photoelectric",
                    "e_in_MeV": E,
                    "e_out_MeV": None,
                    "cos_scatter": None,
                    "t_e_MeV": te,
                    "cos_electron": cose,
                    "t_positron_MeV": None
                })
            # Photon absorbed
            break
            
        else:
            # Rayleigh
            Eout, cos_scat = sample_rayleigh(E)
            in_sv = in_scoring(x, y, z)
            
            if in_sv:
                events.append({
                    "process": "rayleigh",
                    "e_in_MeV": E,
                    "e_out_MeV": Eout,
                    "cos_scatter": cos_scat,
                    "t_e_MeV": 0.0,
                    "cos_electron": None,
                    "t_positron_MeV": None
                })
            
            # Continue (same energy)
            phi = random.uniform(0, 2*math.pi)
            dx, dy, dz = rotate_dir(dx, dy, dz, cos_scat, phi)
    
    return events

# Run simulation
print("Starting simulation...")
random.seed(12345)
all_events = []
n_hist = 0
target = 200
max_hist = 2000000

while len(all_events) < target and n_hist < max_hist:
    n_hist+= 1
    if nhist % 100000 == 0:
        print(f"  History {nhist}, events collected {len(all_events)}")
    evts = simulate_history()
    all_events.extend(evts)

print(f"Done. Total histories: {nhist}, events: {len(all_events)}")
events_final = all_events[:200]
print(f"Events: {len(events_final)}")

# Summary
from collections import Counter
print(Counter([e["process"] for e in events_final]))