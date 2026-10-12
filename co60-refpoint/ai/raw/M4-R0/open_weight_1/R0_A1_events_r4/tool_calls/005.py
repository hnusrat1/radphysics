import numpy as np
import math
import random

# Constants
mec2 = 0.511  # Mev
r0 = 2.8179403262e-13  # cm
n_e_per_cm3 = 3.3456e23  # electrons/cm^3

# Geometry
source_z = -100.0  # cm
phantom_z_min = 0.0
phantom_z_max = 30.0
phantom_half = 15.0  # half-side in x and y

# Scoring volume (cylinder on axis)
scoring_radius = 1.0  # cm
scoring_z_center = 5.0  # cm
scoring_z_half = 0.1  # cm, so thickness 0.2 cm

# Field size at phantom surface
feld_half = 5.0  # cm (10x10 cm^2)

# Cross sections functions (from previous)

def mu_compto(E):
    if E <= 0:
        return 0.0
    a = E / mec2
    a2 = a*a
    x = 1.0 + 2.0*a
    lnx = math.log(x)
    term1 = (1.0+a)/a2 * (2.0*(1.0+a)/x - lnx/a)
    term2 = lnx/(2.0*a) - (1.0+3.0*a)/(x*x)
    sigma = 2.0*math.pi*r0*r0 * (term1 + term2)
    return sigma * n_e_per_cm3

def mu_pair(E):
    if E <= 2*mec2:
        return 0.0
    e_vals = np.array([1.1732, 1.3325, 2.0, 3.0, 5.0, 10.0])
    mu_vals = np.array([0.006205, 0.006286, 0.0082, 0.0109, 0.0126, 0.0120])
    if E <= 1.1732:
        t = E - 2*mec2
        if t <= 0:
            return 0.0
        t0 = 1.1732 - 1.022
        mu0 = 0.006205
        return mu0 * (t/t0)**2 * 1.0
    elif E >= 10.0:
        return 0.0120
    else:
        logE = math.log(E)
        logE_vals = np.log(e_vals)
        logmu_vals = np.log(mu_vals)
        return math.exp(np.interp(logE, logE_vals, logmu_vals)) * 1.0

def mu_pe(E):
    if E <= 0:
        return 0.0
    e_vals = np.array([0.01, 0.02, 0.05, 0.1, 0.2, 0.3, 0.5, 1.0, 1.1732, 1.3325, 2.0])
    mu_vals = np.array([5.33, 0.742, 0.0321, 0.0164, 0.0097, 0.0057, 0.0026, 0.0011, 0.001735, 0.001524, 0.00078])
    if E <= 0.01:
        return mu_vals[0] * (E/0.01)**(-3.5) * 1.0
    elif E >= 2.0:
        return 0.00078 * (E/2.0)**(-3.0) * 1.0
    else:
        logE = math.log(E)
        logE_vals = np.log(e_vals)
        logmu_vals = np.log(mu_vals)
        return math.exp(np.interp(logE, logE_vals, logmu_vals)) * 1.0

def mu_rayleigh(E):
    if E <= 0:
        return 0.0
    e_vals = np.array([0.01, 0.02, 0.05, 0.1, 0.2, 0.5, 1.0, 1.1732, 1.3325, 2.0, 5.0])
    mu_vals = np.array([0.0008, 0.0010, 0.0012, 0.0016, 0.0021, 0.0024, 0.0025, 0.00253, 0.00213, 0.0012, 0.0003])
    if E <= 0.01:
        return 0.0008 * (E/0.01)**0.3 * 1.0
    elif E >= 5.0:
        return 0.0003 * (E/5.0)**(-1.5) * 1.0
    else:
        logE = math.log(E)
        logE_vals = np.log(e_vals)
        logmu_vals = np.log(mu_vals)
        return math.exp(np.interp(logE, logE_vals, logmu_vals)) * 1.0

def mu_total(E):
    return mu_compton(E) + mu_pair(E) + mu_pe(E) + mu_rayleigh(E)

# Test
print("mu_total(1.1732) =", mu_total(1.1732)
print("mu_total(1.3325) =", mu_total(1.3325)

# Sample direction from source to field
def sample_direction():
    # Uniform in field at z=0
    x0 = random.uniform(-feld_half, feld_half)
    y0 = random.uniform(-feld_half, feld_half)
    dx = x0 - 0.0
    dy = y0 - 0.0
    dz = 0.0 - source_z  # = 100
    norm = math.sqrt(dx*dx + dy*dy + dz*dz)
    return dx/norm, dy/norm, dz/norm

# Check if point is inside phantom
def in_phantom(x, y, z):
    return (phantom_z_min <= z <= phantom_z_max and 
            -phantom_half <= x <= phantom_half and 
            -phantom_half <= y <= phantom_half)

# Check if point is in scoring volume
def in_scoring(x, y, z):
    r = math.sqrt(x*x + y*y)
    return (r <= scording_radius and 
            abs(z - scoring_z_center) <= scoring_z_half)

# Compton scattering: sample outgoing photon direction and energy
# Use the Klein-Nishina differential cross section
# Sampling using the Kahn rejection method
def sample_compton(E_in):
    """Return (E_out, cos_theta, t_e, cos_electron)"""
    # E_out is the scattered photon energy
    # cos_theta is cosine of scattering angle
    # t_e = E_in - E_out (electron kinetic energy)
    # cos_electron: angle of electron relative to incident direction
    # Electron direction from conservation of momentum
    
    a = E_in / mec2
    # Kahn's method
    while True:
        # Sample mu uniformly in [-1, 1]
        mu = random.uniform(-1.0, 1.0)
        # Samuel energy ratio
        E_ratio = 1.0 / (1.0 + a * (1.0 - mu))
        # Rejection
        xi = random.random()
        # Acceptance probability proportional to (E_ratio^2 + 1/E_ratio + mu^2 - 1)/2
        # Actually, the Klein-Nishina cross section can be sampled using:
        # dσ/dΩ ∝ (E'/E)^2 * (E/E' + E'/E - sin^2θ)
        # = (E_ratio^2) * (1/E_ratio + E_ratio + (mu^2 - 1)/2?) 
        # Let's use the standard formula:
        # f(mu) = E_ratio^2 * (E_ratio + 1/E_ratio + mu^2 - 1)/2? No.
        # The correct expression:
        # dσ/dcosθ = π r0^2 * (E'/E)^2 * [E/E' + E/E - (1 - cos^2θ)]
        # = π r0^2 * (E'/E)^2 * (E/E' + E'/E + cos^2θ - 1)
        # Let's sample using the Kahn method:
        # 1. Sample mu uniformly in [-1,1]
        # 2. Compute E_ratio = 1/(1+a*(1-mu))
        # 3. If random() <= (E_ratio^2 + 1/E_ratio - (1-mu^2)/2? Hmm.
        # Actually, the Kahn rejection method:
        # f(mu) = (1 + a*(1-mu))^{-2} + (1 + a*(1-mu))^{-1} + (mu^2 - 1)/2? No.
        # Let's just use the standard method:
        # The differential cross section per unit cosθ:
        # dσ/dcosθ = π r0^2 * (E'/E)^2 * (E'/E + E/E' - sin^2θ)
        # = π r0^2 * k^2 * (k + 1/k - (1-mu^2))
        # where k = E'/E = 1/(1+a*(1-mu))
        # So f = k^2 * (k + 1/k + mu^2 - 1)
        # = k^3 + k + k^2*(mu^2 -1))
        # Maximum of f is at mu=1? Let's check.
        # At mu=1: k=1, f=1+1+0=2
        # At mu=-1: k=1/(1+2a), small, f ~ k^3 + k + k^2*0 ~ small
        # So maximum is ~2 at mu=1.
        # But for rejection, we need an upper bound.
        # Actually, the Kahn method uses:
        # f_max = 2 (at mu=1)
        # But we need the full algorithm.
        
        # Let's use a simpler approach: sample using the inverse transform.
        k = 1.0 / (1.0 + a * (1.0 - mu))
        f = k*k * (k + 1.0/k + mu*mu - 1.0)
        # Normalized? Actually, we can compare to the maximum.
        # For a > 0, the maximum of f is at mu=1 where k=1, f=2.
        # But this is not normalized.
        # The standard rejection: sample mu uniformly, accept with prob = f(mu)/f_max
        # where f_max = 2.
        if xi <= f / 2.0:
            break
    
    E_out = k * E_in
    t_e = E_in - E_out
    
    # Electron direction
    # From Compton kinematics:
    # cot(phi_e) = (1 + a) * tan(theta/2)
    # where phi_e is the electron angle relative to incident direction
    cos_theta = mu
    theta = math.acos(cos_theta)
    # Electron scattering angle: tan(phi_e) = 1/((1+a)*tan(theta/2))
    # Actually: cot(phi_e) = (1 + a) * tan(theta/2)
    # So tan(phi_e) = 1 / ((1+a) * tan(theta/2)
    if abs(theta) < 1e-10:
        cos_electon = 1.0
    else:
        tan_half = math.tan(theta/2)
        cot_phie = (1.0 + a) * tan_half
        phie = math.atan2(1.0, cot_phie)  # since cot = adjacent/opposite? Actually cot = cos/sin, so phi = arctan(1/cot)
        # Actually, phi_e = arctan(1 / ((1+a)*tan(theta/2)))
        cos_electron = math.cos(phie)
    
    return E_out, cos_theta, t_e, cos_electron

# Rayleigh scattering: small angle, no energy loss
def sample_rayliegh(E_in):
    """Return (E_out, cos_theta)"""
    # Rayleigh is coherent, no energy loss
    # The angular distribution is given by the form factor F(q).
    # For water, we approximate using the Thomson cross section with form factor.
    # At these energies, Rayleigh is strongly forward peaked.
    # We'll sample using a simple approximation: Gaussian-like forward peak.
    # The typical scattering angle for Rayleigh is ~lambda / (atomic radius)
    # lambda = hc/E, hc = 1240 eV*nm = 1.24e-9 MeV*m = 1.24e-7 MeV*cm
    # For 1 MeV, lambda ~ 1.24e-7 cm
    # Atomic radius ~ 1e-8 cm, so theta ~ 0.01 rad
    # We'll use a simple model: theta sampled from a distribution that peaks at 0.
    # Use the form factor for water: approximate as exp(-q^2 * R^2/3) where R ~ 2.5e-8 cm
    # Actually, let's just use a simple approximation.
    
    # For simplicity, sample mu from a distribution biased towards forward.
    # We'll use: mu = 1 - 0.5e-5 * (-log(rndom))^2? No.
    # Let's just use Thomson scattering (uniform cos) for simplicity? No, Rayleigh is forward.
    
    # Use the Rayleigh cross section differential:
    # dσ/dΩ = (dσ/dΩ)_Thomson * F(q)^2
    # where F(q) ≈ Z * exp(-q^2 * R^2/6) for one atom, but for water it's complex.
    #
    # Let's approximate: sample theta from pdf ~ exp(-theta^2/(2*sigma^2)) * sin(theta)
    # with sigma ~ 0.01 rad
    # Use the Molière approximation for Rayleigh scattering angle.
    sigma = 0.01  # rad
    while True:
        theta = random.gauss(0, sigma)
        if theta >= 0 and theta <= math.pi:
            break
    cos_theta = math.cos(theta)
    
    return E_in, cos_theta  # no energy loss

# Pair production sampling
def sample_pair(E_in):
    """Return (t_e, cos_electron, t_positron, cos_positron)"""
    # Available energy: E_avail = E_in - 2*mec2
    # The positron and electron share the energy.
    # At just above threshold, the distribution is approximately uniform.
    # We'll sample uniformly.
    E_avail = E_in - 2*mec2
    # Electron kinetic energy (can be any fraction)
    t_e = random.uniform(0, E_avail)
    t_pos = E_avail - t_e
    
    # Angular distribution: forward peaked.
    # Approximate: both electron and positron have angles ~ mec2/E_in
    # For simplicity, sample cos from uniform[cos_max, 1] where cos_max ~ 1 - 2*(mec2/E_in)
    # Actually, for pair production near threshold, the electron and positron are emitted 
    # with angles roughly ~ mec2/E_in
    sigma_angle = mec2 / E_in  # rad
    # Sample electron angle
    while True:
        theta_e = random.gauss(0, sigma_angle)
        if 0 <= theta_e <= math.pi/2:
            break
    cos_electron = math.cos(theta_e)
    
    # Positron angle: approximately opposite direction to conserve momentum? 
    # Actually, in pair production, the electron and positron share the momentum,
    # so they are emitted roughly in the forward direction with small angular spread.
    # For simplicity, give the positron the same angle.
    # Actually, the angles can be different but we'll approximate both forward.
    # Let's just use the same angle for simplicity.
    cos_positon = cos_electron  # approximate
    
    return t_e, cos_electron, t_pos, cos_positon

# Photoelectric sampling
def sample_photelectric(E_in):
    """Return (t_e, cos_electron)"""
    # Binding energy negligible at these energies
    t_e = E_in  # kinetic energy of electron
    # Angular distribution: forward peaked at high energies
    # Using the Sauter distribution or simply forward approximation.
    # For simplicity, sample from forward direction with small spread.
    sigma_angle = 0.1  # rad, rough
    while True:
        theta_e = random.gauss(0, sigma_angle)
        if 0 <= theta_e <= math.pi/2:
            break
    cos_electron = math.cos(theta_e)
    return t_e, cos_electron

# Now the main simulation
def simulate_photn():
    """Simulate one photon history, return list of events in scoring volume"""
    events = []
    
    # Sample initial energy
    if random.random() < 0.5:
        E = 1.1732
    else:
        E = 1.3325
    
    # Sample initial direction
    dx, dy, dz = sample_direction()
    
    # Starting position: source
    x, y, z = 0.0, 0.0, source_z
    
    # Track this photon
    while True:
        # Current total cross section
        mu_t = mu_total(E)
        if mu_t <= 0:
            break
        
        # Sample path length
        s = -math.log(random.random()) / mu_t
        
        # Move
        x_new = x + s * dx
        y_new = y + s * dy
        z_new = z + s * dz
        
        # Check if interaction is inside phantom
        if not in_phantom(x_new, y_new, z_new):
            # Photon leaves the phantom
            break
        
        # Update position
        x, y, z = x_new, y_new, z_new
        
        # Determine interaction type
        mu_c = mu_compton(E)
        mu_p = mu_pair(E)
        mu_pe = mu_pe(E)
        mu_r = mu_rayleigh(E)
        mu_t = mu_c + mu_p + mu_pe + mu_r
        
        r0 = random.random() * mu_t
        if r0 < mu_c:
            process = "compton"
            E_out, cos_scatter, t_e, cos_electron = sample_compton(E)
            # Record if in scoring volume
            if in_scoring(x, y, z):
                events.append({
                    "process": process,
                    "e_in_MeV": E,
                    "e_out_MeV": E_out,
                    "cos_scatter": cos_scatter,
                    "t_e_MeV": t_e,
                    "cos_electron": cos_electron,
                    "t_positron_MeV": None
                })
            # Continue tracking scattered photon
            E = E_out
            # Update direction: the photon is scattered
            # We need to rotate the direction vector
            # Sampling the azimuthal angle phi uniformly
            phi = random.uniform(0, 2*math.pi)
            cos_theta = cos_scatter
            sin_theta = math.sqrt(1 - cos_theta*cos_theta)
            # Original direction (dx, dy, dz)
            # We need to rotate by theta in the direction defined by phi
            # Build a new direction
            # Find a vector perpendicular to (dx, dy, dz)
            if abs(dx) < 0.9:
                perp_x = 1.0
                perp_y = 0.0
                perp_z = 0.0
            else:
                perp_x = 0.0
                perp_y = 1.0
                perp_z = 0.0
            # Make it perpendicular
            dot = perp_x*dx + perp_y*dy + perp_z*dz
            perp_x -= dot * dx
            perp_y -= dot * dy
            perp_z -= dot * dz
            norm = math.sqrt(perp_x**2 + perp_y**2 + perp_z**2)
            if norm > 0:
                perp_x /= norm
                perp_y /= norm
                perp_z /= norm
            
            # Second perpendicular vector
            perp2_x = dy*perp_z - dz*perp_y
            perp2_y = dz*perp_x - dx*perp_z
            perp2_z = dx*perp_y - dy*perp_x
            
            # New direction
            dx_new = cos_theta * dx + sin_theta * (math.cos(phi) * perp_x + math.sin(phi) * perp2_x)
            dy_new = cos_theta * dy + sin_theta * (math.cos(phi) * perp_y + math.sin(phi) * perp2_y)
            dz_new = cos_theta * dz + sin_theta * (math.cos(phi) * perp_z + math.sin(phi) * perp2_z)
            
            # Normalize
            norm = math.sqrt(dx_new**2 + dy_new**2 + dz_new**2)
            if norm > 0:
                dx, dy, dz = dx_new/norm, dy_new/norm, dz_new/norm
            else:
                break  # something wrong
            
        elif r0 < mu_c + mu_p:
            process = "pair"
            t_e, cos_electron, t_pos, cos_positron = sample_pair(E)
            if in_scoring(x, y, z):
                events.append({
                    "process": process,
                    "e_in_MeV": E,
                    "e_out_MeV": None,
                    "cos_scatter": None,
                    "t_e_MeV": t_e,
                    "cos_electron": cos_electron,
                    "t_positron_MeV": t_pos
                })
            # Photon is absorbed; stop tracking
            break
            
        elif r0 < mu_c + mu_p + mu_pe:
            process = "photelectric"
            t_e, cos_electron = sample_photelectric(E)
            if in_scoring(x, y, z):
                events.append({
                    "process": process,
                    "e_in_MeV": E,
                    "e_out_MeV": None,
                    "cos_scatter": None,
                    "t_e_MeV": t_e,
                    "cos_electron": cos_electron,
                    "t_positron_MeV": None
                })
            # Photon absorbed
            break
            
        else:
            process = "rayleigh"
            E_out, cos_scatter = sample_rayleigh(E)
            if in_scoring(x, y, z):
                events.append({
                    "process": process,
                    "e_in_MeV": E,
                    "e_out_MeV": E_out,
                    "cos_scatter": cos_scatter,
                    "t_e_MeV": 0.0,
                    "cos_electron": None,
                    "t_positron_MeV": None
                })
            # Continue tracking scattered photon (same energy)
            # Update direction
            phi = random.uniform(0, 2*math.pi)
            cos_theta = cos_scatter
            sin_theta = math.sqrt(1 - cos_theta*cos_theta)
            # Rotate direction
            if abs(dx) < 0.9:
                perp_x = 1.0
                perp_y = 0.0
                perp_z = 0.0
            else:
                perp_x = 0.0
                perp_y = 1.0
                perp_z = 0.0
            dot = perp_x*dx + perp_y*dy + perp_z*dz
            perp_x -= dot * dx
            perp_y -= dot * dy
            perp_z -= dot * dz
            norm = math.sqrt(perp_x**2 + perp_y**2 + perp_z**2)
            if norm > 0:
                perp_x /= norm
                perp_y /= norm
                perp_z /= norm
            perp2_x = dy*perp_z - dz*perp_y
            perp2_y = dz*perp_x - dx*perp_z
            perp2_z = dx*perp_y - dy*perp_x
            
            dx_new = cos_theta * dx + sin_theta * (math.cos(phi) * perp_x + math.sin(phi) * perp2_x)
            dy_new = cos_theta * dy + sin_theta * (math.cos(phi) * perp_y + math.sin(phi) * perp2_y)
            dz_new = cos_theta * dz + sin_theta * (math.cos(phi) * perp_z + math.sin(phi) * perp2_z)
            
            norm = math.sqrt(dx_new**2 + dy_new**2 + dz_new**2)
            if norm > 0:
                dx, dy, dz = dx_new/norm, dy_new/norm, dz_new/norm
            else:
                break
            # E unchanged
        
        # Limit iterations to avoid infinite loops
        if len(events) > 5:  # not needed, just safety
            pass
    
    return events

# Run many histories
def run_simulation(num_target=200, max_histories=200000):
    all_events = []
    history_count = 0
    while len(all_events) < num_target and history_count < max_histories:
        history_count += 1
        if history_count % 10000 == 0:
            print(f"  History {history_count}, collected {len(all_events)} events")
        events = simulate_photon()
        all_events.extend(events)
    print(f"Total histories: {history_count}, events: {len(all_events)}")
    return all_events[:num_target]

# Run
print("Starting simulation...")