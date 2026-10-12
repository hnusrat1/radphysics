import numpy as np
import math
import time

# ===== CONSTANTS & CROSS SECTIONS (same as before) =====
mec2 = 0.511
r0 = 2.8179403262e-13
r0_sq = r0**2
NA = 6.02214076e23
M_H2O = 18.01528
rho = 1.0
electrons_per_gram = 10 * NA / M_H2O
F_COMP = 0.95

def sigma_kn(E):
    if E <= 0: return 0
    alpha = E / mec2
    return (2*np.pi*r0_sq * ((1+alpha)/alpha**2 * (2*(1+alpha)/(1+2*alpha) - np.log(1+2*alpha)/alpha) 
                            + np.log(1+2*alpha)/(2*alpha) - (1+3*alpha)/(1+2*alpha)**2))

def mu_comp_rho(E):
    return F_COMP * sigma_kn(E) * electrons_per_gram

def mu_pe_rho(E):
    E = max(float(E), 0.01); E = min(E, 5.0)
    tE = np.array([0.01,0.015,0.02,0.03,0.04,0.05,0.06,0.08,0.1,0.15,0.2,0.3,0.4,0.5,0.6,0.8,1.0,1.25,1.5,2.0,3.0,5.0])
    tM = np.array([4.79,1.41,0.519,0.169,0.0770,0.0261,0.0137,0.00480,0.00213,0.000388,0.000111,0.0000235,0.00000881,0.00000412,0.00000212,0.000000838,0.000000443,0.000000220,0.000000123,0.0000000581,0.0000000191,0.00000000392])
    return float(np.exp(np.interp(np.log(E), np.log(tE), np.log(tM))))

def mu_pair_rho(E):
    if E <= 1.022: return 0.0
    E = min(E, 10.0)
    tE = np.array([1.022,1.25,1.5,2.0,2.5,3.0,4.0,5.0,6.0,8.0,10.0])
    tM = np.array([0.0,0.000639,0.00151,0.00318,0.00496,0.00720,0.0124,0.0169,0.0211,0.0286,0.0340])
    return float(np.exp(np.interp(np.log(E), np.log(tE), np.log(np.maximum(tM,1e-12)))))

def mu_ray_rho(E):
    E = max(float(E), 0.01); E = min(E, 5.0)
    tE = np.array([0.01,0.015,0.02,0.03,0.05,0.06,0.08,0.1,0.15,0.2,0.3,0.4,0.5,0.6,0.8,1.0,1.25,1.5,2.0,3.0,5.0])
    tM = np.array([0.000404,0.000882,0.00154,0.00349,0.00616,0.00716,0.00722,0.00682,0.00574,0.00467,0.00331,0.00250,0.00199,0.00162,0.00116,0.000873,0.000651,0.000502,0.000326,0.000151,0.0000650])
    return float(np.exp(np.interp(np.log(E), np.log(tE), np.log(tM))))

def mu_total(E):
    return (mu_comp_rho(E) + mu_pe_rho(E) + mu_pair_rho(E) + mu_ray_rho(E)) * rho

# Compton sampling with KN
def sample_compton_params(E_inc, n=1):
    """Sample Compton scattering. Returns arrays (cos_theta, E_scattered, cos_electron)"""
    alpha = E_inc / mec2
    ct_grid = np.linspace(-1, 1, 4001)
    k = 1.0 / (1.0 + alpha*(1.0 - ct_grid))
    dsigma = 2*np.pi * (r0_sq/2.0) * k**2 * (k + 1/k - (1 - ct_grid**2))
    cdf = np.cumsum(dsigma)
    cdf = cdf / cdf[-1]
    u = np.random.rand(n)
    ct = np.interp(u, cdf, ct_grid)
    E_sc = E_inc / (1.0 + alpha*(1.0 - ct))
    # Electron direction: phi is azimuthal (random),
    # polar angle theta_e in the scattering plane:
    # tan(theta_e) = sin(theta_photon) / (E/E_sc - cos(theta_photon))
    # cos(theta_e) = (E - E_sc*ct) / sqrt(E^2 + E_sc^2 - 2*E*E_sc*ct)
    cos_e = (E_inc - E_sc*ct) / np.sqrt(E_inc**2 + E_sc**2 - 2*E_inc*E_sc*ct)
    return ct, E_sc, cos_e

def sample_compton_one(E_inc):
    ct, E_sc, cos_e = sample_compton_params(E_inc, 1)
    return ct[0], E_sc[0], cos_e[0]

# Rayleigh sampling - small angles, approximate
def sample_rayleigh(E_inc):
    """Sample Rayleigh scattering angle. Returns cos_theta.
    Uses Thomson differential cross section × form factor.
    For 2 MeV, scattering is strongly forward."""
    # Use a simple model: sample form factor squared
    # F(q) ~ Z * exp(-q^2 * a^2 / 4) where a ~ atomic radius
    # For simplicity, use the Thompson differential cross section
    # with a forward peak.
    # We'll use a von Mises-like or rejection sampling.
    # Simplify: angular distribution ~ (1+cos^2θ) × exp(-q²a²)
    # For 2 MeV, the characteristic angle is very small (~1/(ka) ~ 0.001 rad)
    # So sample from a very narrow forward peak
    
    # For practical purposes, sample a small angle from a Gaussian
    # A characteristic for 2 MeV in water: angle ~ 1/(2*E*sin(theta/2)*a)...
    # Use an approximate: Gaussian with sigma = 1/(E*Z^(1/3)/... )
    # For simplicity, use sigma_angle = 2e-3 rad (forward peak)
    theta = np.random.normal(0, 0.002)  # ~0.1 degrees
    # Rayleigh photon keeps its energy
    return math.cos(theta)

# Rayleigh: coherent, energy unchanged. Just deflects by small angle.

# ===== GEOMETRY =====
SSD = 100.0
depth_poi = 10.0
field_half = 5.0
phantom_half = 15.0
phantom_depth = 30.0

cyl_radius = 1.0
cyl_half_z = 0.1
cyl_z_min = depth_poi - cyl_half_z
cyl_z_max = depth_poi + cyl_half_z
cyl_volume = np.pi * cyl_radius**2 * 2*cyl_half_z

def is_in_cylinder(x, y, z):
    return (cyl_z_min <= z <= cyl_z_max) and (x*x + y*y <= cyl_radius*cyl_radius)

def ray_cylinder_track(p, u):
    """Track length through cylinder. p: position, u: unit direction."""
    # z faces
    tz_in, tz_out = None, None
    if abs(u[2]) > 1e-12:
        t1 = (cyl_z_min - p[2]) / u[2]
        t2 = (cyl_z_max - p[2]) / u[2]
        tz_in, tz_out = min(t1,t2), max(t1,t2)
        if tz_out < 0: return 0.0
        if tz_in < 0: tz_in = 0.0
    # side
    a = u[0]**2 + u[1]**2
    b = 2*(p[0]*u[0] + p[1]*u[1])
    c = p[0]**2 + p[1]**2 - cyl_radius**2
    ts_in, ts_out = None, None
    if a > 1e-12:
        disc = b*b - 4*a*c
        if disc >= 0:
            sd = math.sqrt(disc)
            t1 = (-b - sd)/(2*a)
            t2 = (-b + sd)/(2*a)
            if t2 < 0: return 0.0
            ts_in = max(0.0, t1)
            ts_out = t2
    
    # Determine entry/exit candidates
    enters = []
    exits = []
    
    if tz_in is not None:
        xp = p[0] + tz_in*u[0]
        yp = p[1] + tz_in*u[1]
        if xp*xp + yp*yp <= cyl_radius*cyl_radius + 1e-8:
            enters.append(tz_in)
    if ts_in is not None:
        zp = p[2] + ts_in*u[2]
        if cyl_z_min <= zp <= cyl_z_max:
            enters.append(ts_in)
    
    if tz_out is not None:
        xp = p[0] + tz_out*u[0]
        yp = p[1] + tz_out*u[1]
        if xp*xp + yp*yp <= cyl_radius*cyl_radius + 1e-8:
            exits.append(tz_out)
    if ts_out is not None:
        zp = p[2] + ts_out*u[2]
        if cyl_z_min <= zp <= cyl_z_max:
            exits.append(ts_out)
    
    if not enters:
        # Starting inside?
        if is_in_cylinder(p[0], p[1], p[2]):
            t_entry = 0.0
        else:
            return 0.0
    else:
        t_entry = min(enters)
    
    t_exit = min(exits) if exits else 1e9
    if t_exit < t_entry:
        return 0.0
    return max(0.0, t_exit - t_entry)

# ===== SCORING STRUCTURE =====
photon_energy_edges = np.array([0, 0.05, 0.1, 0.15, 0.2, 0.25, 0.3, 0.35, 0.4, 0.45, 0.5, 0.55, 0.6, 0.65, 0.7, 0.75, 0.8, 0.85, 0.9, 0.95, 1, 1.05, 1.1, 1.15, 1.2, 1.25, 1.3, 1.35, 1.4])
electron_energy_edges = np.array([0.01, 0.05, 0.1, 0.15, 0.2, 0.25, 0.3, 0.35, 0.4, 0.45, 0.5, 0.55, 0.6, 0.65, 0.7, 0.75, 0.8, 0.85, 0.9, 0.95, 1, 1.05, 1.1, 1.15, 1.2, 1.4])
ct_edges = np.array([-1, -0.8, -0.6, -0.4, -0.2, 0, 0.2, 0.4, 0.6, 0.8, 1])

# Inner loop: transport a photon
def transport_photon(pos, dirn, energy, is_primary, scoring, depth_limit=phantom_depth):
    """Transport a photon through the phantom and score fluence at the cylinder.
    Returns created electrons (list of (pos, energy, cos_theta, parent_primary))."""
    electrons_created = []
    # pos: numpy array (3,), dirn: numpy array (3,) unit vector, energy in MeV
    # is_primary: True if this photon has not interacted yet
    # scoring: dict with keys:
    #   ph_prim_fluence, ph_scat_fluence, ph_prim_spec, ph_scat_spec, ph_prim_ang, ph_scat_ang
    
    p = pos.copy()
    u = dirn.copy()
    E = energy
    primary = is_primary
    
    # First, move from source to phantom surface
    # (already done by caller in this version)
    
    while True:
        # Score current segment from p in direction u
        L = ray_cylinder_track(p, u)
        if L > 1e-12:
            if primary:
                scoring['ph_prim_fluence'] += L
                idx = np.searchsorted(photon_energy_edges, E, side='right') - 1
                if 0 <= idx < 28:
                    scoring['ph_prim_spec'][idx] += L
                idx_a = np.searchsorted(ct_edges, u[2], side='right') - 1
                if 0 <= idx_a < 10:
                    scoring['ph_prim_ang'][idx_a] += L
            else:
                scoring['ph_scat_fluence'] += L
                idx = np.searchsorted(photon_energy_edges, E, side='right') - 1
                if 0 <= idx < 28:
                    scoring['ph_scat_spec'][idx] += L
                idx_a = np.searchsorted(ct_edges, u[2], side='right') - 1
                if 0 <= idx_a < 10:
                    scoring['ph_scat_ang'][idx_a] += L
            # Also track the total track length for normalization
            scoring['total_ph_track'] += L
        
        # Sample distance to next interaction
        if E <= 1e-4:
            break
        mu = mu_total(E)
        if mu <= 0:
            break
        s = -math.log(np.random.rand()) / mu
        new_p = p + s*u
        
        # Check if exited phantom
        if new_p[2] < 0 or new_p[2] > phantom_depth:
            break
        if abs(new_p[0]) > phantom_half or abs(new_p[1]) > phantom_half:
            break
        
        p = new_p
        # Check if interaction is in the scoring cylinder → record Layer 3 electrons
        in_cyl = is_in_cylinder(p[0], p[1], p[2])
        
        # Determine interaction type
        mc = mu_comp_rho(E)*rho
        mpe = mu_pe_rho(E)*rho
        mpair = mu_pair_rho(E)*rho
        mray = mu_ray_rho(E)*rho
        mtot = mc + mpe + mpair + mray
        r = np.random.rand() * mtot
        
        if r < mc:
            # COMPTON
            # Photon scatters
            ct_sc, E_sc, cos_e = sample_compton_one(E)
            # Electron created
            if in_cyl:
                # electron kinetic energy = E - E_sc
                T_e = E - E_sc
                # Choose azimuthal angle for electron
                # (we only record cos_theta relative to beam axis = z direction)
                # Electron direction is in the plane of (u_prim, scattered) - complicated
                # For the angular distribution relative to beam axis, we need the full 3D
                # Simplify: electron is forward-peaked; cos_e is in the scattering plane
                # The electron direction relative to beam axis depends on geometry.
                # For simplicity, we record cos_e directly (relative to primary photon direction,
                # which is approximately along beam axis for primary photons).
                # For scattered photons (not along beam axis), this is an approximation.
                electrons_created.append((p.copy(), T_e, cos_e, primary))
                # Note: for primary photon, cos_e is relative to beam axis (since u ≈ z)
                # For scattered photons, we'd need to rotate. Approximation here.
            # New photon: energy E_sc, direction = scattered photon direction
            # Sample azimuthal angle phi uniformly
            phi = np.random.rand() * 2 * np.pi
            sin_t = math.sqrt(max(0, 1 - ct_sc*ct_sc))
            # Rotate u by angle theta with azimuth phi
            # Build a rotation: new direction makes angle theta with u
            # Use Rodrigues' rotation formula
            # Choose a perpendicular vector
            if abs(u[2]) < 0.99:
                n_perp = np.cross(u, np.array([0.0, 0.0, 1.0]))
            else:
                n_perp = np.cross(u, np.array([1.0, 0.0, 0.0]))
            n_perp = n_perp / np.linalg.norm(n_perp)
            # Second perpendicular
            n_perp2 = np.cross(u, n_perp)
            n_perp2 = n_perp2 / np.linalg.norm(n_perp2)
            u_new = ct_sc * u + sin_t * (math.cos(phi) * n_perp + math.sin(phi) * n_perp2)
            u = u_new / np.linalg.norm(u_new)
            E = E_sc
            primary = False
        elif r < mc + mpe:
            # PHOTOELECTRIC: photon absorbed
            if in_cyl:
                T_e = E - 0.0005  # binding energy ~0.5 keV for O
                # Photoelectron roughly isotropic (or slightly forward);
                # sample isotropic direction
                cos_e = 2*np.random.rand() - 1
                electrons_created.append((p.copy(), T_e, cos_e, primary))
            break  # photon absorbed
        elif r < mc + mpe + mpair:
            # PAIR PRODUCTION: photon absorbed, electron-positron pair created
            T_total = E - 2*mec2  # total kinetic energy of pair
            if T_total > 0:
                if in_cyl:
                    # Split energy: electron gets f*T, positron (1-f)*T
                    f = np.random.rand()
                    T_e = f * T_total
                    T_p = (1-f) * T_total
                    # Pair roughly forward peaked; sample from approximate distribution
                    cos_e = 0.95  # forward
                    electrons_created.append((p.copy(), T_e, cos_e, primary))
                    electrons_created.append((p.copy(), T_p, cos_e, primary))
                    # But for "electrons set in motion by photon interactions" both electron and positron count
                # Annihilation photons: 2 × 511 keV in opposite directions
                # Skip for now (contribute to scattered photons)
                # We can add them:
                # 2 annihilation photons
                # (simplify: skip in this version, add later)
            break  # photon absorbed (pair production)
        else:
            # RAYLEIGH scattering: no energy change, small angle
            ct_sc = sample_rayleigh(E)
            phi = np.random.rand() * 2 * np.pi
            sin_t = math.sqrt(max(0, 1 - ct_sc*ct_sc))
            if abs(u[2]) < 0.99:
                n_perp = np.cross(u, np.array([0.0, 0.0, 1.0]))
            else:
                n_perp = np.cross(u, np.array([1.0, 0.0, 0.0]))
            n_perp = n_perp / np.linalg.norm(n_perp)
            n_perp2 = np.cross(u, n_perp)
            n_perp2 = n_perp2 / np.linalg.norm(n_perp2)
            u_new = ct_sc * u + sin_t * (math.cos(phi) * n_perp + math.sin(phi) * n_perp2)
            u = u_new / np.linalg.norm(u_new)
            primary = False
    
    return electrons_created