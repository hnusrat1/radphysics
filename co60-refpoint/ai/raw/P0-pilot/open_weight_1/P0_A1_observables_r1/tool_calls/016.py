import numpy as np
import math
import time

# Redefine everything in a single cell for testing
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

def sample_compton_one(E_inc):
    alpha = E_inc / mec2
    ct_grid = np.linspace(-1, 1, 4001)
    k = 1.0 / (1.0 + alpha*(1.0 - ct_grid))
    dsigma = 2*np.pi * (r0_sq/2.0) * k**2 * (k + 1/k - (1 - ct_grid**2))
    cdf = np.cumsum(dsigma)
    cdf = cdf / cdf[-1]
    u = np.random.rand()
    ct = float(np.interp(u, cdf, ct_grid))
    E_sc = E_inc / (1.0 + alpha*(1.0 - ct))
    cos_e = (E_inc - E_sc*ct) / math.sqrt(E_inc**2 + E_sc**2 - 2*E_inc*E_sc*ct)
    return ct, E_sc, cos_e

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
    tz_in, tz_out = None, None
    if abs(u[2]) > 1e-12:
        t1 = (cyl_z_min - p[2]) / u[2]
        t2 = (cyl_z_max - p[2]) / u[2]
        tz_in, tz_out = min(t1,t2), max(t1,t2)
        if tz_out < 0: return 0.0
        if tz_in < 0: tz_in = 0.0
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
    enters, exits = [], []
    if tz_in is not None:
        xp = p[0] + tz_in*u[0]; yp = p[1] + tz_in*u[1]
        if xp*xp + yp*yp <= cyl_radius*cyl_radius + 1e-8:
            enters.append(tz_in)
    if ts_in is not None:
        zp = p[2] + ts_in*u[2]
        if cyl_z_min <= zp <= cyl_z_max:
            enters.append(ts_in)
    if tz_out is not None:
        xp = p[0] + tz_out*u[0]; yp = p[1] + tz_out*u[1]
        if xp*xp + yp*yp <= cyl_radius*cyl_radius + 1e-8:
            exits.append(tz_out)
    if ts_out is not None:
        zp = p[2] + ts_out*u[2]
        if cyl_z_min <= zp <= cyl_z_max:
            exits.append(ts_out)
    if not enters:
        if is_in_cylinder(p[0], p[1], p[2]):
            t_entry = 0.0
        else:
            return 0.0
    else:
        t_entry = min(enters)
    t_exit = min(exits) if exits else 1e9
    if t_exit < t_entry: return 0.0
    return max(0.0, t_exit - t_entry)

# Scoring structure
photon_energy_edges = np.array([0, 0.05, 0.1, 0.15, 0.2, 0.25, 0.3, 0.35, 0.4, 0.45, 0.5, 0.55, 0.6, 0.65, 0.7, 0.75, 0.8, 0.85, 0.9, 0.95, 1, 1.05, 1.1, 1.15, 1.2, 1.25, 1.3, 1.35, 1.4])

def make_scoring():
    return {
        'ph_prim_fluence': 0.0,
        'ph_scat_fluence': 0.0,
        'ph_prim_spec': np.zeros(28),
        'ph_scat_spec': np.zeros(28),
        'ph_prim_ang': np.zeros(10),
        'ph_scat_ang': np.zeros(10),
        'total_ph_track': 0.0,
    }

def transport_photon(pos, dirn, energy, is_primary, scoring):
    """Transport one photon. Returns list of (pos, T_e, cos_e, parent_primary) for electrons created inside cylinder."""
    electrons = []
    p = pos.copy()
    u = dirn.copy()
    E = energy
    primary = is_primary
    
    for _ in range(200):  # max 200 steps
        # Score current segment
        L = ray_cylinder_track(p, u)
        if L > 1e-12:
            if primary:
                scoring['ph_prim_fluence'] += L
                idx = np.searchsorted(photon_energy_edges, E, side='right') - 1
                if 0 <= idx < 28: scoring['ph_prim_spec'][idx] += L
                idx_a = int((u[2] + 1) / 0.2)
                idx_a = min(9, max(0, idx_a))
                scoring['ph_prim_ang'][idx_a] += L
            else:
                scoring['ph_scat_fluence'] += L
                idx = np.searchsorted(photon_energy_edges, E, side='right') - 1
                if 0 <= idx < 28: scoring['ph_scat_spec'][idx] += L
                idx_a = int((u[2] + 1) / 0.2)
                idx_a = min(9, max(0, idx_a))
                scoring['ph_scat_ang'][idx_a] += L
            scoring['total_ph_track'] += L
        
        # Sample next interaction
        if E <= 1e-4: break
        mu = mu_total(E)
        if mu <= 0: break
        s = -math.log(np.random.rand()) / mu
        new_p = p + s*u
        
        if new_p[2] < 0 or new_p[2] > phantom_depth: break
        if abs(new_p[0]) > phantom_half or abs(new_p[1]) > phantom_half: break
        
        p = new_p
        in_cyl = is_in_cylinder(p[0], p[1], p[2])
        
        mc = mu_comp_rho(E)*rho
        mpe = mu_pe_rho(E)*rho
        mpair = mu_pair_rho(E)*rho
        mray = mu_ray_rho(E)*rho
        mtot = mc + mpe + mpair + mray
        r = np.random.rand() * mtot
        
        if r < mc:
            # COMPTON
            ct_sc, E_sc, cos_e = sample_compton_one(E)
            if in_cyl:
                T_e = E - E_sc
                electrons.append((p.copy(), T_e, cos_e, primary))
            # Rotate direction
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
            E = E_sc
            primary = False
        elif r < mc + mpe:
            if in_cyl:
                T_e = E - 0.0005
                cos_e = 2*np.random.rand() - 1
                electrons.append((p.copy(), T_e, cos_e, primary))
            break
        elif r < mc + mpe + mpair:
            T_total = E - 2*mec2
            if T_total > 0:
                if in_cyl:
                    f = np.random.rand()
                    T_e = f * T_total
                    T_p = (1-f) * T_total
                    cos_e = 0.95
                    electrons.append((p.copy(), max(0,T_e), cos_e, primary))
                    electrons.append((p.copy(), max(0,T_p), cos_e, primary))
            break
        else:
            # RAYLEIGH - small angle, no energy loss
            ct_sc = math.cos(np.random.normal(0, 0.002))
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
    
    return electrons

# === MAIN MC LOOP ===
def run_photon_mc(n_photons, seed=42):
    np.random.seed(seed)
    scoring = make_scoring()
    all_electrons = []  # (pos, T_e, cos_e, parent_primary)
    
    t0 = time.time()
    for i in range(n_photons):
        # Sample source photon direction: uniform in pyramid
        # Target point uniformly on 10x10 field at surface
        x_t = (np.random.rand() - 0.5) * 2 * field_half
        y_t = (np.random.rand() - 0.5) * 2 * field_half
        # Direction from source (0,0,-100) to (x_t, y_t, 0)
        u = np.array([x_t, y_t, SSD])
        u = u / np.linalg.norm(u)
        
        # Start at source
        pos = np.array([0.0, 0.0, -SSD])
        E = 2.0
        primary = True
        
        # Transport (including moving to surface within the loop by distance sampling in vacuum)
        # First move to surface:
        if u[2] <= 0: continue
        t_surf = SSD / u[2]
        p_surf = pos + t_surf * u
        # Check within phantom x,y bounds at surface
        if abs(p_surf[0]) > phantom_half or abs(p_surf[1]) > phantom_half:
            continue
        
        # Now transport inside phantom starting at surface
        electrons = transport_photon(p_surf, u, E, primary, scoring)
        all_electrons.extend(electrons)
        
        if (i+1) % 20000 == 0:
            elapsed = time.time() - t0
            print(f"  {i+1}/{n_photons} photons done, {elapsed:.1f}s")
    
    print(f"MC done: {n_photons} photons in {time.time()-t0:.1f}s")
    print(f"Primary fluence track: {scoring['ph_prim_fluence']:.2f} cm")
    print(f"Scattered fluence track: {scoring['ph_scat_fluence']:.2f} cm")
    print(f"Total track: {scoring['total_ph_track']:.2f} cm")
    
    # Normalize: fluence = track length / volume
    # For N photons, each with weight 1 (corresponding to 1 photon per steradian-normalized...),
    # the fluence per source photon is track_length/volume/N_photons
    # But we need to normalize to Phi0!
    
    # Phi0 = primary fluence at POI (110 cm) without phantom.
    # With 1 photon per steradian... Actually, let's think about the normalization.
    # In the MC, each "source photon" is a single photon emitted into the pyramid.
    # The total number of source photons N_src determines Phi0:
    # Phi0 = N_src_per_sr / r_poi^2 where N_src_per_sr = N_src / Omega_pyramid
    # So Phi0 = N_src / (Omega_pyramid * r_poi^2)
    # And the primary fluence at the POI with phantom = Phi0 * exp(-mu * 10)
    
    # In the MC, the "fluence per source photon" at a point is:
    # track_length_per_photon / volume = (sum of L over photons) / volume / N_src
    # The primary photon fluence in the cylinder per source photon:
    # = (primary track length) / volume / N_src
    # The primary fluence per source photon should equal:
    # (1/(Omega_pyramid * r_poi^2)) * exp(-mu * 10)
    # = 1/(0.009975 * 110^2) * 0.611  -- but wait, this is per steradian fluence at POI
    
    # Let me compute the expected primary fluence in the cylinder per source photon:
    # The cylinder is on the central axis. Primary photons crossing the cylinder travel 0.2 cm each.
    # Number of primary photons crossing the cylinder = N_src * (cyl_area / field_area_at_10cm) * exp(-mu*10)
    #   where cyl_area = pi * r^2 = pi, field area at 10 cm = (11)^2 = 121
    # So fluence per source photon = (pi/121) * exp(-mu*10) * 0.2 / cyl_volume
    #   = (pi/121) * 0.611 * 0.2 / (pi * 1^2 * 0.2)
    #   = 0.611/121 = 0.00505 per source photon
    
    # Actually more precisely: Nu/photons_crossing = N_src * exp(-mu*10) * (cyl_area / field_area_at_POI)
    # fluence = (photons_crossing * 0.2) / cyl_volume = N_src * 0.611 * (pi/121) * 0.2/(pi*0.2) = N_src * 0.611/121
    
    # Check with MC:
    # primary_fluence_per_source_photon = ph_prim_fluence / cyl_volume / N_src
    # = ph_prim_fluence / (0.6283) / N_src
    # This should equal 0.611/121 = 0.00505
    
    return scoring, all_electrons

# Run small test
scoring, electrons = run_photon_mc(50000, seed=42)
N = 50000
cyl_vol = cyl_volume
print(f"\nPrimary fluence per source photon: {scoring['ph_prim_fluence']/cyl_vol/N:.6f}")
print(f"Expected: {0.611/121:.6f}")
print(f"Scatter fluence per source photon: {scoring['ph_scat_fluence']/cyl_vol/N:.6f}")
print(f"Number of electrons created in cylinder: {len(electrons)}")

# Ratios
prim_flu = scoring['ph_prim_fluence']
scat_flu = scoring['ph_scat_fluence']
print(f"\nScatter-to-primary fluence ratio: {scat_flu/prim_flu:.4f}")
print(f"(Single-scatter analytic estimate was: 0.203/0.611 = {0.203/0.611:.4f})")