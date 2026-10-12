import numpy as np
import math
import time

# ===== CONSTANTS & CROSS SECTIONS =====
mec2 = 0.511
r0 = 2.8179403262e-13
r0_sq = r0**2
NA_m = 6.02214076e23
M_H2O = 18.01528
rho = 1.0
electrons_per_gram = 10 * NA_m / M_H2O
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
    return ct, E_sc

# Compton: compute electron direction in lab frame
def compton_electron_dir(u_in, ct_sc, E_in, E_sc, phi):
    """Return electron unit direction."""
    # u_out = scattered photon direction
    u_out = rotate_dir(u_in, ct_sc, phi)
    # Electron momentum direction = E_in*u_in - E_sc*u_out
    vec = E_in*np.asarray(u_in) - E_sc*u_out
    norm = np.linalg.norm(vec)
    if norm < 1e-10:
        return u_in.copy(), u_out
    return vec/norm, u_out

def rotate_dir(u, ct_sc, phi):
    sin_t = math.sqrt(max(0.0, 1 - ct_sc*ct_sc))
    u = np.asarray(u, dtype=float)
    if abs(u[2]) < 0.99:
        n_perp = np.cross(u, np.array([0.0, 0.0, 1.0]))
    else:
        n_perp = np.cross(u, np.array([1.0, 0.0, 0.0]))
    n_perp = n_perp / np.linalg.norm(n_perp)
    n_perp2 = np.cross(u, n_perp)
    n_perp2 = n_perp2 / np.linalg.norm(n_perp2)
    u_new = ct_sc * u + sin_t * (math.cos(phi) * n_perp + math.sin(phi) * n_perp2)
    return u_new / np.linalg.norm(u_new)

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

photon_energy_edges = np.array([0, 0.05, 0.1, 0.15, 0.2, 0.25, 0.3, 0.35, 0.4, 0.45, 0.5, 0.55, 0.6, 0.65, 0.7, 0.75, 0.8, 0.85, 0.9, 0.95, 1, 1.05, 1.1, 1.15, 1.2, 1.25, 1.3, 1.35, 1.4])
electron_energy_edges = np.array([0.01, 0.05, 0.1, 0.15, 0.2, 0.25, 0.3, 0.35, 0.4, 0.45, 0.5, 0.55, 0.6, 0.65, 0.7, 0.75, 0.8, 0.85, 0.9, 0.95, 1, 1.05, 1.1, 1.15, 1.2, 1.4])
ct_edges = np.array([-1, -0.8, -0.6, -0.4, -0.2, 0, 0.2, 0.4, 0.6, 0.8, 1])

def is_in_cylinder(x, y, z):
    return (cyl_z_min <= z <= cyl_z_max) and (x*x + y*y <= cyl_radius*cyl_radius + 1e-9)

def segment_cylinder_track(p, u, s_max):
    p = np.asarray(p, dtype=float)
    u = np.asarray(u, dtype=float)
    tz_in, tz_out = None, None
    if abs(u[2]) > 1e-12:
        t1 = (cyl_z_min - p[2]) / u[2]
        t2 = (cyl_z_max - p[2]) / u[2]
        tz_in, tz_out = min(t1,t2), max(t1,t2)
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
            ts_in, ts_out = t1, t2
    enters = []
    if tz_in is not None:
        xp = p[0] + tz_in*u[0]; yp = p[1] + tz_in*u[1]
        if xp*xp + yp*yp <= cyl_radius*cyl_radius + 1e-8:
            enters.append(tz_in)
    if ts_in is not None:
        zp = p[2] + ts_in*u[2]
        if cyl_z_min <= zp <= cyl_z_max + 1e-8:
            enters.append(ts_in)
    if not enters:
        if is_in_cylinder(p[0], p[1], p[2]):
            t_entry = 0.0
        else:
            return 0.0
    else:
        t_entry = min(enters)
        if t_entry > s_max: return 0.0
        t_entry = max(0.0, t_entry)
    exits = []
    if tz_out is not None:
        xp = p[0] + tz_out*u[0]; yp = p[1] + tz_out*u[1]
        if xp*xp + yp*yp <= cyl_radius*cyl_radius + 1e-8:
            exits.append(tz_out)
    if ts_out is not None:
        zp = p[2] + ts_out*u[2]
        if cyl_z_min - 1e-8 <= zp <= cyl_z_max + 1e-8:
            exits.append(ts_out)
    t_exit = min(exits) if exits else s_max
    if t_exit > s_max: t_exit = s_max
    track = t_exit - t_entry
    if track < 0: return 0.0
    pmid = p + u * (t_entry + t_exit)/2
    if not is_in_cylinder(pmid[0], pmid[1], pmid[2]):
        return 0.0
    return track

def make_scoring():
    return {
        'ph_prim_fluence': 0.0,
        'ph_scat_fluence': 0.0,
        'ph_prim_spec': np.zeros(28),
        'ph_scat_spec': np.zeros(28),
        'ph_prim_ang': np.zeros(10),
        'ph_scat_ang': np.zeros(10),
        # In-cylinder interactions
        'cyl_int': np.zeros(4),  # compton, pe, pair, rayleigh
        'cyl_int_prim': np.zeros(4),
        'cyl_int_scat': np.zeros(4),
        'cyl_int_energy_transfer': np.zeros(4),  # MeV weighted
        # Electrons created in cylinder
        'elec_created': [],  # (T_e, cos_e, parent_primary, process)
    }

def transport_photon_improved(p_start, u_start, E_start, primary_start, scoring):
    """Photon transport with annihilation photons, in-cylinder interaction scoring,
    correct electron directions."""
    electrons = []
    p = p_start.copy()
    u = u_start.copy()
    E = E_start
    primary = primary_start
    
    for _ in range(300):
        if E <= 1e-4: break
        mu = mu_total(E)
        if mu <= 0: break
        s_interact = -math.log(np.random.rand()) / mu
        
        # Distance to boundary
        s_boundary = 1e30
        if u[2] > 0: s_boundary = min(s_boundary, (phantom_depth - p[2]) / u[2])
        elif u[2] < 0: s_boundary = min(s_boundary, (-p[2]) / u[2])
        if u[0] > 0: s_boundary = min(s_boundary, (phantom_half - p[0]) / u[0])
        elif u[0] < 0: s_boundary = min(s_boundary, (-phantom_half - p[0]) / u[0])
        if u[1] > 0: s_boundary = min(s_boundary, (phantom_half - p[1]) / u[1])
        elif u[1] < 0: s_boundary = min(s_boundary, (-phantom_half - p[1]) / u[1])
        
        s_step = min(s_interact, s_boundary)
        if s_step < 0: s_step = 0
        
        # Score cylinder track
        L = segment_cylinder_track(p, u, s_step)
        if L > 1e-12:
            if primary:
                scoring['ph_prim_fluence'] += L
                idx = np.searchsorted(photon_energy_edges, E, side='right') - 1
                if 0 <= idx < 28: scoring['ph_prim_spec'][idx] += L
                idx_a = int((u[2]+1)/0.2); idx_a = min(9, max(0, idx_a))
                scoring['ph_prim_ang'][idx_a] += L
            else:
                scoring['ph_scat_fluence'] += L
                idx = np.searchsorted(photon_energy_edges, E, side='right') - 1
                if 0 <= idx < 28: scoring['ph_scat_spec'][idx] += L
                idx_a = int((u[2]+1)/0.2); idx_a = min(9, max(0, idx_a))
                scoring['ph_scat_ang'][idx_a] += L
        
        if s_step >= s_boundary - 1e-12:
            break  # exited phantom
        
        p = new_p = p + s_step * u
        in_cyl = is_in_cylinder(p[0], p[1], p[2])
        
        mc = mu_comp_rho(E)*rho
        mpe = mu_pe_rho(E)*rho
        mpair = mu_pair_rho(E)*rho
        mray = mu_ray_rho(E)*rho
        mtot = mc + mpe + mpair + mray
        r = np.random.rand() * mtot
        
        if r < mc:
            # COMPTON
            proc_idx = 0
            ct_sc, E_sc = sample_compton_one(E)
            phi = np.random.rand() * 2 * np.pi
            u_e, u_out = compton_electron_dir(u, ct_sc, E, E_sc, phi)
            u = u_out
            E_new = E_sc
            T_e = E - E_sc
            if in_cyl:
                scoring['cyl_int'][proc_idx] += 1
                if primary: scoring['cyl_int_prim'][proc_idx] += 1
                else: scoring['cyl_int_scat'][proc_idx] += 1
                scoring['cyl_int_energy_transfer'][proc_idx] += T_e
                electrons.append((T_e, u_e[2], primary, proc_idx))
            E = E_new
            primary = False   # after scatter, it's a scattered photon
        elif r < mc + mpe:
            # PHOTOELECTRIC
            proc_idx = 1
            T_e = E - 0.0005
            if in_cyl:
                scoring['cyl_int'][proc_idx] += 1
                if primary: scoring['cyl_int_prim'][proc_idx] += 1
                else: scoring['cyl_int_scat'][proc_idx] += 1
                scoring['cyl_int_energy_transfer'][proc_idx] += T_e
                # Photoelectron roughly isotropic
                cos_e = 2*np.random.rand() - 1
                electrons.append((T_e, cos_e, primary, proc_idx))
            break
        elif r < mc + mpe + mpair:
            # PAIR PRODUCTION
            proc_idx = 2
            T_total = E - 2*mec2
            if in_cyl:
                scoring['cyl_int'][proc_idx] += 1
                if primary: scoring['cyl_int_prim'][proc_idx] += 1
                else: scoring['cyl_int_scat'][proc_idx] += 1
                scoring['cyl_int_energy_transfer'][proc_idx] += T_total
                f = np.random.rand()
                electrons.append((f*T_total, u[2]*0.98, primary, proc_idx))
                electrons.append(((1-f)*T_total, u[2]*0.98, primary, proc_idx))
            # Annihilation photons: 2 × 511 keV opposite directions
            # Sample random direction for first, opposite for second
            theta1 = np.arccos(2*np.random.rand() - 1)
            phi1 = np.random.rand() * 2 * np.pi
            u_ann1 = np.array([math.sin(theta1)*math.cos(phi1), math.sin(theta1)*math.sin(phi1), math.cos(theta1)])
            u_ann2 = -u_ann1
            # Track annihilation photons (they are secondary/scattered photons)
            # Add them to photons to continue tracking
            # In a proper MC, we'd push them onto a stack. Here, let's handle recursively.
            # For simplicity, track one and recursively the other.
            electrons.extend(transport_annihilation_photon(p, u_ann1, 0.511, scoring))
            electrons.extend(transport_annihilation_photon(p, u_ann2, 0.511, scoring))
            break
        else:
            # RAYLEIGH
            proc_idx = 3
            if in_cyl:
                scoring['cyl_int'][proc_idx] += 1
                if primary: scoring['cyl_int_prim'][proc_idx] += 1
                else: scoring['cyl_int_scat'][proc_idx] += 1
                # No energy transfer
            ct_sc = math.cos(np.random.normal(0, 0.002))
            phi = np.random.rand() * 2 * np.pi
            u = rotate_dir(u, ct_sc, phi)
            primary = False
    
    return electrons

def transport_annihilation_photon(p_start, u_start, E_start, scoring):
    """Transport a 511 keV annihilation photon."""
    # Reuse transport_photon_improved but with primary=False
    # Need to avoid recursion issues - use a simpler version
    p = p_start.copy()
    u = u_start.copy()
    E = E_start
    electrons = []
    
    for _ in range(100):
        if E <= 1e-4: break
        mu = mu_total(E)
        if mu <= 0: break
        s_interact = -math.log(np.random.rand()) / mu
        s_boundary = 1e30
        if u[2] > 0: s_boundary = min(s_boundary, (phantom_depth - p[2]) / u[2])
        elif u[2] < 0: s_boundary = min(s_boundary, (-p[2]) / u[2])
        if u[0] > 0: s_boundary = min(s_boundary, (phantom_half - p[0]) / u[0])
        elif u[0] < 0: s_boundary = min(s_boundary, (-phantom_half - p[0]) / u[0])
        if u[1] > 0: s_boundary = min(s_boundary, (phantom_half - p[1]) / u[1])
        elif u[1] < 0: s_boundary = min(s_boundary, (-phantom_half - p[1]) / u[1])
        s_step = min(s_interact, s_boundary)
        if s_step < 0: s_step = 0
        
        L = segment_cylinder_track(p, u, s_step)
        if L > 1e-12:
            scoring['ph_scat_fluence'] += L
            idx = np.searchsorted(photon_energy_edges, E, side='right') - 1
            if 0 <= idx < 28: scoring['ph_scat_spec'][idx] += L
            idx_a = int((u[2]+1)/0.2); idx_a = min(9, max(0, idx_a))
            scoring['ph_scat_ang'][idx_a] += L
        
        if s_step >= s_boundary - 1e-12: break
        p = p + s_step * u
        in_cyl = is_in_cylinder(p[0], p[1], p[2])
        
        mc = mu_comp_rho(E)*rho
        mpe = mu_pe_rho(E)*rho
        mpair = mu_pair_rho(E)*rho
        mray = mu_ray_rho(E)*rho
        mtot = mc + mpe + mpair + mray
        r = np.random.rand() * mtot
        
        if r < mc:
            ct_sc, E_sc = sample_compton_one(E)
            phi = np.random.rand() * 2 * np.pi
            u_e, u_out = compton_electron_dir(u, ct_sc, E, E_sc, phi)
            u = u_out
            E = E_sc
            if in_cyl:
                scoring['cyl_int'][0] += 1
                scoring['cyl_int_scat'][0] += 1
                T_e = E_old_to_transfer = 0  # placeholder
                # We need E before update; fix below
            # Fixed: recompute T_e
            T_e = E - E_sc  # WRONG: E already updated
            # Let me store: T_e before update
        elif r < mc + mpe:
            if in_cyl:
                scoring['cyl_int'][1] += 1
                scoring['cyl_int_scat'][1] += 1
                T_e = E - 0.0005
                cos_e = 2*np.random.rand() - 1
                electrons.append((T_e, cos_e, False, 1))
            break
        elif r < mc + mpe + mpair:
            if in_cyl:
                scoring['cyl_int'][2] += 1
                scoring['cyl_int_scat'][2] += 1
                T_total = E - 2*mec2
                if T_total > 0:
                    f = np.random.rand()
                    electrons.append((f*T_total, u[2]*0.98, False, 2))
                    electrons.append(((1-f)*T_total, u[2]*0.98, False, 2))
            break
        else:
            if in_cyl:
                scoring['cyl_int'][3] += 1
                scoring['cyl_int_scat'][3] += 1
            ct_sc = math.cos(np.random.normal(0, 0.002))
            phi = np.random.rand() * 2 * np.pi
            u = rotate_dir(u, ct_sc, phi)
    
    return electrons

# Fix the Compton electron T_e in annihilation photon transport
def transport_annihilation_photon_v2(p_start, u_start, E_start, scoring):
    p = p_start.copy()
    u = u_start.copy()
    E = E_start
    electrons = []
    
    for _ in range(100):
        if E <= 1e-4: break
        mu = mu_total(E)
        if mu <= 0: break
        s_interact = -math.log(np.random.rand()) / mu
        s_boundary = 1e30
        if u[2] > 0: s_boundary = min(s_boundary, (phantom_depth - p[2]) / u[2])
        elif u[2] < 0: s_boundary = min(s_boundary, (-p[2]) / u[2])
        if u[0] > 0: s_boundary = min(s_boundary, (phantom_half - p[0]) / u[0])
        elif u[0] < 0: s_boundary = min(s_boundary, (-phantom_half - p[0]) / u[0])
        if u[1] > 0: s_boundary = min(s_boundary, (phantom_half - p[1]) / u[1])
        elif u[1] < 0: s_boundary = min(s_boundary, (-phantom_half - p[1]) / u[1])
        s_step = min(s_interact, s_boundary)
        
        L = segment_cylinder_track(p, u, s_step)
        if L > 1e-12:
            scoring['ph_scat_fluence'] += L
            idx = np.searchsorted(photon_energy_edges, E, side='right') - 1
            if 0 <= idx < 28: scoring['ph_scat_spec'][idx] += L
            idx_a = int((u[2]+1)/0.2); idx_a = min(9, max(0, idx_a))
            scoring['ph_scat_ang'][idx_a] += L
        
        if s_step >= s_boundary - 1e-12: break
        p = p + s_step * u
        in_cyl = is_in_cylinder(p[0], p[1], p[2])
        
        mc = mu_comp_rho(E)*rho
        mpe = mu_pe_rho(E)*rho
        mpair = mu_pair_rho(E)*rho
        mray = mu_ray_rho(E)*rho
        mtot = mc + mpe + mpair + mray
        r = np.random.rand() * mtot
        
        if r < mc:
            ct_sc, E_sc = sample_compton_one(E)
            T_e = E - E_sc
            phi = np.random.rand() * 2 * np.pi
            u_e, u_out = compton_electron_dir(u, ct_sc, E, E_sc, phi)
            u = u_out
            E = E_sc
            if in_cyl:
                scoring['cyl_int'][0] += 1
                scoring['cyl_int_scat'][0] += 1
                scoring['cyl_int_energy_transfer'][0] += T_e
                electrons.append((T_e, u_e[2], False, 0))
        elif r < mc + mpe:
            if in_cyl:
                scoring['cyl_int'][1] += 1
                scoring['cyl_int_scat'][1] += 1
                T_e = E - 0.0005
                scoring['cyl_int_energy_transfer'][1] += T_e
                cos_e = 2*np.random.rand() - 1
                electrons.append((T_e, cos_e, False, 1))
            break
        elif r < mc + mpe + mpair:
            if in_cyl:
                scoring['cyl_int'][2] += 1
                scoring['cyl_int_scat'][2] += 1
                T_total = E - 2*mec2
                scoring['cyl_int_energy_transfer'][2] += T_total
                if T_total > 0:
                    f = np.random.rand()
                    electrons.append((f*T_total, u[2]*0.98, False, 2))
                    electrons.append(((1-f)*T_total, u[2]*0.98, False, 2))
            break
        else:
            if in_cyl:
                scoring['cyl_int'][3] += 1
                scoring['cyl_int_scat'][3] += 1
            ct_sc = math.cos(np.random.normal(0, 0.002))
            phi = np.random.rand() * 2 * np.pi
            u = rotate_dir(u, ct_sc, phi)
    return electrons

# Update: use transport_photon_improved with the correct annihilation handling
def transport_photon_v2(p_start, u_start, E_start, primary_start, scoring):
    electrons = []
    p = p_start.copy()
    u = u_start.copy()
    E = E_start
    primary = primary_start
    
    for _ in range(300):
        if E <= 1e-4: break
        mu = mu_total(E)
        if mu <= 0: break
        s_interact = -math.log(np.random.rand()) / mu
        s_boundary = 1e30
        if u[2] > 0: s_boundary = min(s_boundary, (phantom_depth - p[2]) / u[2])
        elif u[2] < 0: s_boundary = min(s_boundary, (-p[2]) / u[2])
        if u[0] > 0: s_boundary = min(s_boundary, (phantom_half - p[0]) / u[0])
        elif u[0] < 0: s_boundary = min(s_boundary, (-phantom_half - p[0]) / u[0])
        if u[1] > 0: s_boundary = min(s_boundary, (phantom_half - p[1]) / u[1])
        elif u[1] < 0: s_boundary = min(s_boundary, (-phantom_half - p[1]) / u[1])
        s_step = min(s_interact, s_boundary)
        if s_step < 0: s_step = 0
        
        L = segment_cylinder_track(p, u, s_step)
        if L > 1e-12:
            if primary:
                scoring['ph_prim_fluence'] += L
                idx = np.searchsorted(photon_energy_edges, E, side='right') - 1
                if 0 <= idx < 28: scoring['ph_prim_spec'][idx] += L
                idx_a = int((u[2]+1)/0.2); idx_a = min(9, max(0, idx_a))
                scoring['ph_prim_ang'][idx_a] += L
            else:
                scoring['ph_scat_fluence'] += L
                idx = np.searchsorted(photon_energy_edges, E, side='right') - 1
                if 0 <= idx < 28: scoring['ph_scat_spec'][idx] += L
                idx_a = int((u[2]+1)/0.2); idx_a = min(9, max(0, idx_a))
                scoring['ph_scat_ang'][idx_a] += L
        
        if s_step >= s_boundary - 1e-12: break
        p = p + s_step * u
        in_cyl = is_in_cylinder(p[0], p[1], p[2])
        
        mc = mu_comp_rho(E)*rho
        mpe = mu_pe_rho(E)*rho
        mpair = mu_pair_rho(E)*rho
        mray = mu_ray_rho(E)*rho
        mtot = mc + mpe + mpair + mray
        r = np.random.rand() * mtot
        
        if r < mc:
            proc_idx = 0
            ct_sc, E_sc = sample_compton_one(E)
            T_e = E - E_sc
            phi = np.random.rand() * 2 * np.pi
            u_e, u_out = compton_electron_dir(u, ct_sc, E, E_sc, phi)
            u = u_out
            if in_cyl:
                scoring['cyl_int'][proc_idx] += 1
                if primary: scoring['cyl_int_prim'][proc_idx] += 1
                else: scoring['cyl_int_scat'][proc_idx] += 1
                scoring['cyl_int_energy_transfer'][proc_idx] += T_e
                electrons.append((T_e, u_e[2], primary, proc_idx))
            E = E_sc
            primary = False
        elif r < mc + mpe:
            proc_idx = 1
            T_e = E - 0.0005
            if in_cyl:
                scoring['cyl_int'][proc_idx] += 1
                if primary: scoring['cyl_int_prim'][proc_idx] += 1
                else: scoring['cyl_int_scat'][proc_idx] += 1
                scoring['cyl_int_energy_transfer'][proc_idx] += T_e
                cos_e = 2*np.random.rand() - 1
                electrons.append((T_e, cos_e, primary, proc_idx))
            break
        elif r < mc + mpe + mpair:
            proc_idx = 2
            T_total = E - 2*mec2
            if in_cyl:
                scoring['cyl_int'][proc_idx] += 1
                if primary: scoring['cyl_int_prim'][proc_idx] += 1
                else: scoring['cyl_int_scat'][proc_idx] += 1
                scoring['cyl_int_energy_transfer'][proc_idx] += T_total
                if T_total > 0:
                    f = np.random.rand()
                    electrons.append((f*T_total, u[2]*0.98, primary, proc_idx))
                    electrons.append(((1-f)*T_total, u[2]*0.98, primary, proc_idx))
            # Annihilation photons
            theta1 = np.arccos(2*np.random.rand() - 1)
            phi1 = np.random.rand() * 2 * np.pi
            u_ann1 = np.array([math.sin(theta1)*math.cos(phi1), math.sin(theta1)*math.sin(phi1), math.cos(theta1)])
            electrons.extend(transport_annihilation_photon_v2(p, u_ann1, 0.511, scoring))
            electrons.extend(transport_annihilation_photon_v2(p, -u_ann1, 0.511, scoring))
            break
        else:
            proc_idx = 3
            if in_cyl:
                scoring['cyl_int'][proc_idx] += 1
                if primary: scoring['cyl_int_prim'][proc_idx] += 1
                else: scoring['cyl_int_scat'][proc_idx] += 1
            ct_sc = math.cos(np.random.normal(0, 0.002))
            phi = np.random.rand() * 2 * np.pi
            u = rotate_dir(u, ct_sc, phi)
            primary = False
    return electrons

def run_mc(n_photons, seed=123, verbose=True):
    np.random.seed(seed)
    scoring = make_scoring()
    all_electrons = []
    t0 = time.time()
    for i in range(n_photons):
        x_t = (np.random.rand() - 0.5) * 2 * field_half
        y_t = (np.random.rand() - 0.5) * 2 * field_half
        u = np.array([x_t, y_t, SSD])
        u = u / np.linalg.norm(u)
        p_surf = np.array([x_t, y_t, 0.0])
        electrons = transport_photon_v2(p_surf, u, 2.0, True, scoring)
        all_electrons.extend(electrons)
        if verbose and (i+1) % 100000 == 0:
            print(f"  {i+1}/{n_photons}, {time.time()-t0:.1f}s, electrons={len(all_electrons)}")
    if verbose:
        print(f"Done: {n_photons} photons in {time.time()-t0:.1f}s")
    return scoring, all_electrons

# Run 200k for testing
scoring, electrons = run_mc(200000, seed=123)
N = 200000

print(f"\n=== RESULTS ===")
print(f"Primary fluence track: {scoring['ph_prim_fluence']:.2f}")
print(f"Scattered fluence track: {scoring['ph_scat_fluence']:.2f}")
print(f"SPR (fluence): {scoring['ph_scat_fluence']/scoring['ph_prim_fluence']:.4f}")

# Normalization to Phi0
# Phi0 per source photon = pi*0.2/(121*V_cyl) = pi/(605*V_cyl)
Phi0_per_src = np.pi * 0.2 / (121 * cyl_volume)
print(f"\nPhi0 per source photon: {Phi0_per_src:.6f} cm^-2")

prim_per_phi0 = scoring['ph_prim_fluence']/(cyl_volume*N)/Phi0_per_src
scat_per_phi0 = scoring['ph_scat_fluence']/(cyl_volume*N)/Phi0_per_src
print(f"L1_primary_fluence_per_phi0 = {prim_per_phi0:.4f} (expect exp(-mu*10)={np.exp(-mu_total(2.0)*10):.4f})")
print(f"L1_scattered_fluence_per_phi0 = {scat_per_phi0:.4f}")

# Interaction stats in cylinder
print(f"\nCylinder interactions total: {scoring['cyl_int']}")
print(f"  Compton: {scoring['cyl_int'][0]}, PE: {scoring['cyl_int'][1]}, Pair: {scoring['cyl_int'][2]}, Rayleigh: {scoring['cyl_int'][3]}")
print(f"Cylinder interactions primary: {scoring['cyl_int_prim']}")
print(f"Cylinder interactions scattered: {scoring['cyl_int_scat']}")
print(f"Energy transfer in cylinder: {scoring['cyl_int_energy_transfer']} MeV")
print(f"Electrons created in cylinder: {len(electrons)}")