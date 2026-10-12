import numpy as np
import math
import time
import pickle

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

# Precompute cross-section lookup tables for photon energy 0.01-10 MeV
# for fast mu_total lookup
E_lut = np.linspace(0.01, 10.0, 2000)
mu_lut = np.array([mu_total(e) for e in E_lut])
mu_comp_lut = np.array([mu_comp_rho(e)*rho for e in E_lut])
mu_pe_lut = np.array([mu_pe_rho(e)*rho for e in E_lut])
mu_pair_lut = np.array([mu_pair_rho(e)*rho for e in E_lut])
mu_ray_lut = np.array([mu_ray_rho(e)*rho for e in E_lut])

def mu_total_fast(E):
    return float(np.interp(E, E_lut, mu_lut))

def mu_components_fast(E):
    idx = np.searchsorted(E_lut, E)
    idx = min(max(idx, 0), len(E_lut)-1)
    return (mu_comp_lut[idx], mu_pe_lut[idx], mu_pair_lut[idx], mu_ray_lut[idx], mu_lut[idx])

# Precompute Compton CDF for a grid of energies
comp_E_grid = np.linspace(0.01, 2.0, 400)
comp_ct_grid = np.linspace(-1, 1, 2001)
comp_cdf_table = {}
for Eg in comp_E_grid:
    alpha = Eg / mec2
    k = 1.0 / (1.0 + alpha*(1.0 - comp_ct_grid))
    dsigma = 2*np.pi * (r0_sq/2.0) * k**2 * (k + 1/k - (1 - comp_ct_grid**2))
    cdf = np.cumsum(dsigma)
    cdf = cdf / cdf[-1]
    comp_cdf_table[Eg] = cdf

def sample_compton_fast(E_inc):
    """Fast Compton sampling using precomputed CDF."""
    # Find nearest energy in grid
    idx = int(round((E_inc - 0.01) / (2.0 - 0.01) * 399))
    idx = min(max(idx, 0), 399)
    Eg = comp_E_grid[idx]
    cdf = comp_cdf_table[Eg]
    u = np.random.rand()
    ct = float(np.interp(u, cdf, comp_ct_grid))
    # Correct for energy difference
    alpha = E_inc / mec2
    E_sc = E_inc / (1.0 + alpha*(1.0 - ct))
    return ct, E_sc

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

def compton_electron_dir(u_in, ct_sc, E_in, E_sc, phi):
    u_out = rotate_dir(u_in, ct_sc, phi)
    vec = E_in*np.asarray(u_in) - E_sc*u_out
    norm = np.linalg.norm(vec)
    if norm < 1e-10:
        return u_in.copy(), u_out
    return vec/norm, u_out

def make_scoring():
    return {
        'ph_prim_fluence': 0.0,
        'ph_scat_fluence': 0.0,
        'ph_prim_spec': np.zeros(28),
        'ph_scat_spec': np.zeros(28),
        'ph_prim_ang': np.zeros(10),
        'ph_scat_ang': np.zeros(10),
        'cyl_int': np.zeros(4),
        'cyl_int_prim': np.zeros(4),
        'cyl_int_scat': np.zeros(4),
        'cyl_int_energy_transfer': np.zeros(4),
        'elec_created': [],  # (T_e, cos_e_beam, parent_primary, process_idx)
    }

def transport_photon_v3(p_start, u_start, E_start, primary_start, scoring):
    electrons = []
    p = p_start.copy()
    u = u_start.copy()
    E = E_start
    primary = primary_start
    
    for _ in range(300):
        if E <= 1e-4: break
        # Fast mu lookup
        mu = float(np.interp(E, E_lut, mu_lut))
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
        
        # Interaction
        idx_mu = np.searchsorted(E_lut, E)
        idx_mu = min(max(idx_mu, 1), len(E_lut)-1)
        mc = mu_comp_lut[idx_mu]
        mpe = mu_pe_lut[idx_mu]
        mpair = mu_pair_lut[idx_mu]
        mray = mu_ray_lut[idx_mu]
        mtot = mu_lut[idx_mu]
        
        r = np.random.rand() * mtot
        if r < mc:
            proc_idx = 0
            ct_sc, E_sc = sample_compton_fast(E)
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
            # annihilation photons
            theta1 = np.arccos(2*np.random.rand() - 1)
            phi1 = np.random.rand() * 2 * np.pi
            u_ann1 = np.array([math.sin(theta1)*math.cos(phi1), math.sin(theta1)*math.sin(phi1), math.cos(theta1)])
            electrons.extend(transport_annihilation_v3(p, u_ann1, scoring))
            electrons.extend(transport_annihilation_v3(p, -u_ann1, scoring))
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

def transport_annihilation_v3(p_start, u_start, scoring):
    p = p_start.copy()
    u = u_start.copy()
    E = 0.511
    electrons = []
    for _ in range(100):
        if E <= 1e-4: break
        mu = float(np.interp(E, E_lut, mu_lut))
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
        
        idx_mu = np.searchsorted(E_lut, E)
        idx_mu = min(max(idx_mu, 1), len(E_lut)-1)
        mc = mu_comp_lut[idx_mu]; mpe = mu_pe_lut[idx_mu]
        mpair = mu_pair_lut[idx_mu]; mray = mu_ray_lut[idx_mu]
        mtot = mu_lut[idx_mu]
        r = np.random.rand() * mtot
        if r < mc:
            ct_sc, E_sc = sample_compton_fast(E)
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
                electrons.append((T_e, 2*np.random.rand()-1, False, 1))
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

def run_mc_v3(n_photons, seed=123, verbose=True):
    np.random.seed(seed)
    scoring = make_scoring()
    all_electrons = []
    t0 = time.time()
    for i in range(n_photons):
        x_t = (np.random.rand() - 0.5) * 10
        y_t = (np.random.rand() - 0.5) * 10
        u = np.array([x_t, y_t, SSD])
        norm = math.sqrt(x_t*x_t + y_t*y_t + SSD*SSD)
        u = u / norm
        p_surf = np.array([x_t, y_t, 0.0])
        electrons = transport_photon_v3(p_surf, u, 2.0, True, scoring)
        all_electrons.extend(electrons)
        if verbose and (i+1) % 100000 == 0:
            print(f"  {i+1}/{n_photons}, {time.time()-t0:.1f}s")
    if verbose:
        print(f"Done: {n_photons} photons in {time.time()-t0:.1f}s")
    return scoring, all_electrons

# Quick test with 20k first
scoring_test, electrons_test = run_mc_v3(20000, seed=1)
print(f"Test: prim_track={scoring_test['ph_prim_fluence']:.1f}, scat_track={scoring_test['ph_scat_fluence']:.1f}")
print(f"SPR: {scoring_test['ph_scat_fluence']/scoring_test['ph_prim_fluence']:.4f}")
print(f"Cyl ints: {scoring_test['cyl_int']}")