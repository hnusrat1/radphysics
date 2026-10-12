import numpy as np
import math
import time
import pickle

# ===== ALL PHYSICS MODELS =====
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

# Precompute LUTs
E_lut = np.linspace(0.01, 10.0, 2000)
mu_lut = np.array([mu_total(e) for e in E_lut])
mu_comp_lut = np.array([mu_comp_rho(e)*rho for e in E_lut])
mu_pe_lut = np.array([mu_pe_rho(e)*rho for e in E_lut])
mu_pair_lut = np.array([mu_pair_rho(e)*rho for e in E_lut])
mu_ray_lut = np.array([mu_ray_rho(e)*rho for e in E_lut])

# Compton CDF cache
comp_E_grid = np.linspace(0.01, 2.0, 400)
comp_ct_grid = np.linspace(-1, 1, 2001)
comp_cdf_table = {}
for Eg in comp_E_grid:
    alpha = Eg / mec2
    k = 1.0 / (1.0 + alpha*(1.0 - comp_ct_grid))
    dsigma = 2*np.pi * (r0_sq/2.0) * k**2 * (k + 1/k - (1 - comp_ct_grid**2))
    cdf = np.cumsum(dsigma); cdf = cdf / cdf[-1]
    comp_cdf_table[Eg] = cdf

def sample_compton_fast(E_inc):
    idx = int(round((E_inc - 0.01) / (2.0 - 0.01) * 399))
    idx = min(max(idx, 0), 399)
    Eg = comp_E_grid[idx]
    cdf = comp_cdf_table[Eg]
    u_rand = np.random.rand()
    ct = float(np.interp(u_rand, cdf, comp_ct_grid))
    alpha = E_inc / mec2
    E_sc = E_inc / (1.0 + alpha*(1.0 - ct))
    return ct, E_sc

# Electron stopping power (ICRU 49)
E_sp = np.array([0.010, 0.015, 0.020, 0.030, 0.050, 0.070, 0.100, 0.150, 0.200, 
                 0.300, 0.400, 0.500, 0.600, 0.700, 0.800, 0.900, 1.000, 
                 1.250, 1.500, 1.750, 2.000, 2.500, 3.000])
S_col_tab = np.array([23.14, 16.87, 13.29, 9.67, 6.47, 5.10, 4.10, 3.25, 2.85,
                  2.40, 2.16, 2.03, 1.95, 1.90, 1.87, 1.86, 1.85,
                  1.86, 1.86, 1.89, 1.92, 2.00, 2.08])
S_rad_tab = np.array([0.0, 0.0, 0.0, 0.0, 0.0, 0.0, 0.0, 0.0, 0.0,
                  0.0, 0.0, 0.0, 0.0, 0.0, 0.0, 0.0, 0.0156,
                  0.0296, 0.0477, 0.071, 0.099, 0.155, 0.220])
R_csda_tab = np.array([0.000257, 0.000506, 0.000812, 0.00158, 0.00438, 0.00786, 0.0142, 0.0265, 0.0435,
                   0.0858, 0.135, 0.181, 0.235, 0.286, 0.342, 0.388, 0.437,
                   0.618, 0.712, 0.856, 1.01, 1.40, 1.80])

def S_col_w(E):
    E = np.clip(E, E_sp[0], E_sp[-1])
    return float(np.interp(E, E_sp, S_col_tab)) * rho
def S_rad_w(E):
    E = np.clip(E, E_sp[0], E_sp[-1])
    return float(np.interp(E, E_sp, S_rad_tab)) * rho
def R_csda_w(T):
    if T <= E_sp[0]: return R_csda_tab[0]
    if T >= E_sp[-1]: return R_csda_tab[-1]
    return math.exp(float(np.interp(math.log(T), np.log(E_sp), np.log(R_csda_tab))))

# Mu_en/rho for water (XCOM-like), for kerma calculations
xcom_en_E = np.array([0.01, 0.03, 0.05, 0.1, 0.15, 0.2, 0.3, 0.4, 0.5, 0.6, 0.8, 1.0, 1.25, 1.5, 2.0, 3.0])
xcom_en_mu = np.array([4.78, 0.160, 0.0405, 0.0255, 0.0314, 0.0303, 0.0318, 0.0325, 0.0326, 0.0327, 0.0326, 0.0310, 0.0292, 0.0280, 0.0233, 0.0200])
def mu_en_rho_w(E):
    E = np.clip(E, 0.01, 3.0)
    return float(np.exp(np.interp(np.log(E), np.log(xcom_en_E), np.log(xcom_en_mu))))

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
cyl_center = np.array([0.0, 0.0, depth_poi])

photon_energy_edges = np.array([0, 0.05, 0.1, 0.15, 0.2, 0.25, 0.3, 0.35, 0.4, 0.45, 0.5, 0.55, 0.6, 0.65, 0.7, 0.75, 0.8, 0.85, 0.9, 0.95, 1, 1.05, 1.1, 1.15, 1.2, 1.25, 1.3, 1.35, 1.4])
electron_energy_edges = np.array([0.01, 0.05, 0.1, 0.15, 0.2, 0.25, 0.3, 0.35, 0.4, 0.45, 0.5, 0.55, 0.6, 0.65, 0.7, 0.75, 0.8, 0.85, 0.9, 0.95, 1, 1.05, 1.1, 1.15, 1.2, 1.4])
ct_edges = np.array([-1, -0.8, -0.6, -0.4, -0.2, 0, 0.2, 0.4, 0.6, 0.8, 1])

def is_in_cylinder(x, y, z):
    return (cyl_z_min <= z <= cyl_z_max) and (x*x + y*y <= cyl_radius*cyl_radius + 1e-9)

def segment_cylinder_track(p, u, s_max):
    p = np.asarray(p, dtype=float); u = np.asarray(u, dtype=float)
    tz_in, tz_out = None, None
    if abs(u[2]) > 1e-12:
        t1 = (cyl_z_min - p[2]) / u[2]; t2 = (cyl_z_max - p[2]) / u[2]
        tz_in, tz_out = min(t1,t2), max(t1,t2)
    a = u[0]**2 + u[1]**2; b = 2*(p[0]*u[0] + p[1]*u[1]); c = p[0]**2 + p[1]**2 - cyl_radius**2
    ts_in, ts_out = None, None
    if a > 1e-12:
        disc = b*b - 4*a*c
        if disc >= 0:
            sd = math.sqrt(disc); t1 = (-b-sd)/(2*a); t2 = (-b+sd)/(2*a)
            ts_in, ts_out = t1, t2
    enters = []
    if tz_in is not None:
        xp = p[0]+tz_in*u[0]; yp = p[1]+tz_in*u[1]
        if xp*xp+yp*yp <= cyl_radius*cyl_radius+1e-8: enters.append(tz_in)
    if ts_in is not None:
        zp = p[2]+ts_in*u[2]
        if cyl_z_min <= zp <= cyl_z_max+1e-8: enters.append(ts_in)
    if not enters:
        if is_in_cylinder(p[0],p[1],p[2]): t_entry = 0.0
        else: return 0.0
    else:
        t_entry = min(enters)
        if t_entry > s_max: return 0.0
        t_entry = max(0.0, t_entry)
    exits = []
    if tz_out is not None:
        xp = p[0]+tz_out*u[0]; yp = p[1]+tz_out*u[1]
        if xp*xp+yp*yp <= cyl_radius*cyl_radius+1e-8: exits.append(tz_out)
    if ts_out is not None:
        zp = p[2]+ts_out*u[2]
        if cyl_z_min-1e-8 <= zp <= cyl_z_max+1e-8: exits.append(ts_out)
    t_exit = min(exits) if exits else s_max
    if t_exit > s_max: t_exit = s_max
    track = t_exit - t_entry
    if track < 0: return 0.0
    pmid = p + u*(t_entry+t_exit)/2
    if not is_in_cylinder(pmid[0],pmid[1],pmid[2]): return 0.0
    return track

def rotate_dir(u, ct_sc, phi):
    sin_t = math.sqrt(max(0.0, 1-ct_sc*ct_sc))
    u = np.asarray(u, dtype=float)
    if abs(u[2]) < 0.99: n_perp = np.cross(u, np.array([0.0,0.0,1.0]))
    else: n_perp = np.cross(u, np.array([1.0,0.0,0.0]))
    n_perp = n_perp/np.linalg.norm(n_perp)
    n_perp2 = np.cross(u, n_perp); n_perp2 = n_perp2/np.linalg.norm(n_perp2)
    u_new = ct_sc*u + sin_t*(math.cos(phi)*n_perp + math.sin(phi)*n_perp2)
    return u_new/np.linalg.norm(u_new)

def compton_electron_dir(u_in, ct_sc, E_in, E_sc, phi):
    u_out = rotate_dir(u_in, ct_sc, phi)
    vec = E_in*np.asarray(u_in) - E_sc*u_out
    norm = np.linalg.norm(vec)
    if norm < 1e-10: return u_in.copy(), u_out
    return vec/norm, u_out

def multiple_scatter_angle(E, s):
    if E < 0.02: return 1.0
    beta_pc = math.sqrt(E*(E+2*mec2))
    X0 = 36.1; x_over_X0 = max(s/X0, 1e-6)
    theta_0 = 13.6/beta_pc*math.sqrt(x_over_X0)*(1+0.038*math.log(x_over_X0))
    theta = abs(np.random.normal(0, theta_0))
    return math.cos(theta)

# ===== SCORING =====
def make_scoring():
    return {
        'ph_prim_fluence': 0.0, 'ph_scat_fluence': 0.0,
        'ph_prim_spec': np.zeros(28), 'ph_scat_spec': np.zeros(28),
        'ph_prim_ang': np.zeros(10), 'ph_scat_ang': np.zeros(10),
        'cyl_int': np.zeros(4), 'cyl_int_prim': np.zeros(4), 'cyl_int_scat': np.zeros(4),
        'cyl_int_energy_transfer': np.zeros(4),
        'electrons_cylinder': [],  # (T_e, cos_e, parent_primary)
        'electron_sources': [],  # (pos, dir, T, parent_primary, process)
    }

# ===== PHOTON TRANSPORT WITH ELECTRON SOURCE COLLECTION =====
elec_collect_radius = 2.5  # cm around cylinder center

def transport_photon_full(p_start, u_start, E_start, primary_start, scoring):
    electrons_in_cyl = []
    p = p_start.copy(); u = u_start.copy(); E = E_start; primary = primary_start
    
    for _ in range(300):
        if E <= 1e-4: break
        mu = float(np.interp(E, E_lut, mu_lut))
        if mu <= 0: break
        s_interact = -math.log(np.random.rand())/mu
        s_boundary = 1e30
        if u[2] > 0: s_boundary = min(s_boundary, (phantom_depth-p[2])/u[2])
        elif u[2] < 0: s_boundary = min(s_boundary, (-p[2])/u[2])
        if u[0] > 0: s_boundary = min(s_boundary, (phantom_half-p[0])/u[0])
        elif u[0] < 0: s_boundary = min(s_boundary, (-phantom_half-p[0])/u[0])
        if u[1] > 0: s_boundary = min(s_boundary, (phantom_half-p[1])/u[1])
        elif u[1] < 0: s_boundary = min(s_boundary, (-phantom_half-p[1])/u[1])
        s_step = min(s_interact, s_boundary)
        if s_step < 0: s_step = 0
        
        L = segment_cylinder_track(p, u, s_step)
        if L > 1e-12:
            if primary:
                scoring['ph_prim_fluence'] += L
                idx = np.searchsorted(photon_energy_edges, E, side='right')-1
                if 0 <= idx < 28: scoring['ph_prim_spec'][idx] += L
                idx_a = int((u[2]+1)/0.2); idx_a = min(9,max(0,idx_a))
                scoring['ph_prim_ang'][idx_a] += L
            else:
                scoring['ph_scat_fluence'] += L
                idx = np.searchsorted(photon_energy_edges, E, side='right')-1
                if 0 <= idx < 28: scoring['ph_scat_spec'][idx] += L
                idx_a = int((u[2]+1)/0.2); idx_a = min(9,max(0,idx_a))
                scoring['ph_scat_ang'][idx_a] += L
        
        if s_step >= s_boundary - 1e-12: break
        p = p + s_step*u
        in_cyl = is_in_cylinder(p[0],p[1],p[2])
        dist_to_cyl = math.sqrt(p[0]**2 + p[1]**2 + (p[2]-depth_poi)**2)
        near_cyl = dist_to_cyl < elec_collect_radius
        
        idx_mu = min(max(np.searchsorted(E_lut, E),1), len(E_lut)-1)
        mc = mu_comp_lut[idx_mu]; mpe = mu_pe_lut[idx_mu]
        mpair = mu_pair_lut[idx_mu]; mray = mu_ray_lut[idx_mu]
        mtot = mu_lut[idx_mu]
        r = np.random.rand()*mtot
        
        if r < mc:
            proc=0
            ct_sc, E_sc = sample_compton_fast(E)
            T_e = E - E_sc
            phi = np.random.rand()*2*np.pi
            u_e, u_out = compton_electron_dir(u, ct_sc, E, E_sc, phi)
            u = u_out; E_new = E_sc
            if in_cyl:
                scoring['cyl_int'][proc]+=1
                if primary: scoring['cyl_int_prim'][proc]+=1
                else: scoring['cyl_int_scat'][proc]+=1
                scoring['cyl_int_energy_transfer'][proc]+=T_e
                electrons_in_cyl.append((T_e, u_e[2], primary))
            if near_cyl and T_e > 0.010:
                scoring['electron_sources'].append((p.copy(), u_e.copy(), T_e, primary))
            E = E_new
            primary = False
        elif r < mc+mpe:
            proc=1
            T_e = E - 0.0005
            if in_cyl:
                scoring['cyl_int'][proc]+=1
                if primary: scoring['cyl_int_prim'][proc]+=1
                else: scoring['cyl_int_scat'][proc]+=1
                scoring['cyl_int_energy_transfer'][proc]+=T_e
                cos_e = 2*np.random.rand()-1
                electrons_in_cyl.append((T_e, cos_e, primary))
            if near_cyl and T_e > 0.010:
                cos_e = 2*np.random.rand()-1
                u_e = np.array([math.sqrt(1-cos_e**2)*math.cos(np.random.rand()*2*math.pi),
                                math.sqrt(1-cos_e**2)*math.sin(np.random.rand()*2*math.pi), cos_e])
                scoring['electron_sources'].append((p.copy(), u_e.copy(), T_e, primary))
            break
        elif r < mc+mpe+mpair:
            proc=2
            T_total = E - 2*mec2
            if in_cyl:
                scoring['cyl_int'][proc]+=1
                if primary: scoring['cyl_int_prim'][proc]+=1
                else: scoring['cyl_int_scat'][proc]+=1
                scoring['cyl_int_energy_transfer'][proc]+=T_total
                if T_total > 0:
                    f = np.random.rand()
                    electrons_in_cyl.append((f*T_total, u[2]*0.98, primary))
                    electrons_in_cyl.append(((1-f)*T_total, u[2]*0.98, primary))
            if near_cyl and T_total > 0.010:
                f = np.random.rand()
                # pair roughly forward
                u_e1 = rotate_dir(u, 0.98, np.random.rand()*2*np.pi)
                u_e2 = rotate_dir(u, 0.98, np.random.rand()*2*np.pi)
                scoring['electron_sources'].append((p.copy(), u_e1.copy(), f*T_total, primary))
                scoring['electron_sources'].append((p.copy(), u_e2.copy(), (1-f)*T_total, primary))
            # annihilation photons
            theta1 = np.arccos(2*np.random.rand()-1)
            phi1 = np.random.rand()*2*np.pi
            u_ann1 = np.array([math.sin(theta1)*math.cos(phi1), math.sin(theta1)*math.sin(phi1), math.cos(theta1)])
            transport_annihilation(p, u_ann1, scoring)
            transport_annihilation(p, -u_ann1, scoring)
            break
        else:
            proc=3
            if in_cyl:
                scoring['cyl_int'][proc]+=1
                if primary: scoring['cyl_int_prim'][proc]+=1
                else: scoring['cyl_int_scat'][proc]+=1
            ct_sc = math.cos(np.random.normal(0, 0.002))
            phi = np.random.rand()*2*np.pi
            u = rotate_dir(u, ct_sc, phi)
            primary = False
    return electrons_in_cyl

def transport_annihilation(p_start, u_start, scoring):
    p = p_start.copy(); u = u_start.copy(); E = 0.511
    for _ in range(100):
        if E <= 1e-4: break
        mu = float(np.interp(E, E_lut, mu_lut))
        if mu <= 0: break
        s_interact = -math.log(np.random.rand())/mu
        s_boundary = 1e30
        if u[2] > 0: s_boundary = min(s_boundary, (phantom_depth-p[2])/u[2])
        elif u[2] < 0: s_boundary = min(s_boundary, (-p[2])/u[2])
        if u[0] > 0: s_boundary = min(s_boundary, (phantom_half-p[0])/u[0])
        elif u[0] < 0: s_boundary = min(s_boundary, (-phantom_half-p[0])/u[0])
        if u[1] > 0: s_boundary = min(s_boundary, (phantom_half-p[1])/u[1])
        elif u[1] < 0: s_boundary = min(s_boundary, (-phantom_half-p[1])/u[1])
        s_step = min(s_interact, s_boundary)
        
        L = segment_cylinder_track(p, u, s_step)
        if L > 1e-12:
            scoring['ph_scat_fluence'] += L
            idx = np.searchsorted(photon_energy_edges, E, side='right')-1
            if 0 <= idx < 28: scoring['ph_scat_spec'][idx] += L
            idx_a = int((u[2]+1)/0.2); idx_a = min(9,max(0,idx_a))
            scoring['ph_scat_ang'][idx_a] += L
        
        if s_step >= s_boundary - 1e-12: break
        p = p + s_step*u
        in_cyl = is_in_cylinder(p[0],p[1],p[2])
        dist_to_cyl = math.sqrt(p[0]**2+p[1]**2+(p[2]-depth_poi)**2)
        near_cyl = dist_to_cyl < elec_collect_radius
        
        idx_mu = min(max(np.searchsorted(E_lut,E),1),len(E_lut)-1)
        mc=mu_comp_lut[idx_mu]; mpe=mu_pe_lut[idx_mu]; mpair=mu_pair_lut[idx_mu]; mray=mu_ray_lut[idx_mu]; mtot=mu_lut[idx_mu]
        r = np.random.rand()*mtot
        if r < mc:
            ct_sc, E_sc = sample_compton_fast(E)
            T_e = E - E_sc
            phi = np.random.rand()*2*np.pi
            u_e, u_out = compton_electron_dir(u, ct_sc, E, E_sc, phi)
            u = u_out; E = E_sc
            if in_cyl:
                scoring['cyl_int'][0]+=1; scoring['cyl_int_scat'][0]+=1
                scoring['cyl_int_energy_transfer'][0]+=T_e
            if near_cyl and T_e > 0.010:
                scoring['electron_sources'].append((p.copy(), u_e.copy(), T_e, False))
        elif r < mc+mpe:
            if in_cyl:
                scoring['cyl_int'][1]+=1; scoring['cyl_int_scat'][1]+=1
                T_e = E - 0.0005
                scoring['cyl_int_energy_transfer'][1]+=T_e
            if near_cyl:
                T_e = E - 0.0005
                cos_e = 2*np.random.rand()-1
                u_e = np.array([math.sqrt(1-cos_e**2)*math.cos(np.random.rand()*2*math.pi),
                                math.sqrt(1-cos_e**2)*math.sin(np.random.rand()*2*math.pi), cos_e])
                if T_e > 0.010:
                    scoring['electron_sources'].append((p.copy(), u_e.copy(), T_e, False))
            break
        elif r < mc+mpe+mpair:
            if in_cyl:
                scoring['cyl_int'][2]+=1; scoring['cyl_int_scat'][2]+=1
                T_total = E - 2*mec2
                scoring['cyl_int_energy_transfer'][2]+=T_total
            if near_cyl:
                T_total = E - 2*mec2
                if T_total > 0.010:
                    f = np.random.rand()
                    u_e1 = rotate_dir(u, 0.98, np.random.rand()*2*np.pi)
                    u_e2 = rotate_dir(u, 0.98, np.random.rand()*2*np.pi)
                    scoring['electron_sources'].append((p.copy(), u_e1.copy(), f*T_total, False))
                    scoring['electron_sources'].append((p.copy(), u_e2.copy(), (1-f)*T_total, False))
            break
        else:
            if in_cyl:
                scoring['cyl_int'][3]+=1; scoring['cyl_int_scat'][3]+=1
            ct_sc = math.cos(np.random.normal(0, 0.002))
            phi = np.random.rand()*2*np.pi
            u = rotate_dir(u, ct_sc, phi)
    return

# ===== ELECTRON TRANSPORT =====
def make_elec_scoring():
    return {'e_fluence': 0.0, 'e_spec': np.zeros(25), 'e_ang': np.zeros(10), 'e_dose': 0.0,
            'delta_sources': []}

def transport_electron(p0, u0, T0, elec_scoring, is_delta=False, ds=0.02):
    p = p0.copy(); u = u0.copy(); T = T0
    if T < 0.010: return
    range_max = R_csda_w(T)
    max_steps = int(1.5*range_max/ds) + 50
    if max_steps > 8000: max_steps = 8000
    
    for _ in range(max_steps):
        if T < 0.010: break
        s = min(ds, range_max)
        L = segment_cylinder_track(p, u, s)
        if L > 1e-12:
            elec_scoring['e_fluence'] += L
            idx = np.searchsorted(electron_energy_edges, T, side='right')-1
            if 0 <= idx < 25: elec_scoring['e_spec'][idx] += L
            idx_a = int((u[2]+1)/0.2); idx_a = min(9,max(0,idx_a))
            elec_scoring['e_ang'][idx_a] += L
            elec_scoring['e_dose'] += L*S_col_w(T)
            if is_delta:
                elec_scoring['delta_fluence'] += L
        new_p = p + s*u
        if new_p[2] < 0 or new_p[2] > phantom_depth: break
        if abs(new_p[0]) > phantom_half or abs(new_p[1]) > phantom_half: break
        p = new_p
        dE = S_total_w_local(T)*s
        T_new = T - dE
        if T_new < 0.010: break
        T = T_new
        ct_scat = multiple_scatter_angle(T, s)
        phi = np.random.rand()*2*np.pi
        u = rotate_dir(u, ct_scat, phi)
    return

# Actually I need S_total_w defined
def S_total_w(E):
    return S_col_w(E) + S_rad_w(E)
def S_total_w_local(E):
    return S_col_w(E) + S_rad_w(E)

# ===== MAIN MC =====
def run_full_mc(n_photons, seed=123, verbose=True):
    np.random.seed(seed)
    scoring = make_scoring()
    elec_scoring = make_elec_scoring()
    elec_scoring['delta_fluence'] = 0.0  # track delta ray fluence separately
    
    t0 = time.time()
    for i in range(n_photons):
        x_t = (np.random.rand()-0.5)*10
        y_t = (np.random.rand()-0.5)*10
        norm = math.sqrt(x_t*x_t+y_t*y_t+SSD*SSD)
        u = np.array([x_t,y_t,SSD])/norm
        p_surf = np.array([x_t,y_t,0.0])
        electrons_in_cyl = transport_photon_full(p_surf, u, 2.0, True, scoring)
        scoring['electrons_cylinder'].extend(electrons_in_cyl)
        if verbose and (i+1)%100000 == 0:
            print(f"  {i+1}/{n_photons}, {time.time()-t0:.1f}s, elec_sources={len(scoring['electron_sources'])}")
    
    # Now transport all electron sources
    print(f"\nTransporting {len(scoring['electron_sources'])} electron sources...")
    t1 = time.time()
    n_delta = 0
    for (p, u, T, parent_prim) in scoring['electron_sources']:
        transport_electron(p, u, T, elec_scoring, is_delta=False)
    
    print(f"Electron transport done in {time.time()-t1:.1f}s")
    print(f"Electron fluence track: {elec_scoring['e_fluence']:.2f} cm")
    print(f"Electron dose: {elec_scoring['e_dose']:.3f} MeV")
    print(f"Delta fluence: {elec_scoring['delta_fluence']:.3f} cm")
    
    return scoring, elec_scoring

# Quick test with 20k photons
scoring_t, elec_t = run_full_mc(20000, seed=42)
print(f"\nTest results:")
print(f"  Prim fluence track: {scoring_t['ph_prim_fluence']:.1f}")
print(f"  Scat fluence track: {scoring_t['ph_scat_fluence']:.1f}")
print(f"  Electrons in cylinder: {len(scoring_t['electrons_cylinder'])}")
print(f"  Electron sources: {len(scoring_t['electron_sources'])}")
print(f"  Cylinder interactions: {scoring_t['cyl_int']}")
print(f"  Electron fluence: {elec_t['e_fluence']:.3f}")
print(f"  Electron dose: {elec_t['e_dose']:.3f} MeV")