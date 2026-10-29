import numpy as np
import math
import time

rho = 1.0
mec2 = 0.511

# ===== ELECTRON STOPPING POWER AND RANGE DATA FOR WATER (ICRU 49) =====
E_sp = np.array([0.010, 0.015, 0.020, 0.030, 0.050, 0.070, 0.100, 0.150, 0.200, 
                 0.300, 0.400, 0.500, 0.600, 0.700, 0.800, 0.900, 1.000, 
                 1.250, 1.500, 1.750, 2.000, 2.500, 3.000])
S_col = np.array([23.14, 16.87, 13.29, 9.67, 6.47, 5.10, 4.10, 3.25, 2.85,
                  2.40, 2.16, 2.03, 1.95, 1.90, 1.87, 1.86, 1.85,
                  1.86, 1.86, 1.89, 1.92, 2.00, 2.08])
S_rad = np.array([0.0, 0.0, 0.0, 0.0, 0.0, 0.0, 0.0, 0.0, 0.0,
                  0.0, 0.0, 0.0, 0.0, 0.0, 0.0, 0.0, 0.0156,
                  0.0296, 0.0477, 0.071, 0.099, 0.155, 0.220])
R_csda = np.array([0.000257, 0.000506, 0.000812, 0.00158, 0.00438, 0.00786, 0.0142, 0.0265, 0.0435,
                   0.0858, 0.135, 0.181, 0.235, 0.286, 0.342, 0.388, 0.437,
                   0.618, 0.712, 0.856, 1.01, 1.40, 1.80])

def S_col_water(E):
    E = np.clip(E, E_sp[0], E_sp[-1])
    return float(np.interp(E, E_sp, S_col)) * rho

def S_rad_water(E):
    E = np.clip(E, E_sp[0], E_sp[-1])
    return float(np.interp(E, E_sp, S_rad)) * rho

def S_total_water(E):
    return S_col_water(E) + S_rad_water(E)

def R_csda_water(T):
    if T <= E_sp[0]: return R_csda[0]
    if T >= E_sp[-1]: return R_csda[-1]
    logT = math.log(T)
    logE = np.log(E_sp)
    logR = np.log(R_csda)
    return math.exp(float(np.interp(logT, logE, logR)))

for T in [0.05, 0.1, 0.5, 1.0, 2.0]:
    print(f"T={T:.2f} MeV: S_col={S_col_water(T):.3f} MeV/cm, S_rad={S_rad_water(T):.5f} MeV/cm, R={R_csda_water(T):.4f} cm")

# === ELECTRON TRANSPORT ===
cyl_radius = 1.0
cyl_half_z = 0.1
cyl_z_min = 9.9
cyl_z_max = 10.1
cyl_volume = np.pi * cyl_radius**2 * 2*cyl_half_z
phantom_half = 15.0
phantom_depth = 30.0
electron_energy_edges = np.array([0.01, 0.05, 0.1, 0.15, 0.2, 0.25, 0.3, 0.35, 0.4, 0.45, 0.5, 0.55, 0.6, 0.65, 0.7, 0.75, 0.8, 0.85, 0.9, 0.95, 1, 1.05, 1.1, 1.15, 1.2, 1.4])

def is_in_cylinder(x, y, z):
    return (cyl_z_min <= z <= cyl_z_max) and (x*x + y*y <= cyl_radius*cyl_radius + 1e-9)

def segment_cylinder_track_e(p, u, s_max):
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

def multiple_scatter_angle(E, s):
    if E < 0.02: return 1.0
    beta_pc = math.sqrt(E*(E+2*mec2))
    X0 = 36.1
    x_over_X0 = max(s / X0, 1e-6)
    theta_0 = 13.6 / beta_pc * math.sqrt(x_over_X0) * (1 + 0.038*math.log(x_over_X0))
    theta = abs(np.random.normal(0, theta_0))
    return math.cos(theta)

def rotate_dir_e(u, ct_sc, phi):
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

def make_elec_scoring():
    return {'e_fluence': 0.0, 'e_spec': np.zeros(25), 'e_ang': np.zeros(10), 'e_dose': 0.0}

def transport_electron(p0, u0, T0, scoring_elec, ds=0.02):
    p = p0.copy()
    u = u0.copy()
    T = T0
    delta_rays = []
    
    if T < 0.010: return delta_rays
    range_max = R_csda_water(T)
    max_steps = int(1.5 * range_max / ds) + 50
    if max_steps > 5000: max_steps = 5000
    
    for _ in range(max_steps):
        if T < 0.010: break
        s = min(ds, range_max)
        L = segment_cylinder_track_e(p, u, s)
        if L > 1e-12:
            scoring_elec['e_fluence'] += L
            idx = np.searchsorted(electron_energy_edges, T, side='right') - 1
            if 0 <= idx < 25:
                scoring_elec['e_spec'][idx] += L
            idx_a = int((u[2]+1)/0.2); idx_a = min(9, max(0, idx_a))
            scoring_elec['e_ang'][idx_a] += L
            scoring_elec['e_dose'] += L * S_col_water(T)
        
        new_p = p + s*u
        if new_p[2] < 0 or new_p[2] > phantom_depth: break
        if abs(new_p[0]) > phantom_half or abs(new_p[1]) > phantom_half: break
        p = new_p
        
        dE = S_total_water(T) * s
        T_new = T - dE
        if T_new < 0.010: break
        T = T_new
        
        ct_scat = multiple_scatter_angle(T, s)
        phi = np.random.rand() * 2 * np.pi
        u = rotate_dir_e(u, ct_scat, phi)
    
    return delta_rays

# Test with electrons starting near cylinder
scoring_elec_test = make_elec_scoring()
t0 = time.time()
for i in range(200):
    p0 = np.array([0.0, 0.0, 9.8])
    u0 = np.array([0.0, 0.0, 1.0])
    deltas = transport_electron(p0, u0, 1.0, scoring_elec_test)
print(f"200 electrons ({time.time()-t0:.2f}s)")
print(f"Fluence track: {scoring_elec_test['e_fluence']:.2f} cm")
print(f"Dose: {scoring_elec_test['e_dose']:.2f} MeV")
print(f"Expected approximate: ~{200*0.2*some_factor:.0f} cm (if no scattering)")

# Test with electrons at angle
scoring_elec_test2 = make_elec_scoring()
for i in range(200):
    p0 = np.array([0.0, 0.0, 9.8])
    u0 = np.array([0.3, 0.0, math.sqrt(1-0.09)])
    transport_electron(p0, u0, 1.0, scoring_elec_test2)
print(f"\nAngled electrons: fluence=", scoring_elec_test2['e_fluence'])