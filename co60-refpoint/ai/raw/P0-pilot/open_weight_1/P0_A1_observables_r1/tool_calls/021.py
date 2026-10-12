import numpy as np
import math
import time

# ===== ELECTRON STOPPING POWER AND RANGE DATA FOR WATER (ICRU 49) =====
# Collision stopping power (MeV cm^2/g)
E_sp = np.array([0.010, 0.015, 0.020, 0.030, 0.050, 0.070, 0.100, 0.150, 0.200, 
                 0.300, 0.400, 0.500, 0.600, 0.700, 0.800, 0.900, 1.000, 
                 1.250, 1.500, 1.750, 2.000, 2.500, 3.000])
S_col = np.array([23.14, 16.87, 13.29, 9.67, 6.47, 5.10, 4.10, 3.25, 2.85,
                  2.40, 2.16, 2.03, 1.95, 1.90, 1.87, 1.86, 1.85,
                  1.86, 1.86, 1.89, 1.92, 2.00, 2.08])
# Radiative stopping power (MeV cm^2/g)
S_rad = np.array([0.0, 0.0, 0.0, 0.0, 0.0, 0.0, 0.0, 0.0, 0.0,
                  0.0, 0.0, 0.0, 0.0, 0.0, 0.0, 0.0, 0.0156,
                  0.0296, 0.0477, 0.071, 0.099, 0.155, 0.220])
# CSDA range (g/cm^2)
R_csda = np.array([0.000257, 0.000506, 0.000812, 0.00158, 0.00438, 0.00786, 0.0142, 0.0265, 0.0435,
                   0.0858, 0.135, 0.181, 0.235, 0.286, 0.342, 0.388, 0.437,
                   0.618, 0.712, 0.856, 1.01, 1.40, 1.80])

def S_col_water(E):
    """Collision stopping power (MeV/cm) for water"""
    E = np.clip(E, E_sp[0], E_sp[-1])
    return float(np.interp(E, E_sp, S_col)) * rho

def S_rad_water(E):
    """Radiative stopping power (MeV/cm) for water"""
    E = np.clip(E, E_sp[0], E_sp[-1])
    return float(np.interp(E, E_sp, S_rad)) * rho

def S_total_water(E):
    return S_col_water(E) + S_rad_water(E)

def R_csda_water(T):
    """CSDA range in cm for water (T in MeV)"""
    T = np.clip(T, R_csda[0], R_csda[-1] )
    # interpolate log-log
    if T < E_sp[0]:
        return R_csda[0]
    if T > E_sp[-1]:
        return R_csda[-1]
    logT = math.log(T)
    logE = np.log(E_sp)
    logR = np.log(R_csda)
    return math.exp(float(np.interp(logT, logE, logR)))

# Test
for T in [0.05, 0.1, 0.5, 1.0, 2.0]:
    print(f"T={T:.2f} MeV: S_col={S_col_water(T):.3f} MeV/cm, S_rad={S_rad_water(T):.4f} MeV/cm, R={R_csda_water(T):.4f} cm")

# ===== GEOMETRY =====
cyl_radius = 1.0
cyl_half_z = 0.1
cyl_z_min = 9.9
cyl_z_max = 10.1
cyl_volume = np.pi * cyl_radius**2 * 2*cyl_half_z
phantom_half = 15.0
phantom_depth = 30.0
electron_energy_edges = np.array([0.01, 0.05, 0.1, 0.15, 0.2, 0.25, 0.3, 0.35, 0.4, 0.45, 0.5, 0.55, 0.6, 0.65, 0.7, 0.75, 0.8, 0.85, 0.9, 0.95, 1, 1.05, 1.1, 1.15, 1.2, 1.4])
ct_edges = np.array([-1, -0.8, -0.6, -0.4, -0.2, 0, 0.2, 0.4, 0.6, 0.8, 1])

def is_in_cylinder(x, y, z):
    return (cyl_z_min <= z <= cyl_z_max) and (x*x + y*y <= cyl_radius*cyl_radius + 1e-9)

def segment_cylinder_track_e(p, u, s_max):
    """Same function as for photons but for electrons."""
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

# Multiple scattering: Gaussian approximation
def multiple_scatter_angle(E, s):
    """Sample polar scattering angle (cos_theta) for electron step of length s (cm)."""
    if E < 0.02:
        return 1.0  # no scattering for very low energy
    # Moliere: theta_0 = 13.6/(beta p c) * sqrt(s/X_0) * (1+0.038*ln(s/X_0))
    # beta p c = sqrt(T*(T+2*mec2)) in MeV
    beta_pc = math.sqrt(E*(E+2*0.511))
    X0 = 36.1  # cm (radiation length in water)
    x_over_X0 = s / X0
    if x_over_X0 <= 0: return 1.0
    theta_0 = 13.6 / beta_pc * math.sqrt(x_over_X0) * (1 + 0.038*math.log(max(x_over_X0, 1e-6)))
    # Sample Gaussian angle
    theta = abs(np.random.normal(0, theta_0))
    return math.cos(theta)

def transport_electron(p0, u0, T0, scoring_elec, ds=0.02):
    """Transport one electron (or positron treated as electron) with CSDA.
    Scores track length in cylinder, energy spectrum, angular dist.
    Returns list of delta rays created with energy > 10 keV."""
    p = p0.copy()
    u = u0.copy()
    T = T0
    delta_rays = []
    
    # Check if electron can reach the cylinder
    # If distance from p to cylinder > range, skip early (optimization)
    # For now, just transport
    
    max_steps = int(R_csda_water(T)/ds) + 500
    if max_steps > 20000:
        max_steps = 20000
    
    for _ in range(max_steps):
        if T < 0.010:  # below 10 keV cutoff
            break
        # Step length: limit by energy loss
        s = min(ds, R_csda_water(T))
        # Score current segment
        L = segment_cylinder_track_e(p, u, s)
        if L > 1e-12:
            scoring_elec['e_fluence'] += L
            idx = np.searchsorted(electron_energy_edges, T, side='right') - 1
            if 0 <= idx < 25:
                scoring_elec['e_spec'][idx] += L
            idx_a = int((u[2]+1)/0.2); idx_a = min(9, max(0, idx_a))
            scoring_elec['e_ang'][idx_a] += L
            scoring_elec['e_dose'] += L * S_col_water(T)  # collision energy deposited (MeV/cm * cm = MeV)
        
        # Move
        new_p = p + s*u
        # Check phantom boundary
        if new_p[2] < 0 or new_p[2] > phantom_depth: break
        if abs(new_p[0]) > phantom_half or abs(new_p[1]) > phantom_half: break
        p = new_p
        
        # Energy loss
        dE = S_total_water(T) * s
        T_new = T - dE
        if T_new < 0.010:
            # Deposit remaining energy
            break
        T = T_new
        
        # Multiple scattering deflection
        ct_scat = multiple_scatter_angle(T, s)
        phi = np.random.rand() * 2 * np.pi
        u = rotate_dir_e(u, ct_scat, phi)
    
    return delta_rays

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

# Test electron transport with a few electrons
scoring_elec_test = {'e_fluence': 0.0, 'e_spec': np.zeros(25), 'e_ang': np.zeros(10), 'e_dose': 0.0}
# Electron created at (0,0,9.8) going downward with 1 MeV
t0 = time.time()
for i in range(100):
    p0 = np.array([0.0, 0.0, 9.8])
    u0 = np.array([0.0, 0.0, 1.0])
    transport_electron(p0, u0, 1.0, scoring_elec_test)
print(f"100 electrons: {time.time()-t0:.2f}s")
print(f"Fluence track: {scoring_elec_test['e_fluence']:.2f} cm")
print(f"Dose: {scoring_elec_test['e_dose']:.2f} MeV")

# Check: 1 MeV electron created at z=9.8 going down (toward +z).
# It passes through the cylinder (z 9.9-10.1), with range 0.44 cm.
# Track through cylinder ≈ 0.2 cm per electron.
# 100 electrons should give about 20 cm of track if all pass through.
print(f"Expected if all pass through: ~{100*0.2:.0f} cm")
# But scattering may deflect some electrons out of the 1cm radius cylinder