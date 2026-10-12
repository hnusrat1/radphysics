import numpy as np
from scipy.interpolate import interp1d
import json
import sys

np.random.seed(42)

# ============================================================
# Physical constants
# ============================================================
m_e_c2 = 0.51099895  # MeV
r_e = 2.8179403262e-13  # cm
N_A = 6.02214076e23

# ============================================================
# Cross sections for water (NIST-based, approximated)
# ============================================================
energy_pts = np.array([1.0, 1.1, 1.1732, 1.2, 1.25, 1.3, 1.3325, 1.4, 1.5])
mu_rho_total = np.array([0.07070, 0.06780, 0.06590, 0.06520, 0.06400, 0.06280, 0.06220, 0.06080, 0.05780])
mu_rho_comp = np.array([0.06950, 0.06650, 0.06430, 0.06350, 0.06200, 0.06060, 0.06000, 0.05850, 0.05550])
mu_rho_photo = np.array([0.00040, 0.00030, 0.00025, 0.00022, 0.00020, 0.00017, 0.00015, 0.00013, 0.00010])
mu_rho_pair = np.array([0.0, 0.0, 0.00020, 0.00028, 0.00050, 0.00080, 0.00100, 0.00130, 0.00150])
mu_rho_rayl = np.array([0.00080, 0.00068, 0.00060, 0.00055, 0.00050, 0.00043, 0.00040, 0.00035, 0.00030])

f_mu_total = interp1d(energy_pts, mu_rho_total, kind='cubic', bounds_error=False, fill_value='extrapolate')
f_mu_comp = interp1d(energy_pts, mu_rho_comp, kind='cubic', bounds_error=False, fill_value='extrapolate')
f_mu_photo = interp1d(energy_pts, mu_rho_photo, kind='cubic', bounds_error=False, fill_value='extrapolate')
f_mu_pair = interp1d(energy_pts, mu_rho_pair, kind='cubic', bounds_error=False, fill_value='extrapolate')
f_mu_rayl = interp1d(energy_pts, mu_rho_rayl, kind='cubic', bounds_error=False, fill_value='extrapolate')

def mu_total(E):
    return float(f_mu_total(np.clip(E, 0.01, 10.0)))
def mu_comp(E):
    return float(f_mu_comp(np.clip(E, 0.01, 10.0)))
def mu_photo(E):
    return float(f_mu_photo(np.clip(E, 0.01, 10.0)))
def mu_pair(E):
    return float(f_mu_pair(np.clip(E, 0.01, 10.0)))
def mu_rayl(E):
    return float(f_mu_rayl(np.clip(E, 0.01, 10.0)))

# ============================================================
# Geometry
# ============================================================
SSD = 100.0
z_surface = 0.0
field_half = 5.0
z_scoring_center = 5.0
scoring_half_thick = 0.1
z_scoring_min = z_scoring_center - scoring_half_thick
z_scoring_max = z_scoring_center + scoring_half_thick
scoring_radius = 1.0
src_pos = np.array([0.0, 0.0, -SSD])

# Electron density of water
n_e = 10 * N_A * 1.0 / 18.015  # e-/cm^3

# ============================================================
# Physics functions
# ============================================================

def compton_energy(E_in, cos_theta):
    return E_in / (1 + (E_in / m_e_c2) * (1 - cos_theta))

def klein_nishina_dcs(E_in, cos_theta):
    theta = np.arccos(np.clip(cos_theta, -1, 1))
    E_out = compton_energy(E_in, cos_theta)
    return (r_e**2 / 2) * (E_out / E_in)**2 * (E_in/E_out + E_out/E_in - np.sin(theta)**2)

def sample_source_photon():
    E = 1.1732 if np.random.random() < 0.5 else 1.3325
    x_surf = np.random.uniform(-field_half, field_half)
    y_surf = np.random.uniform(-field_half, field_half)
    dx = x_surf - src_pos[0]
    dy = y_surf - src_pos[1]
    dz = z_surface - src_pos[2]
    norm = np.sqrt(dx**2 + dy**2 + dz**2)
    direction = np.array([dx, dy, dz]) / norm
    return E, direction

def sample_compton(E_in):
    """Sample Compton scatter. Returns E_out, cos_th, t_e, cos_elect."""
    max_dcs = r_e**2 * 1.1
    while True:
        cos_th = np.random.uniform(-1, 1)
        dcs = klein_nishina_dcs(E_in, cos_th)
        if np.random.random() * max_dcs < dcs:
            break
    E_out = compton_energy(E_in, cos_th)
    t_e = E_in - E_out
    num = E_in - E_out * cos_th
    den = np.sqrt(E_in**2 + E_out**2 - 2*E_in*E_out*cos_th)
    cos_el = num / den if den > 0 else 1.0
    return E_out, cos_th, t_e, cos_el

def random_rotation(dir_in, cos_theta):
    u = dir_in / np.linalg.norm(dir_in)
    theta = np.arccos(np.clip(cos_theta, -1, 1))
    phi = np.random.uniform(0, 2*np.pi)
    if abs(u[2]) < 0.999:
        v = np.cross(u, np.array([0, 0, 1]))
    else:
        v = np.cross(u, np.array([1, 0, 0]))
    v = v / np.linalg.norm(v)
    w = np.cross(u, v)
    dir_out = u * cos_theta + v * np.sin(theta) * np.cos(phi) + w * np.sin(theta) * np.sin(phi)
    return dir_out / np.linalg.norm(dir_out)

def ray_cylinder_intersection(origin, direction, radius, z_min, z_max):
    """Find intersection of ray with cylinder on z-axis."""
    d = direction / np.linalg.norm(direction)
    if d[2] > 0:
        t_z_enter = (z_min - origin[2]) / d[2]
        t_z_exit = (z_max - origin[2]) / d[2]
    elif d[2] < 0:
        t_z_enter = (z_max - origin[2]) / d[2]
        t_z_exit = (z_min - origin[2]) / d[2]
    else:
        if origin[2] < z_min or origin[2] > z_max:
            return None
        t_z_enter = -np.inf
        t_z_exit = np.inf
    
    a = d[0]**2 + d[1]**2
    if a < 1e-15:
        if origin[0]**2 + origin[1]**2 <= radius**2:
            return (t_z_enter, t_z_exit) if t_z_enter < t_z_exit else (t_z_exit, t_z_enter)
        return None
    
    b = 2 * (origin[0]*d[0] + origin[1]*d[1])
    c = origin[0]**2 + origin[1]**2 - radius**2
    disc = b**2 - 4*a*c
    if disc < 0:
        return None
    sqrt_disc = np.sqrt(disc)
    t1 = (-b - sqrt_disc) / (2*a)
    t2 = (-b + sqrt_disc) / (2*a)
    if t1 > t2:
        t1, t2 = t2, t1
    
    t_enter = max(t_z_enter, t1)
    t_exit = min(t_z_exit, t2)
    
    if t_enter >= t_exit or t_exit < 0:
        return None
    return (t_enter, t_exit)

def sample_interaction_type(E):
    mu_c = mu_comp(E)
    mu_p = mu_photo(E)
    mu_pp = mu_pair(E)
    mu_r = mu_rayl(E)
    mu_tot = mu_c + mu_p + mu_pp + mu_r
    r = np.random.random() * mu_tot
    if r < mu_c:
        return 'compton'
    elif r < mu_c + mu_p:
        return 'photoelectric'
    elif r < mu_c + mu_p + mu_pp:
        return 'pair'
    else:
        return 'rayleigh'

# ============================================================
# Main simulation
# ============================================================

def compute_primary_rate(n_mc=500000):
    """Compute primary photon interaction rate in scoring volume per source photon."""
    total_w = 0.0
    by_proc = {'compton': 0.0, 'photoelectric': 0.0, 'pair': 0.0, 'rayleigh': 0.0}
    
    for _ in range(n_mc):
        E, dir_in = sample_source_photon()
        dz = dir_in[2]
        if dz <= 0:
            continue
        
        t_to_scoring = (z_scoring_min - z_surface) / dz
        if t_to_scoring < 0:
            continue
        
        entry_pt = np.array([0, 0, z_surface]) + t_to_scoring * dir_in
        if np.sqrt(entry_pt[0]**2 + entry_pt[1]**2) > scoring_radius:
            continue
        
        path_in_scoring = 0.2 / dz
        
        mu_e = mu_total(E)
        atten = np.exp(-mu_e * t_to_scoring)
        p_int = 1 - np.exp(-mu_e * path_in_scoring)
        
        w = atten * p_int
        
        mu_c = mu_comp(E)
        mu_p = mu_photo(E)
        mu_pp = mu_pair(E)
        mu_r = mu_rayl(E)
        mu_tot = mu_c + mu_p + mu_pp + mu_r
        
        total_w += w
        by_proc['compton'] += w * mu_c / mu_tot
        by_proc['photoelectric'] += w * mu_p / mu_tot
        by_proc['pair'] += w * mu_pp / mu_tot
        by_proc['rayleigh'] += w * mu_r / mu_tot
    
    total_w /= n_mc
    for k in by_proc:
        by_proc[k] /= n_mc
    
    return total_w, by_proc

def compute_scatter_rate(n_mc=500000):
    """Compute scattered photon interaction rate in scoring volume per source photon."""
    total_w = 0.0
    by_proc = {'compton': 0.0, 'photoelectric': 0.0, 'pair': 0.0, 'rayleigh': 0.0}
    
    for _ in range(n_mc):
        E, dir_in = sample_source_photon()
        dz = dir_in[2]
        if dz <= 0:
            continue
        
        mu_e = mu_total(E)
        t_max = 30.0 / dz
        
        C_samp = 1 - np.exp(-mu_e * t_max)
        if C_samp <= 0:
            continue
        u = np.random.random()
        t_scatter = -np.log(1 - u * C_samp) / mu_e
        
        scatter_pt = np.array([0, 0, z_surface]) + t_scatter * dir_in
        if abs(scatter_pt[0]) > 15 or abs(scatter_pt[1]) > 15:
            continue
        
        atten_to_scatter = np.exp(-mu_e * t_scatter)
        pdf_scatter = mu_e * np.exp(-mu_e * t_scatter) / C_samp
        
        E_out, cos_th, t_e, cos_el = sample_compton(E)
        dir_out = random_rotation(dir_in, cos_th)
        
        dz_out = dir_out[2]
        if dz_out <= 0:
            continue
        
        it = ray_cylinder_intersection(scatter_pt, dir_out, scoring_radius, z_scoring_min, z_scoring_max)
        if it is None:
            continue
        
        t_enter_vol, t_exit_vol = it
        if t_enter_vol < 0:
            t_enter_vol = 0
        path_in_vol = t_exit_vol - t_enter_vol
        if path_in_vol <= 0:
            continue
        
        mu_out = mu_total(E_out)
        atten_to_vol = np.exp(-mu_out * t_enter_vol)
        p_int_vol = 1 - np.exp(-mu_out * path_in_vol)
        
        dcs = klein_nishina_dcs(E, cos_th)
        dist_to_vol = t_enter_vol
        solid_angle = np.pi * scoring_radius**2 / (dist_to_vol**2) if dist_to_vol > 0.1 else 0
        
        pdf_angle = 0.5
        
        w = (n_e * dcs * solid_angle * atten_to_scatter * atten_to_vol * p_int_vol) / (pdf_scatter * pdf_angle)
        
        total_w += w
        
        mu_c = mu_comp(E_out)
        mu_p = mu_photo(E_out)
        mu_pp = mu_pair(E_out)
        mu_r = mu_rayl(E_out)
        mu_tot = mu_c + mu_p + mu_pp + mu_r
        
        by_proc['compton'] += w * mu_c / mu_tot
        by_proc['photoelectric'] += w * mu_p / mu_tot
        by_proc['pair'] += w * mu_pp / mu_tot
        by_proc['rayleigh'] += w * mu_r / mu_tot
    
    total_w /= n_mc
    for k in by_proc:
        by_proc[k] /= n_mc
    
    return total_w, by_proc

# Compute rates
print("Computing primary interaction rate...")
prim_total, prim_by_proc = compute_primary_rate(500000)
print(f"Primary total: {prim_total:.10f}")
for k, v in prim_by_proc.items():
    print(f"  {k}: {v:.10f}")

print("\nComputing scattered interaction rate...")
scat_total, scat_by_proc = compute_scatter_rate(500000)
print(f"Scattered total: {scat_total:.10f}")
for k, v in scat_by_proc.items():
    print(f"  {k}: {v:.10f}")

total = prim_total + scat_total
print(f"\nGrand total: {total:.10f}")
print(f"Scatter/primary ratio: {scat_total/prim_total:.4f}")
print(f"Scatter fraction: {scat_total/total:.4f}")