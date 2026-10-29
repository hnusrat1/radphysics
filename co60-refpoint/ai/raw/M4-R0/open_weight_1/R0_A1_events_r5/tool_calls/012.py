import numpy as np
from scipy.interpolate import interp1d
import json

np.random.seed(42)

# === Constants ===
m_e = 0.51099895
r_e = 2.8179403262e-13
NA = 6.02214076e23

# === Cross sections for water ===
e_arr = np.array([1.0, 1.1, 1.1732, 1.2, 1.25, 1.3, 1.3325, 1.4, 1.5])
mu_tot_v = np.array([0.07070, 0.06780, 0.06590, 0.06520, 0.06400, 0.06280, 0.06220, 0.06080, 0.05780])
mu_comp_v = np.array([0.06950, 0.06650, 0.06430, 0.06350, 0.06200, 0.06060, 0.06000, 0.05850, 0.05550])
mu_phot_v = np.array([0.00040, 0.00030, 0.00025, 0.00022, 0.00020, 0.00017, 0.00015, 0.00013, 0.00010])
mu_pair_v = np.array([0.0, 0.0, 0.00020, 0.00028, 0.00050, 0.00080, 0.00100, 0.00130, 0.00150])
mu_rayl_v = np.array([0.00080, 0.00068, 0.00060, 0.00055, 0.00050, 0.00043, 0.00040, 0.00035, 0.00030])

f_tot = interp1d(e_arr, mu_tot_v, kind='cubic', bounds_error=False, fill_value='extrapolate')
f_comp = interp1d(e_arr, mu_comp_v, kind='cubic', bounds_error=False, fill_value='extrapolate')
f_phot = interp1d(e_arr, mu_phot_v, kind='cubic', bounds_error=False, fill_value='extrapolate')
f_pair = interp1d(e_arr, mu_pair_v, kind='cubic', bounds_error=False, fill_value='extrapolate')
f_rayl = interp1d(e_arr, mu_rayl_v, kind='cubic', bounds_error=False, fill_value='extrapolate')

def mu_t(E): return float(f_tot(np.clip(E, 0.01, 10.0)))
def mu_c(E): return float(f_comp(np.clip(E, 0.01, 10.0)))
def mu_ph(E): return float(f_phot(np.clip(E, 0.01, 10.0)))
def mu_pp(E): return float(f_pair(np.clip(E, 0.01, 10.0)))
def mu_rl(E): return float(f_rayl(np.clip(E, 0.01, 10.0)))

# === Geometry ===
SSD = 100.0
src = np.array([0.0, 0.0, -SSD])
field_half = 5.0
z_surf = 0.0
z_ctr = 5.0
z_half = 0.1
z_lo = z_ctr - z_half
z_hi = z_ctr + z_half
cyl_R = 1.0
scoring_ctr = np.array([0.0, 0.0, z_ctr])

# Pyramid solid angle
th_half = np.arctan(field_half / SSD)
Omega_pyr = 4 * np.arcsin(np.sin(th_half)**2)

# Electron density
n_e = 10.0 * NA / 18.015

print(f"Omega_pyr = {Omega_pyr:.6f} sr")
print(f"n_e = {n_e:.4e}")

# === Physics functions ===
def compton_E(E_in, costh):
    return E_in / (1.0 + (E_in / m_e) * (1.0 - costh))

def KN_dcs(E_in, costh):
    th = np.arccos(np.clip(costh, -1.0, 1.0))
    E_out = compton_E(E_in, costh)
    return (r_e**2 / 2.0) * (E_out/E_in)**2 * (E_in/E_out + E_out/E_in - np.sin(th)**2)

def KN_total_sigma(E):
    """Total Compton cross section per electron (cm^2/e-), integrated over all angles."""
    # Numerical integration of 2*pi * integral of DCS from -1 to 1
    n_pts = 1000
    ct = np.linspace(-1, 1, n_pts)
    dcs = np.array([KN_dcs(E, c) for c in ct])
    # Simpson's rule
    h = 2.0 / (n_pts - 1)
    integral = h/3 * (dcs[0] + dcs[-1] + 4*np.sum(dcs[1:-1:2]) + 2*np.sum(dcs[2:-1:2]))
    return 2.0 * np.pi * integral

def sample_compton(E_in):
    max_d = r_e**2 * 1.05
    while True:
        ct = np.random.uniform(-1.0, 1.0)
        d = KN_dcs(E_in, ct)
        if np.random.random() * max_d < d:
            break
    E_out = compton_E(E_in, ct)
    t_e = E_in - E_out
    num = E_in - E_out * ct
    den = np.sqrt(E_in**2 + E_out**2 - 2.0*E_in*E_out*ct)
    ce = num / den if den > 0 else 1.0
    return E_out, ct, t_e, ce

def random_rot(v, costh):
    u = v / np.linalg.norm(v)
    th = np.arccos(np.clip(costh, -1.0, 1.0))
    ph = np.random.uniform(0.0, 2.0*np.pi)
    if abs(u[2]) < 0.999:
        w = np.cross(u, np.array([0.0, 0.0, 1.0]))
    else:
        w = np.cross(u, np.array([1.0, 0.0, 0.0]))
    w = w / np.linalg.norm(w)
    vv = np.cross(w, u)
    out = u*costh + w*np.sin(th)*np.cos(ph) + vv*np.sin(th)*np.sin(ph)
    return out / np.linalg.norm(out)

def source_sample():
    E = 1.1732 if np.random.random() < 0.5 else 1.3325
    x = np.random.uniform(-field_half, field_half)
    y = np.random.uniform(-field_half, field_half)
    d = np.array([x - src[0], y - src[1], z_surf - src[2]])
    d = d / np.linalg.norm(d)
    return E, d

def intersect_cyl(ori, dir, R, zL, zH):
    d = dir / np.linalg.norm(dir)
    if d[2] > 0:
        t_zL = (zL - ori[2]) / d[2]
        t_zH = (zH - ori[2]) / d[2]
    elif d[2] < 0:
        t_zL = (zH - ori[2]) / d[2]
        t_zH = (zL - ori[2]) / d[2]
    else:
        if ori[2] < zL or ori[2] > zH:
            return None
        t_zL = -1e30
        t_zH = 1e30
    
    a = d[0]**2 + d[1]**2
    if a < 1e-15:
        if ori[0]**2 + ori[1]**2 <= R**2:
            return (t_zL, t_zH) if t_zL < t_zH else (t_zH, t_zL)
        return None
    
    b = 2.0 * (ori[0]*d[0] + ori[1]*d[1])
    c = ori[0]**2 + ori[1]**2 - R**2
    disc = b*b - 4.0*a*c
    if disc < 0:
        return None
    sq = np.sqrt(disc)
    t1 = (-b - sq) / (2.0*a)
    t2 = (-b + sq) / (2.0*a)
    if t1 > t2:
        t1, t2 = t2, t1
    
    t_ent = max(t_zL, t1)
    t_ext = min(t_zH, t2)
    if t_ent >= t_ext or t_ext < 0:
        return None
    return (t_ent, t_ext)

def sample_proc(E):
    mt = mu_c(E) + mu_ph(E) + mu_pp(E) + mu_rl(E)
    r = np.random.random() * mt
    if r < mu_c(E):
        return 'compton'
    elif r < mu_c(E) + mu_ph(E):
        return 'photoelectric'
    elif r < mu_c(E) + mu_ph(E) + mu_pp(E):
        return 'pair'
    else:
        return 'rayleigh'

# === Compute total Compton cross section for the two energies ===
sigma_c_117 = KN_total_sigma(1.1732)
sigma_c_133 = KN_total_sigma(1.3325)
print(f"\nTotal Compton sigma (cm^2/e-): σ(1.1732)={sigma_c_117:.4e}, σ(1.3325)={sigma_c_133:.4e}")
print(f"Expected μ_comp = n_e*σ: at 1.1732: {n_e*sigma_c_117:.6f}, at 1.3325: {n_e*sigma_c_133:.6f}")
print(f"From NIST: at 1.1732: {mu_c(1.1732):.6f}, at 1.3325: {mu_c(1.3325):.6f}")

# === Compute rates ===
# For scattered photons, use forced detection approach:
# Instead of sampling the scatter angle, compute the exact scattering angle
# needed for the photon to reach the scoring volume, then weight by the DCS.

# Primary rate
print("\nComputing primary rate...")
N_MC = 500000
prim_tot = 0.0
prim_by = {'compton':0.0, 'photoelectric':0.0, 'pair':0.0, 'rayleigh':0.0}

for _ in range(N_MC):
    E, d = source_sample()
    dz = d[2]
    if dz <= 0:
        continue
    
    t_sc = (z_lo - z_surf) / dz
    if t_sc < 0:
        continue
    
    ep = np.array([0.0, 0.0, z_surf]) + t_sc * d
    if np.sqrt(ep[0]**2 + ep[1]**2) > cyl_R:
        continue
    
    path = 0.2 / dz
    muE = mu_t(E)
    att = np.exp(-muE * t_sc)
    p_int = 1.0 - np.exp(-muE * path)
    w = att * p_int
    
    prim_tot += w
    mt = mu_c(E) + mu_ph(E) + mu_pp(E) + mu_rl(E)
    prim_by['compton'] += w * mu_c(E) / mt
    prim_by['photoelectric'] += w * mu_ph(E) / mt
    prim_by['pair'] += w * mu_pp(E) / mt
    prim_by['rayleigh'] += w * mu_rl(E) / mt

prim_tot /= N_MC
for k in prim_by:
    prim_by[k] /= N_MC

print(f"Primary: total={prim_tot:.8f}")
for k, v in prim_by.items():
    print(f"  {k}: {v:.10f}")

# Scatter rate using forced detection
print("\nComputing scatter rate (forced detection)...")
scat_tot = 0.0
scat_by = {'compton':0.0, 'photoelectric':0.0, 'pair':0.0, 'rayleigh':0.0}
n_scat_hits = 0

for _ in range(N_MC):
    E, d = source_sample()
    dz = d[2]
    if dz <= 0:
        continue
    
    muE = mu_t(E)
    t_max = 30.0 / dz
    C = 1.0 - np.exp(-muE * t_max)
    if C <= 0:
        continue
    
    # Sample scatter point
    t_scat = -np.log(1.0 - np.random.random() * C) / muE
    sp = np.array([0.0, 0.0, z_surf]) + t_scat * d
    
    if abs(sp[0]) > 15 or abs(sp[1]) > 15:
        continue
    
    pdf_scat = muE * np.exp(-muE * t_scat) / C
    
    # Compute the direction from scatter point to scoring volume center
    dir_to_vol = scoring_ctr - sp
    dist_to_vol = np.linalg.norm(dir_to_vol)
    if dist_to_vol < 0.01:
        continue
    dir_to_vol = dir_to_vol / dist_to_vol
    
    # Scatter angle needed
    costh_scat = np.dot(d, dir_to_vol)
    costh_scat = np.clip(costh_scat, -1.0, 1.0)
    
    # DCS at this angle
    dcs = KN_dcs(E, costh_scat)
    
    # Solid angle of scoring volume (approximate)
    # Area = π * R^2, projected area ≈ π * R^2 * |cos(θ_view)|
    # where θ_view is the angle between the scattered direction and the normal to the scoring volume
    # The scoring volume is a cylinder on the z-axis, so the normal to the circular face is along z
    # The scattered photon approaches at angle arccos(dir_to_vol[2]) from the z-axis
    # The projected area is π*R^2 * abs(dir_to_vol[2]) for small angles
    cos_view = abs(dir_to_vol[2])
    proj_area = np.pi * cyl_R**2 * max(cos_view, 0.01)
    solid_angle = proj_area / (dist_to_vol**2)
    
    # Attenuation of scattered photon
    # Check if ray from scatter point to scoring volume intersects the cylinder
    it = intersect_cyl(sp, dir_to_vol, cyl_R, z_lo, z_hi)
    if it is None:
        continue
    
    t_ent, t_ext = it
    if t_ent < 0:
        t_ent = 0
    path_v = t_ext - t_ent
    if path_v <= 0:
        continue
    
    mu_out = mu_t(E_out := compton_E(E, costh_scat))
    att_to_vol = np.exp(-mu_out * t_ent)
    p_int_v = 1.0 - np.exp(-mu_out * path_v)
    
    # Weight = contribution / PDF
    # Contribution: n_e * dcs * solid_angle * att_to_vol * p_int_v * (dt factor)
    # PDF: pdf_scat (per unit t)
    # The dt factor cancels
    w = n_e * dcs * solid_angle * att_to_vol * p_int_v / pdf_scat
    
    scat_tot += w
    mt_out = mu_c(E_out) + mu_ph(E_out) + mu_pp(E_out) + mu_rl(E_out)
    scat_by['compton'] += w * mu_c(E_out) / mt_out
    scat_by['photoelectric'] += w * mu_ph(E_out) / mt_out
    scat_by['pair'] += w * mu_pp(E_out) / mt_out
    scat_by['rayleigh'] += w * mu_rl(E_out) / mt_out
    n_scat_hits += 1

scat_tot /= N_MC
for k in scat_by:
    scat_by[k] /= N_MC

print(f"Scatter: total={scat_tot:.8f}, hits={n_scat_hits}/{N_MC}")
for k, v in scat_by.items():
    print(f"  {k}: {v:.10f}")

tot_rate = prim_tot + scat_tot
print(f"\nTotal rate: {tot_rate:.8f}")
print(f"Scatter/prim: {scat_tot/prim_tot:.4f}")
print(f"Scatter fraction: {scat_tot/tot_rate:.4f}")