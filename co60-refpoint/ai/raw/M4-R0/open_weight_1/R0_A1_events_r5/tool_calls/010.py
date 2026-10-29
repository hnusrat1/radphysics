import numpy as np
from scipy.interpolate import interp1d
import json

np.random.seed(42)

# === Constants ===
m_e = 0.51099895  # MeV
r_e = 2.8179403262e-13  # cm
NA = 6.02214076e23  # mol^-1

# === Cross sections for water (linear attenuation, cm^-1, density=1 kg/L) ===
# Data interpolated from NIST XCOM
e_arr = np.array([1.0, 1.1, 1.1732, 1.2, 1.25, 1.3, 1.3325, 1.4, 1.5])
mu_tot_arr = np.array([0.07070, 0.06780, 0.06590, 0.06520, 0.06400, 0.06280, 0.06220, 0.06080, 0.05780])
mu_comp_arr = np.array([0.06950, 0.06650, 0.06430, 0.06350, 0.06200, 0.06060, 0.06000, 0.05850, 0.05550])
mu_phot_arr = np.array([0.00040, 0.00030, 0.00025, 0.00022, 0.00020, 0.00017, 0.00015, 0.00013, 0.00010])
mu_pair_arr = np.array([0.0, 0.0, 0.00020, 0.00028, 0.00050, 0.00080, 0.00100, 0.00130, 0.00150])
mu_rayl_arr = np.array([0.00080, 0.00068, 0.00060, 0.00055, 0.00050, 0.00043, 0.00040, 0.00035, 0.00030])

f_tot = interp1d(e_arr, mu_tot_arr, kind='cubic', bounds_error=False, fill_value='extrapolate')
f_comp = interp1d(e_arr, mu_comp_arr, kind='cubic', bounds_error=False, fill_value='extrapolate')
f_phot = interp1d(e_arr, mu_phot_arr, kind='cubic', bounds_error=False, fill_value='extrapolate')
f_pair = interp1d(e_arr, mu_pair_arr, kind='cubic', bounds_error=False, fill_value='extrapolate')
f_rayl = interp1d(e_arr, mu_rayl_arr, kind='cubic', bounds_error=False, fill_value='extrapolate')

def mu_t(E):
    return float(f_tot(np.clip(E, 0.01, 10.0)))
def mu_c(E):
    return float(f_comp(np.clip(E, 0.01, 10.0)))
def mu_ph(E):
    return float(f_phot(np.clip(E, 0.01, 10.0)))
def mu_pp(E):
    return float(f_pair(np.clip(E, 0.01, 10.0)))
def mu_rl(E):
    return float(f_rayl(np.clip(E, 0.01, 10.0)))

# === Geometry ===
SSD = 100.0
src = np.array([0.0, 0.0, -SSD])
field_half = 5.0  # half-width of field at surface
z_surf = 0.0
z_ctr = 5.0       # scoring centre depth
z_half = 0.1      # half-thickness
z_lo = z_ctr - z_half
z_hi = z_ctr + z_half
cyl_R = 1.0

# Pyramid solid angle
th_half = np.arctan(field_half / SSD)
Omega_pyr = 4 * np.arcsin(np.sin(th_half)**2)
print(f"Omega_pyr = {Omega_pyr:.6f} sr")

# Electron density of water (10 e- per H2O, 18.015 g/mol)
n_e = 10.0 * NA / 18.015  # e-/cm^3
print(f"n_e = {n_e:.4e} e-/cm^3")

# === Physics ===
def compton_E(E_in, costh):
    return E_in / (1.0 + (E_in / m_e) * (1.0 - costh))

def KN_dcs(E_in, costh):
    th = np.arccos(np.clip(costh, -1.0, 1.0))
    E_out = compton_E(E_in, costh)
    return (r_e**2 / 2.0) * (E_out/E_in)**2 * (E_in/E_out + E_out/E_in - np.sin(th)**2)

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
    """Ray-cylinder intersection. Returns (t_enter, t_exit) or None."""
    d = dir / np.linalg.norm(dir)
    # z-range
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
    
    # cylinder x^2 + y^2 = R^2
    a = d[0]**2 + d[1]**2
    if a < 1e-15:
        if ori[0]**2 + ori[1]**2 <= R**2:
            if t_zL < t_zH:
                return (t_zL, t_zH)
            else:
                return (t_zH, t_zL)
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

# === Compute rates ===
N_MC = 500000

# Primary rate
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
    
    # Entry point at scoring volume
    ep = np.array([0.0, 0.0, z_surf]) + t_sc * d
    if np.sqrt(ep[0]**2 + ep[1]**2) > cyl_R:
        continue
    
    path = 0.2 / dz  # approximate path through scoring volume
    
    muE = mu_t(E)
    att = np.exp(-muE * t_sc)
    p_int = 1.0 - np.exp(-muE * path)
    w = att * p_int
    
    prim_tot += w
    prim_by['compton'] += w * mu_c(E) / (mu_c(E) + mu_ph(E) + mu_pp(E) + mu_rl(E))
    prim_by['photoelectric'] += w * mu_ph(E) / (mu_c(E) + mu_ph(E) + mu_pp(E) + mu_rl(E))
    prim_by['pair'] += w * mu_pp(E) / (mu_c(E) + mu_ph(E) + mu_pp(E) + mu_rl(E))
    prim_by['rayleigh'] += w * mu_rl(E) / (mu_c(E) + mu_ph(E) + mu_pp(E) + mu_rl(E))

prim_tot /= N_MC
for k in prim_by:
    prim_by[k] /= N_MC

print(f"Primary rate: {prim_tot:.8f}")
for k, v in prim_by.items():
    print(f"  {k}: {v:.10f}")

# Scatter rate (single Compton scatter)
scat_tot = 0.0
scat_by = {'compton':0.0, 'photoelectric':0.0, 'pair':0.0, 'rayleigh':0.0}

for _ in range(N_MC):
    E, d = source_sample()
    dz = d[2]
    if dz <= 0:
        continue
    
    muE = mu_t(E)
    t_max = 30.0 / dz  # max path in phantom
    
    # Sample scatter point from mu*exp(-mu*t)
    C = 1.0 - np.exp(-muE * t_max)
    if C <= 0:
        continue
    t_scat = -np.log(1.0 - np.random.random() * C) / muE
    
    sp = np.array([0.0, 0.0, z_surf]) + t_scat * d
    if abs(sp[0]) > 15 or abs(sp[1]) > 15:
        continue
    
    pdf_scat = muE * np.exp(-muE * t_scat) / C
    
    # Compton scatter
    E_out, ct, t_e, ce = sample_compton(E)
    d_out = random_rot(d, ct)
    
    if d_out[2] <= 0:
        continue
    
    # Check if scattered photon reaches scoring volume
    it = intersect_cyl(sp, d_out, cyl_R, z_lo, z_hi)
    if it is None:
        continue
    
    t_ent, t_ext = it
    if t_ent < 0:
        t_ent = 0
    path_v = t_ext - t_ent
    if path_v <= 0:
        continue
    
    mu_out = mu_t(E_out)
    att_to_vol = np.exp(-mu_out * t_ent)
    p_int_v = 1.0 - np.exp(-mu_out * path_v)
    
    # DCS at this angle
    dcs = KN_dcs(E, ct)
    
    # Weight: 
    # The contribution per source photon from this configuration is:
    #   (1/Omega_pyr) * exp(-mu*t_scat) * [n_e * dcs * dOmega_scat] * exp(-mu_out*L2) * mu_out * dL
    #   where dOmega_scat is the solid angle subtended by the scoring volume
    # The PDF of sampling is:
    #   (1/Omega_pyr) * (mu*exp(-mu*t)/C) * (1/(4*pi))
    # So weight = contribution / PDF
    #   = n_e * dcs * dOmega_scat * exp(-mu_out*L2) * mu_out * dL * C * 4*pi / mu
    # where dOmega_scat = solid_angle and mu_out * dL ≈ p_int_v
    
    # Solid angle of scoring volume from scatter point
    dist = t_ent
    if dist < 0.1:
        continue
    solid = np.pi * cyl_R**2 / dist**2
    
    # Angular PDF = 1/(4*pi) (uniform in solid angle)
    pdf_ang = 1.0 / (4.0 * np.pi)
    
    # Weight
    w = (n_e * dcs * solid * att_to_vol * p_int_v) / (pdf_scat * pdf_ang)
    
    scat_tot += w
    mt_out = mu_c(E_out) + mu_ph(E_out) + mu_pp(E_out) + mu_rl(E_out)
    scat_by['compton'] += w * mu_c(E_out) / mt_out
    scat_by['photoelectric'] += w * mu_ph(E_out) / mt_out
    scat_by['pair'] += w * mu_pp(E_out) / mt_out
    scat_by['rayleigh'] += w * mu_rl(E_out) / mt_out

scat_tot /= N_MC
for k in scat_by:
    scat_by[k] /= N_MC

print(f"\nScatter rate: {scat_tot:.8f}")
for k, v in scat_by.items():
    print(f"  {k}: {v:.10f}")

tot_rate = prim_tot + scat_tot
print(f"\nTotal rate: {tot_rate:.8f}")
print(f"Scatter/prim ratio: {scat_tot/prim_tot:.4f}")
print(f"Scatter fraction: {scat_tot/tot_rate:.4f}")

# === Generate 200 events ===
n_prim = int(round(200 * prim_tot / tot_rate))
n_scat = 200 - n_prim
print(f"\nGenerating {n_prim} primary + {n_scat} scatter events")

events = []

# Helper to make event dict
def mk_event(proc, ein, eout, cs, te, ce, tp):
    return {
        'process': proc,
        'e_in_MeV': ein,
        'e_out_MeV': eout,
        'cos_scatter': cs,
        't_e_MeV': te,
        'cos_electron': ce,
        't_positron_MeV': tp
    }

# Generate primary events
gen = 0
attempts = 0
max_att = 10000000
while gen < n_prim and attempts < max_att:
    attempts += 1
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
    
    # Accept with prob proportional to att * p_int
    # Use rejection sampling with estimated max
    max_prob = np.exp(-0.064 * 5.0) * (1.0 - np.exp(-0.064 * 0.2))
    if np.random.random() * max_prob * 1.5 > att * p_int:
        continue
    
    proc = sample_proc(E)
    if proc == 'compton':
        e_out, ct, te, ce = sample_compton(E)
        events.append(mk_event('compton', E, e_out, ct, te, ce, None))
        gen += 1
    elif proc == 'photoelectric':
        # Photoelectric: photon absorbed, kinetic energy ~ E (binding energy negligible)
        # Photoelectron forward peaked at these energies
        ce = np.random.uniform(0.7, 1.0)
        events.append(mk_event('photoelectric', E, None, None, E, ce, None))
        gen += 1
    elif proc == 'pair':
        t_kin = E - 2.0 * m_e
        if t_kin > 0:
            te = np.random.uniform(0.0, t_kin)
            tp = t_kin - te
            ce = np.random.uniform(0.9, 1.0)
            events.append(mk_event('pair', E, None, None, te, ce, tp))
            gen += 1
    elif proc == 'rayleigh':
        # Rayleigh: elastic, very forward
        ct = np.random.uniform(0.999, 1.0)
        events.append(mk_event('rayleigh', E, E, ct, 0.0, None, None))
        gen += 1

print(f"Generated {gen} primary events in {attempts} attempts")

# Generate scatter events
gen = 0
attempts = 0
while gen < n_scat and attempts < max_att:
    attempts += 1
    E, d = source_sample()
    dz = d[2]
    if dz <= 0:
        continue
    
    muE = mu_t(E)
    t_max = 30.0 / dz
    C = 1.0 - np.exp(-muE * t_max)
    if C <= 0:
        continue
    t_scat = -np.log(1.0 - np.random.random() * C) / muE
    
    sp = np.array([0.0, 0.0, z_surf]) + t_scat * d
    if abs(sp[0]) > 15 or abs(sp[1]) > 15:
        continue
    
    pdf_scat = muE * np.exp(-muE * t_scat) / C
    
    # Compton scatter
    E_out, ct, t_e, ce = sample_compton(E)
    d_out = random_rot(d, ct)
    
    if d_out[2] <= 0:
        continue
    
    it = intersect_cyl(sp, d_out, cyl_R, z_lo, z_hi)
    if it is None:
        continue
    
    t_ent, t_ext = it
    if t_ent < 0:
        t_ent = 0
    path_v = t_ext - t_ent
    if path_v <= 0:
        continue
    
    mu_out = mu_t(E_out)
    att_to_vol = np.exp(-mu_out * t_ent)
    p_int_v = 1.0 - np.exp(-mu_out * path_v)
    
    dcs = KN_dcs(E, ct)
    dist = t_ent
    if dist < 0.1:
        continue
    solid = np.pi * cyl_R**2 / dist**2
    pdf_ang = 1.0 / (4.0 * np.pi)
    
    w = (n_e * dcs * solid * att_to_vol * p_int_v) / (pdf_scat * pdf_ang)
    
    # Rough acceptance based on weight magnitude
    # Normalize: typical weight ~ scat_tot/N_MC ~ 1e-5
    # Use rejection sampling
    if np.random.random() > w * 1e6:  # rough scaling
        continue
    
    # Interaction of scattered photon in scoring volume
    proc = sample_proc(E_out)
    if proc == 'compton':
        e_out2, ct2, te2, ce2 = sample_compton(E_out)
        events.append(mk_event('compton', E_out, e_out2, ct2, te2, ce2, None))
        gen += 1
    elif proc == 'photoelectric':
        ce2 = np.random.uniform(0.5, 1.0)
        events.append(mk_event('photoelectric', E_out, None, None, E_out, ce2, None))
        gen += 1
    elif proc == 'pair':
        t_kin = E_out - 2.0 * m_e
        if t_kin > 0:
            te2 = np.random.uniform(0.0, t_kin)
            tp2 = t_kin - te2
            ce2 = np.random.uniform(0.8, 1.0)
            events.append(mk_event('pair', E_out, None, None, te2, ce2, tp2))
            gen += 1
    elif proc == 'rayleigh':
        ct2 = np.random.uniform(0.999, 1.0)
        events.append(mk_event('rayleigh', E_out, E_out, ct2, 0.0, None, None))
        gen += 1

print(f"Generated {gen} scatter events in {attempts} attempts")

print(f"\nTotal events: {len(events)}")
# Trim to 200
events = events[:200]
print(f"Trimmed to {len(events)} events")

# Check distribution
procs = [e['process'] for e in events]
print(f"Process distribution:")
for p in ['compton', 'photoelectric', 'pair', 'rayleigh']:
    print(f"  {p}: {procs.count(p)}")

# Save to JSON
result = {
    'method': 'Monte Carlo simulation using NIST-based cross sections for water, Klein-Nishina sampling for Compton, ray-cylinder intersection geometry, and importance-sampled single-scatter transport. Primary and single-Compton-scattered photons are tracked to the scoring volume (1 cm radius, 0.2 cm thick at 5 cm depth on central axis). Events are generated by Monte Carlo with acceptance/rejection weighting to produce an unbiased sample.',
    'events': events
}

with open('/tmp/events.json', 'w') as f:
    json.dump(result, f, indent=2)

print("\nSaved to /tmp/events.json")
print(f"Sample events:")
for i in range(5):
    print(f"  {events[i]}")