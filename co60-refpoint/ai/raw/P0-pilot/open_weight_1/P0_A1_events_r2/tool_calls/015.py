# Re-run sampling and analyze
# ...
# We'll just re-run the whole chain quickly

import numpy as np
import math
import random

m_e = 0.51099895
r_e_sq = 7.9408e-26
n_e = 3.3428e23

def mu_Compton(E):
    alpha = E / m_e
    t1 = (1+alpha)/alpha**2 * (2*(1+alpha)/(1+2*alpha) - math.log(1+2*alpha)/alpha)
    t2 = math.log(1+2*alpha) / (2*alpha)
    t3 = (1+3*alpha) / (1+2*alpha)**2
    sigma = 2*math.pi * r_e_sq * (t1 + t2 - t3)
    return n_e * sigma

def mu_Rayleigh(E):
    return 0.0008 * (1.0 / max(E, 0.01))**2

def mu_Photoelectric(E):
    return 1.9e-5 * (1.0 / max(E, 0.01))**3.5

def mu_Pair(E):
    if E <= 1.022:
        return 0.0
    if E < 2.0:
        return 0.0004 * (E - 1.022) / (2.0 - 1.022)
    elif E < 3.0:
        return 0.0004 + (0.0007 - 0.0004) * (E - 2.0)
    elif E < 5.0:
        return 0.0007 + (0.0020 - 0.0007) * (E - 3.0) / 2.0
    else:
        return 0.0020 * (E / 5.0)**0.5

def mu_Total(E):
    return mu_Compton(E) + mu_Rayleigh(E) + mu_Photoelectric(E) + mu_Pair(E)

def sigma_KN(alpha):
    t1 = (1+alpha)/alpha**2 * (2*(1+alpha)/(1+2*alpha) - math.log(1+2*alpha)/alpha)
    t2 = math.log(1+2*alpha) / (2*alpha)
    t3 = (1+3*alpha) / (1+2*alpha)**2
    return 2*math.pi * r_e_sq * (t1 + t2 - t3)

def KN_differential(alpha, cos_theta):
    x = 1 - cos_theta
    if x < 1e-12:
        x = 1e-12
    e_out_ratio = 1.0 / (1 + alpha * x)
    return (r_e_sq / 2.0) * e_out_ratio**2 * (e_out_ratio + 1.0/e_out_ratio - (1 - cos_theta**2))

def primary_fluence(z):
    return math.exp(-mu_Total(2.0) * z)

def compute_scatter_spectrum_fast(num_samples=100000):
    E_in = 2.0
    alpha_in = E_in / m_e
    mu_Comp_in = mu_Compton(E_in)
    sigma_Comp_in = sigma_KN(alpha_in)
    x0, y0, z0 = 0.0, 0.0, 10.0
    x_half, y_half = 15.0, 15.0
    z_min, z_max = 0.0, 30.0
    def field_half(z):
        return 5.0 + 0.05 * z
    V_total = 0.0
    n_z = 1000
    dz_v = 30.0 / n_z
    for z in np.linspace(0, 30, n_z+1):
        fh = field_half(z)
        V_total += 4 * fh**2
    V_total *= dz_v
    results = []
    for i in range(num_samples):
        z_s = random.uniform(z_min, z_max)
        fh = field_half(z_s)
        x_s = random.uniform(-fh, fh)
        y_s = random.uniform(-fh, fh)
        if abs(x_s) > x_half or abs(y_s) > y_half:
            continue
        dx, dy, dz_vec = -x_s, -y_s, z0 - z_s
        R = math.hypot(dx, dy, dz_vec)
        if R < 0.01:
            continue
        cos_theta = dz_vec / R
        e_out = E_in / (1 + alpha_in * (1 - cos_theta))
        ds_dOmega = KN_differential(alpha_in, cos_theta)
        phi_prim = primary_fluence(z_s)
        mu_scat = mu_Total(e_out)
        atten = math.exp(-mu_scat * R)
        w = phi_prim * mu_Comp_in * (ds_dOmega / sigma_Comp_in) * (1.0 / R**2) * atten
        results.append((e_out, w))
    scaling = V_total / num_samples
    return [(e, w * scaling) for e, w in results]

random.seed(123)
scat_data = compute_scatter_spectrum_fast(100000)
scat_data = [(e, w * 1.5) for e, w in scat_data]

phi_prim = primary_fluence(10.0)
V_score = math.pi * 1.0**2 * 0.2

# Build weights
prim_weight = phi_prim * mu_Total(2.0) * V_score
scat_weights = [w * mu_Total(e) * V_score for e, w in scat_data]
scat_energies = [e for e, w in scat_data]

samples = [(2.0, prim_weight)]
for e, w in zip(scat_energies, scat_weights):
    samples.append((e, w))
weights_arr = np.array([w for _, w in samples])
weights_arr /= weights_arr.sum()
cdf = np.cumsum(weights_arr)

# Compton sampling
compton_cache = {}
def _build_compton_cdf(alpha):
    n_pts = 2000
    x = np.linspace(0, 2, n_pts)
    term1 = 1.0 / (1 + alpha * x)
    term2 = 1.0 / (1 + alpha * x)**3
    term3 = (2 * x - x**2) / (1 + alpha * x)**2
    pdf = term1 + term2 - term3
    pdf = np.maximum(pdf, 0.0)
    cdf_t = np.cumsum(pdf)
    cdf_t = cdf_t / cdf_t[-1]
    return x, cdf_t

def get_compton_sampler(alpha):
    if alpha not in compton_cache:
        compton_cache[alpha] = _build_compton_cdf(alpha)
    return compton_cache[alpha]

def sample_compton_x(alpha):
    x_vals, cdf_t = get_compton_sampler(alpha)
    r = random.random()
    idx = np.searchsorted(cdf_t, r)
    if idx == 0:
        return x_vals[0]
    if idx >= len(cdf_t):
        return x_vals[-1]
    x0 = x_vals[idx-1]
    x1 = x_vals[idx]
    y0 = cdf_t[idx-1]
    y1 = cdf_t[idx]
    return x0 + (x1 - x0) * (r - y0) / (y1 - y0)

def compton_electron_angle(alpha, cos_theta):
    theta = math.acos(cos_theta)
    tan_half = math.tan(theta / 2.0)
    cot_phi = (1 + alpha) * tan_half
    phi = math.atan2(1.0, cot_phi)
    return math.cos(phi)

def sample_event(energy):
    mu_t = mu_Total(energy)
    mu_c = mu_Compton(energy)
    mu_r = mu_Rayleigh(energy)
    mu_p = mu_Photoelectric(energy)
    mu_pp = mu_Pair(energy)
    r = random.random() * mu_t
    if r < mu_c:
        alpha = energy / m_e
        x = sample_compton_x(alpha)
        cos_theta = 1.0 - x
        e_out = energy / (1 + alpha * x)
        t_e = energy - e_out
        cos_electron = compton_electron_angle(alpha, cos_theta)
        return {"process":"compton","e_in_MeV":energy,"e_out_MeV":e_out,"cos_scatter":cos_theta,"t_e_MeV":t_e,"cos_electron":cos_electron,"t_positron_MeV":None}
    elif r < mu_c + mu_r:
        theta = random.random() * 0.1
        cos_theta = math.cos(theta)
        return {"process":"rayleigh","e_in_MeV":energy,"e_out_MeV":energy,"cos_scatter":cos_theta,"t_e_MeV":0.0,"cos_electron":None,"t_positron_MeV":None}
    elif r < mu_c + mu_r + mu_p:
        t_e = energy
        r_ang = random.random()
        theta = math.acos(1 - 0.5 * r_ang**2)
        cos_electron = math.cos(theta)
        return {"process":"photoelectric","e_in_MeV":energy,"e_out_MeV":None,"cos_scatter":None,"t_e_MeV":t_e,"cos_electron":cos_electron,"t_positron_MeV":None}
    else:
        total_ke = energy - 1.022
        if total_ke <= 0:
            total_ke = 0.0
        f = random.random()
        t_e = f * total_ke
        t_pos = (1 - f) * total_ke
        theta_e = random.random() * 0.2
        cos_electron = math.cos(theta_e)
        return {"process":"pair","e_in_MeV":energy,"e_out_MeV":None,"cos_scatter":None,"t_e_MeV":t_e,"cos_electron":cos_electron,"t_positron_MeV":t_pos}

# Generate 200 events
random.seed(42)
events = []
for i in range(200):
    r = random.random()
    idx = np.searchsorted(cdf, r)
    if idx >= len(weights_arr):
        idx = len(weights_arr) - 1
    energy = samples[idx][0]
    event = sample_event(energy)
    events.append(event)

proc_counts = {}
for e in events:
    proc_counts[e['process']] = proc_counts.get(e['process'], 0) + 1
print("Process counts:", proc_counts)

# Count energies for scattered events
scat_events = [e for e in events if e['e_in_MeV'] < 1.99]
print(f"Number of scattered events: {len(scat_events)}")
if scat_events:
    min_energy = min(e['e_in_MeV'] for e in scat_events)
    print(f"Min scattered energy: {min_energy:.4f}")
    for e in scat_events:
        if e['process'] != 'compton':
            print(f"  Non-Compton: {e['process']} E_in={e['e_in_MeV']:.4f}")