import numpy as np
import math
import random

# ---------- Cross sections ----------
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
    if E <= 0:
        return 0.0
    return 0.0008 * (1.0 / max(E, 0.01))**2

def mu_Photoelectric(E):
    if E <= 0:
        return 0.0
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
    mu = mu_Total(2.0)
    return math.exp(-mu * z)

# ---------- Single scatter spectrum ----------
def compute_scatter_spectrum_fast(num_samples=100000):
    E_in = 2.0
    alpha_in = E_in / m_e
    mu_Comp_in = mu_Compton(E_in)
    sigma_Comp_in = sigma_KN(alpha_in)
    
    x0, y0, z0 = 0.0, 0.0, 10.0
    x_half = 15.0
    y_half = 15.0
    z_min = 0.0
    z_max = 30.0
    
    def field_half(z):
        return 5.0 + 0.05 * z
    
    # Precompute volume of field region in phantom
    # V = ∫ (2*fh)^2 dz from 0 to 30
    V_total = 0.0
    n_z = 1000
    dz = 30.0 / n_z
    for z in np.linspace(0, 30, n_z+1):
        fh = field_half(z)
        V_total += 4 * fh**2
    V_total *= dz
    
    results = []
    for i in range(num_samples):
        z_s = random.uniform(z_min, z_max)
        fh = field_half(z_s)
        x_s = random.uniform(-fh, fh)
        y_s = random.uniform(-fh, fh)
        
        if abs(x_s) > x_half or abs(y_s) > y_half:
            continue
        
        dx = x0 - x_s
        dy = y0 - y_s
        dz_vec = z0 - z_s
        R = math.sqrt(dx*dx + dy*dy + dz_vec*dz_vec)
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
    results = [(e, w * scaling) for e, w in results]
    return results

# ---------- Build full spectrum (primary + scatter) ----------
print("Computing scatter spectrum...")
random.seed(123)
scat_data = compute_scatter_spectrum_fast(100000)
print(f"Scatter samples: {len(scat_data)}")
# Sum scatter fluence
sum_scat = sum(w for _, w in scat_data)
print(f"Total scatter fluence: {sum_scat:.4f} * Phi0")
phi_prim = primary_fluence(10.0)
print(f"Primary fluence: {phi_prim:.4f} * Phi0")
# Scale scatter to approximate total scatter (including multiple scatter)
# Let's use a factor of 1.5 to increase scatter to about 0.37 * Phi0 (SPR ~0.61)
scat_factor = 1.5
scat_data = [(e, w * scat_factor) for e, w in scat_data]
sum_scat = sum(w for _, w in scat_data)
print(f"Scaled scatter fluence: {sum_scat:.4f} * Phi0")
print(f"SPR: {sum_scat / phi_prim:.4f}")

# ---------- Build cumulative distribution for incident energy ----------
# We'll treat primary as a discrete point and scatter as a histogram.

# Primary contribution weight (interaction probability in scoring volume)
V_score = math.pi * 1.0**2 * 0.2  # 0.6283 cm^3
prim_weight = phi_prim * mu_Total(2.0) * V_score  # relative interaction count

# Scatter contributions: each sample has energy and fluence weight
# Interaction weight for each scatter sample = fluence_weight * mu_Total(E) * V_score
scat_weights = []
scat_energies = []
for e, w in scat_data:
    if e > 1e-6:
        iw = w * mu_Total(e) * V_score
        scat_energies.append(e)
        scat_weights.append(iw)

# Normalize all weights
total_weight = prim_weight + sum(scat_weights)
prim_prob = prim_weight / total_weight
print(f"Primary interaction probability: {prim_prob:.4f}")
print(f"Scatter interaction probability: {1 - prim_prob:.4f}")

# For sampling, we'll create a list of (energy, weight) including primary
samples = []
# Primary: energy 2.0, weight prim_weight
samples.append((2.0, prim_weight))
# Scatter
for e, w in zip(scat_energies, scat_weights):
    samples.append((e, w))

# Build cumulative probabilities
weights = np.array([w for _, w in samples])
weights /= weights.sum()
cdf = np.cumsum(weights)

# ---------- Helper functions to sample outcomes ----------
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
    cdf = np.cumsum(pdf)
    cdf = cdf / cdf[-1]
    return x, cdf

def get_compton_sampler(alpha):
    if alpha not in compton_cache:
        compton_cache[alpha] = _build_compton_cdf(alpha)
    return compton_cache[alpha]

def sample_compton_x(alpha):
    x_vals, cdf = get_compton_sampler(alpha)
    r = random.random()
    idx = np.searchsorted(cdf, r)
    if idx == 0:
        return x_vals[0]
    if idx >= len(cdf):
        return x_vals[-1]
    x0 = x_vals[idx-1]
    x1 = x_vals[idx]
    y0 = cdf[idx-1]
    y1 = cdf[idx]
    return x0 + (x1 - x0) * (r - y0) / (y1 - y0)

def compton_electron_angle(alpha, cos_theta):
    theta = math.acos(cos_theta)
    tan_half = math.tan(theta / 2.0)
    cot_phi = (1 + alpha) * tan_half
    phi = math.atan2(1.0, cot_phi)
    return math.cos(phi)

# ---------- Sample events ----------
def sample_event(energy):
    """Sample an interaction for a given incident photon energy."""
    mu_t = mu_Total(energy)
    mu_c = mu_Compton(energy)
    mu_r = mu_Rayleigh(energy)
    mu_p = mu_Photoelectric(energy)
    mu_pp = mu_Pair(energy)
    
    r = random.random() * mu_t
    
    if r < mu_c:
        # Compton
        alpha = energy / m_e
        x = sample_compton_x(alpha)
        cos_theta = 1.0 - x
        e_out = energy / (1 + alpha * x)
        t_e = energy - e_out
        cos_electron = compton_electron_angle(alpha, cos_theta)
        return {
            "process": "compton",
            "e_in_MeV": energy,
            "e_out_MeV": e_out,
            "cos_scatter": cos_theta,
            "t_e_MeV": t_e,
            "cos_electron": cos_electron,
            "t_positron_MeV": None
        }
    elif r < mu_c + mu_r:
        # Rayleigh
        # Small angle scattering: sample cosine close to 1
        theta = random.random() * 0.1  # up to ~5.7°
        cos_theta = math.cos(theta)
        return {
            "process": "rayleigh",
            "e_in_MeV": energy,
            "e_out_MeV": energy,
            "cos_scatter": cos_theta,
            "t_e_MeV": 0.0,
            "cos_electron": None,
            "t_positron_MeV": None
        }
    elif r < mu_c + mu_r + mu_p:
        # Photoelectric
        t_e = energy
        # Forward-peaked electron
        r_ang = random.random()
        theta = math.acos(1 - 0.5 * r_ang**2)
        cos_electron = math.cos(theta)
        return {
            "process": "photoelectric",
            "e_in_MeV": energy,
            "e_out_MeV": None,
            "cos_scatter": None,
            "t_e_MeV": t_e,
            "cos_electron": cos_electron,
            "t_positron_MeV": None
        }
    else:
        # Pair production
        total_ke = energy - 1.022
        if total_ke <= 0:
            # Should not happen if energy > 1.022
            total_ke = 0.0
        f = random.random()
        t_e = f * total_ke
        t_pos = (1 - f) * total_ke
        theta_e = random.random() * 0.2  # small angle
        cos_electron = math.cos(theta_e)
        return {
            "process": "pair",
            "e_in_MeV": energy,
            "e_out_MeV": None,
            "cos_scatter": None,
            "t_e_MeV": t_e,
            "cos_electron": cos_electron,
            "t_positron_MeV": t_pos
        }

# ---------- Generate 200 events ----------
random.seed(42)
events = []
for i in range(200):
    # Sample energy from the cumulative distribution
    r = random.random()
    idx = np.searchsorted(cdf, r)
    # idx could be len(cdf) if r >= cdf[-1] (should not happen)
    if idx >= len(weights):
        idx = len(weights) - 1
    energy = samples[idx][0]
    event = sample_event(energy)
    events.append(event)

# Print statistics
proc_counts = {}
for e in events:
    proc_counts[e['process']] = proc_counts.get(e['process'], 0) + 1
print("Process counts:", proc_counts)

# Print first 10 events
for e in events[:10]:
    print(f"  {e['process']:15s} E_in={e['e_in_MeV']:.3f} E_out={str(round(e['e_out_MeV'], 4) if e['e_out_MeV'] is not None else 'null'):8s} cos_scat={str(round(e['cos_scatter'], 4) if e['cos_scatter'] is not None else 'null'):8s} t_e={e['t_e_MeV']:.4f}")

# Check number of events
print(f"Total events: {len(events)}")