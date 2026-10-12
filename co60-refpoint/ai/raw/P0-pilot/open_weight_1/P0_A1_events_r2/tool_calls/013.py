import numpy as np
import math
import random

# Redo the computation quickly
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

# Compute scatter spectrum
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

random.seed(123)
scat_data = compute_scatter_spectrum_fast(100000)
# Scale
scat_data = [(e, w * 1.5) for e, w in scat_data]

energies = [e for e, w in scat_data]
weights = [w for e, w in scat_data]
print(f"Min energy: {min(energies):.4f} MeV")
print(f"Max energy: {max(energies):.4f} MeV")

bins = np.linspace(0, 2, 21)
hist, edges = np.histogram(energies, bins=bins, weights=weights)
print("Energy bins (MeV) and fluence:")
for i in range(len(hist)):
    if hist[i] > 0:
        print(f"  {edges[i]:.2f}-{edges[i+1]:.2f}: {hist[i]:.4f}")

V_score = math.pi * 1.0**2 * 0.2
scat_int_weights = [w * mu_Total(e) * V_score for e, w in scat_data]
hist_int, _ = np.histogram(energies, bins=bins, weights=scat_int_weights)
print("Interaction weight per bin (relative):")
for i in range(len(hist_int)):
    if hist_int[i] > 0:
        print(f"  {edges[i]:.2f}-{edges[i+1]:.2f}: {hist_int[i]:.6f}")