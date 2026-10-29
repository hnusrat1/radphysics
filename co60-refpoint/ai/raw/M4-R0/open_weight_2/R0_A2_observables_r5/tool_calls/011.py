import numpy as np
from scipy.special import eval_legendre

# Reload data (since variables didn't persist)
with open('/tmp/data/xcom_water.txt', 'r') as f:
    xcom_lines = f.readlines()
    
with open('/tmp/data/estar_water.txt', 'r') as f:
    estar_lines = f.readlines()

xcom_data = []
for line in xcom_lines[4:]:
    parts = line.split()
    if len(parts) >= 7:
        try:
            vals = [float(p) for p in parts]
            xcom_data.append(vals)
        except:
            pass
xcom_array = np.array(xcom_data)

estar_data = []
for line in estar_lines[4:]:
    parts = line.split()
    if len(parts) >= 7:
        try:
            vals = [float(p) for p in parts]
            estar_data.append(vals)
        except:
            pass
estar_array = np.array(estar_data)

def interpolate_xlog(E, col_idx):
    energies = xcom_array[:, 0]
    idx = np.searchsorted(energies, E)
    if idx == 0 or idx >= len(energies):
        return None
    
    log_E = np.log10(max(E, 1e-20))
    log_E_low = np.log10(energies[idx-1])
    log_E_high = np.log10(energies[idx])
    
    t = (log_E - log_E_low) / (log_E_high - log_E_low)
    
    val_low = xcom_array[idx-1, col_idx]
    val_high = xcom_array[idx, col_idx]
    
    if val_low > 0 and val_high > 0:
        return val_low * (val_high/val_low)**t
    elif val_high > 0:
        return val_high
    else:
        return 0.0

def interpolate_estar_log(E):
    energies = estar_array[:, 0]
    idx = np.searchsorted(energies, E)
    if idx == 0 or idx >= len(energies):
        return None
    
    log_E = np.log10(max(E, 1e-20))
    log_E_low = np.log10(energies[idx-1])
    log_E_high = np.log10(energies[idx])
    
    t = (log_E - log_E_low) / (log_E_high - log_E_low)
    
    result = {}
    cols_names = ['collision', 'radiative', 'total']
    col_indices = [1, 2, 3]
    
    for name, idx_col in zip(cols_names, col_indices):
        val_low = estar_array[idx-1, idx_col]
        val_high = estar_array[idx, idx_col]
        
        if val_low > 0 and val_high > 0:
            result[name] = val_low * (val_high/val_low)**t
        elif val_high > 0:
            result[name] = val_high
        else:
            result[name] = 0.0
    
    return result

E_co1 = 1.1732
E_co2 = 1.3325
rho = 1.0
depth = 5.0

xs_co1 = {k: interpolate_xlog(E_co1, i+1) for i, k in enumerate(['coherent','incoherent','photoelectric','pair_nuclear','pair_electron','total'])}
xs_co2 = {k: interpolate_xlog(E_co2, i+1) for i, k in enumerate(['coherent','incoherent','photoelectric','pair_nuclear','pair_electron','total'])}

# Klein-Nishina function
def klein_nishinan_dsigma_dcoss(theta, E_gamma):
    alpha = E_gamma / 0.510998950
    E_prime = E_gamma / (1 + alpha * (1 - np.cos(theta)))
    r0_sq_over_4pi = 0.066524e-24 / (4 * np.pi)
    factor = (1 + np.cos(theta)**2) / 2
    eprime_over_E = E_prime / E_gamma
    d_sigma = factor * eprime_over_E**2 * r0_sq_over_4pi
    return d_sigma, E_prime

# Attenuation
mu1 = xs_co1['total'] * rho
mu2 = xs_co2['total'] * rho
att1 = np.exp(-mu1 * depth)
att2 = np.exp(-mu2 * depth)
L1_primary = 0.5 * att1 + 0.5 * att2

# Buildup approximation at 5cm for Co-60, 10x10 field
# At 5cm depth, scattered photon contribution is modest
B_photon = 1.12  # Reasonable buildup factor
L1_total = L1_primary * B_photon
L1_scatter_frac = 1 - 1/B_photon

# Mean energy of all photons
E_mean_prim = 0.5 * E_co1 + 0.5 * E_co2
# Scattered photon average ~0.5 MeV (Compton continuum peak)
E_mean_scat = 0.5
frac_prim = 1.0/B_photon
L1_mean_energy = frac_prim * E_mean_prim + (1-frac_prim) * E_mean_scat

# Photon spectra
photon_energy_bins = [0, 0.05, 0.1, 0.15, 0.2, 0.25, 0.3, 0.35, 0.4, 0.45, 0.5, 
                      0.55, 0.6, 0.65, 0.7, 0.75, 0.8, 0.85, 0.9, 0.95, 1, 
                      1.05, 1.1, 1.15, 1.2, 1.25, 1.3, 1.35, 1.4]
n_ph_bins = len(photon_energy_bins) - 1

spec_prim = np.zeros(n_ph_bins)
# Find bins containing the two energies
for E_line in [E_co1, E_co2]:
    for b in range(n_ph_bins):
        if photon_energy_bins[b] <= E_line < photon_energy_bins[b+1]:
            spec_prim[b] += 0.5 * L1_primary

spec_scat = np.zeros(n_ph_bins)
scat_flux = L1_total - L1_primary
# Distribute Compton scattered photons over 0.05 to 1.1 MeV range
for b in range(n_ph_bins):
    E_mid = (photon_energy_bins[b] + photon_energy_bins[b+1]) / 2
    if 0.05 <= E_mid < 1.1:
        # Roughly flat distribution for simplicity, normalized
        if 0.05 <= E_mid < 0.5:
            spec_scat[b] = scat_flux * 0.6 / 8  # Lower part gets 60%
        else:
            spec_scat[b] = scat_flux * 0.4 / 7  # Higher part gets 40%

# Normalize scattered spectrum to sum correctly
scat_sum = np.sum(spec_scat)
if scat_sum > 0:
    spec_scat *= scat_flux / scat_sum

# Angular distributions
cos_theta_bins = [-1, -0.8, -0.6, -0.4, -0.2, 0, 0.2, 0.4, 0.6, 0.8, 1]
n_cos_bins = len(cos_theta_bins) - 1

ang_prim = np.zeros(n_cos_bins)
ang_prim[-1] = 1.0  # All forward (beam axis direction)

ang_scat = np.zeros(n_cos_bins)
# Compute Klein-Nishina integrated probabilities over cos theta ranges
prob = []
for i in range(n_cos_bins):
    c_low = cos_theta_bins[i]
    c_high = cos_theta_bins[i+1]
    c_mid = (c_low + c_high) / 2
    theta = np.arccos(np.clip(c_mid, -1, 1))
    sigma_rel, _ = klein_nishinan_dsigma_dcoss(theta, E_mean_prim)
    prob.append(sigma_rel)
prob = np.array(prob)
prob /= prob.sum()
ang_scat = prob

print("=== LAYER 1 SUMMARY ===")
print(f"L1_primary_fluence_per_phi0 = {L1_primary:.6f}")
print(f"L1_total_fluence_per_phi0 = {L1_total:.6f}")
print(f"L1_scattered_fraction_fluence = {L1_scatter_frac:.4f}")
print(f"L1_mean_photon_energy_MeV = {L1_mean_energy:.4f}")
print(f"\nAngular primary sum: {np.sum(ang_prim):.6f}")
print(f"Angular scattered sum: {np.sum(ang_scat):.6f}")
print(f"Primary spectrum sum: {np.sum(spec_prim):.6f}")
print(f"Scattered spectrum sum: {np.sum(spec_scat):.6f}")