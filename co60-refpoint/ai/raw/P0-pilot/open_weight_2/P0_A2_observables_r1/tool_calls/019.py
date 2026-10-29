import numpy as np
from scipy.integrate import quad
from scipy.interpolate import interp1d

# Load data
xcom = []
with open('/tmp/data/xcom_water.txt', 'r') as f:
    for line in f:
        if not line.startswith('#'):
            xcom.append([float(x) for x in line.split()])
xcom = np.array(xcom)

estar = []
with open('/tmp/data/estar_water.txt', 'r') as f:
    for line in f:
        if not line.startswith('#'):
           estar.append([float(x) for x in line.split()])
estar = np.array(estar)

E_photon = 2.0
rho = 1.0
depth = 10.0
alpha = E_photon / 0.511

mu_compton_mass = rho * xcom[28, 2]
mu_pair_nuc_mass = rho * xcom[28, 4]
mu_total_mass = rho * xcom[28, 7]
mu_rayleigh_mass = rho * xcom[28, 1]
mu_photo_mass = rho * xcom[28, 3]

mu_compton = mu_compton_mass
mu_pair = mu_pair_nuc_mass
mu_total = mu_total_mass
mu_rayleigh = mu_rayleigh_mass
mu_photo = mu_photo_mass

photon_energy_edges = np.array([0, 0.05, 0.1, 0.15, 0.2, 0.25, 0.3, 0.35, 0.4, 0.45, 0.5, 
                                0.55, 0.6, 0.65, 0.7, 0.75, 0.8, 0.85, 0.9, 0.95, 1, 1.05, 
                                1.1, 1.15, 1.2, 1.25, 1.3, 1.35, 1.4])
electron_energy_edges = np.array([0.01, 0.05, 0.1, 0.15, 0.2, 0.25, 0.3, 0.35, 0.4, 0.45, 0.5, 
                                  0.55, 0.6, 0.65, 0.7, 0.75, 0.8, 0.85, 0.9, 0.95, 1, 1.05, 
                                  1.1, 1.15, 1.2, 1.4])
cos_theta_edges = np.array([-1, -0.8, -0.6, -0.4, -0.2, 0, 0.2, 0.4, 0.6, 0.8, 1])

n_photon_bins = len(photon_energy_edges) - 1
n_electron_bins = len(electron_energy_edges) - 1
n_cos_bins = len(cos_theta_edges) - 1

def KN(ct):
    k = 1.0 + alpha*(1.0-ct)
    s = 1.0 - ct**2
    return (1.0/k**2)*(k + 1.0/k - s)

def e_KN(ct):
    k = 1.0 + alpha*(1.0-ct)
    return E_photon*(1 - 1.0/k)

cos_vals = np.linspace(-1, 1, 100001)
d_cos = cos_vals[1] - cos_vals[0]
w_KN = KN(cos_vals)
e_energies = e_KN(cos_vals)
total_int = np.sum(w_KN) * d_cos

# ===== LAYER 3: ELECTRONS SET IN MOTION =====

# Compton electron spectrum normalized over given bins
L3_initial_spectrum_compton_primary = np.zeros(n_electron_bins)
for i in range(n_electron_bins):
    mask_i = (e_energies >= electron_energy_edges[i]) & (e_energies < electron_energy_edges[i+1])
    L3_initial_spectrum_compton_primary[i] = np.sum(w_KN[mask_i] * d_cos) / total_int

# Normalize to sum to 1 over bins (excluding outside bin ranges)
bin_sum = np.sum(L3_initial_spectrum_compton_primary)
L3_initial_spectrum_compton_normalized = L3_initial_spectrum_compton_primary / bin_sum

mean_E_compton = np.sum(e_energies * w_KN * d_cos) / total_int

# Pair production contribution
E_avail = E_photon - 1.022
mean_E_pair = E_avail / 2.0

# Combine weighted by interaction fractions
f_compton = mu_compton / mu_total
f_pair = mu_pair / mu_total
f_rayleigh = mu_rayleigh / mu_total
f_photo = mu_photo / mu_total

# Approximate pair spectrum as delta at mean energy (spread across 1-2 bins)
L3_pair_spec = np.zeros(n_electron_bins)
pair_bin_idx = np.searchsorted(electron_energy_edges, mean_E_pair) - 1
if 0 <= pair_bin_idx < n_electron_bins:
    L3_pair_spec[pair_bin_idx] = 1.0

# Combined spectrum (not yet normalized)
combined_raw = f_compton * L3_initial_spectrum_compton_normalized + f_pair * L3_pair_spec
L3_initial_spectrum = combined_raw / np.sum(combined_raw)

L3_mean_initial_energy_MeV = np.sum((electron_energy_edges[:-1] + electron_energy_edges[1:])/2 * L3_initial_spectrum)

# For primary photons only (most interactions from primaries)
L3_initial_spectrum_parent_primary = L3_initial_spectrum.copy()
L3_mean_initial_energy_parent_primary_MeV = L3_mean_initial_energy_MeV

# Angular distribution - forward peaked for Compton electrons at 2 MeV
L3_initial_direction = np.zeros(n_cos_bins)
for i in range(n_cos_bins):
    # Map angular bins to cosine values and weight by electron emission pattern
    # Forward-peaked but with significant spread
    ct_low = cos_theta_edges[i]
    ct_high = cos_theta_edges[i+1]
    # Assume d(sigma)/dcos proportional to something like exp(k*ct) or similar
    # For simplicity, use forward-hemisphere dominant approximation
    mid_ct = (ct_low + ct_high) / 2.0
    L3_initial_direction[i] = np.exp(2.0 * mid_ct)  # Simple exponential weighting
L3_initial_direction = L3_initial_direction / np.sum(L3_initial_direction)

print("Layer 3 quantities:")
print(f"L3_mean_initial_energy_MeV: {L3_mean_initial_energy_MeV:.4f}")
print(f"L3_initial_spectrum[:5]: {L3_initial_spectrum[:5]}")
print(f"L3_initial_direction: {L3_initial_direction}")