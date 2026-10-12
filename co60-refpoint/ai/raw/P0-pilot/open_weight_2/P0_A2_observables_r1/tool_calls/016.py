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

# Constants
E_photon = 2.0
rho = 1.0
depth = 10.0
alpha = E_photon / 0.511

# Cross-sections at 2.0 MeV (index 28)
mu_ray_mass = rho * xcom[28, 1]
mu_compton_mass = rho * xcom[28, 2]
mu_photo_mass = rho * xcom[28, 3]
mu_pair_nuc_mass = rho * xcom[28, 4]
mu_pair_el_mass = rho * xcom[28, 5]
mu_total_mass = rho * xcom[28, 7]

mu_total = mu_total_mass
mu_compton = mu_compton_mass
mu_pair = mu_pair_nuc_mass + mu_pair_el_mass
mu_rayleigh = mu_ray_mass
mu_photo = mu_photo_mass

# Bins
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

print("Computing all quantities...")
print("="*60)

# ===== LAYER 1: PHOTONS ARRIVING AT THE POINT =====

# Primary fluence
L1_primary_fluence_per_phi0 = np.exp(-mu_total * depth)
print(f"L1_primary_fluence_per_phi0 = {L1_primary_fluence_per_phi0:.6f}")

# Primary photon fluence spectrum (all at 2.0 MeV)
L1_photon_fluence_spectrum_primary = np.zeros(n_photon_bins)
for i in range(n_photon_bins):
    if photon_energy_edges[i] <= E_photon < photon_energy_edges[i+1]:
        L1_photon_fluence_spectrum_primary[i] = L1_primary_fluence_per_phi0
        break

# Angular distribution primary (all forward)
L1_angular_primary = np.zeros(n_cos_bins)
L1_angular_primary[-1] = 1.0  # All at cos(theta)=1

# Scattered calculations
def KN(ct):
    k = 1.0 + alpha*(1.0-ct)
    s = 1.0 - ct**2
    return (1.0/k**2)*(k + 1.0/k - s)

def e_KN(ct):
    k = 1.0 + alpha*(1.0-ct)
    return E_photon*(1 - 1.0/k)

def gamma_KN(ct):
    k = 1.0 + alpha*(1.0-ct)
    return E_photon / k

# Discretize for spectra
cos_vals = np.linspace(-1, 1, 100001)
d_cos = cos_vals[1] - cos_vals[0]
w_KN = KN(cos_vals)
e_energies = e_KN(cos_vals)
g_energies = gamma_KN(cos_vals)
total_int = np.sum(w_KN) * d_cos

# Mean scattered photon energy from Compton
num_int_gamma = np.sum(g_energies * w_KN * d_cos)
mean_scattered_E_gamma = num_int_gamma / total_int
print(f"\nMean scattered photon energy (Compton): {mean_scattered_E_gamma:.4f} MeV")

# Estimate scattered fluence
P_interact = 1 - np.exp(-mu_total * depth)
f_compton_frac = mu_compton / mu_total
# Average attenuation factor for scattered photons (~0.5 since produced at various depths)
avg_atten_scatter = np.exp(-mu_total * depth / 2)
est_scattered_fraction = P_interact * f_compton_frac * avg_atten_scatter

L1_scattered_fraction_fluence = est_scattered_fraction
L1_total_fluence_per_phi0 = L1_primary_fluence_per_phi0 + L1_scattered_fraction_fluence

print(f"L1_scattered_fraction_fluence = {L1_scattered_fraction_fluence:.6f}")
print(f"L1_total_fluence_per_phi0 = {L1_total_fluence_per_phi0:.6f}")

# Fluence-weighted mean photon energy
mean_E_primary_bin = E_photon
energy_fluence_primary = L1_primary_fluence_per_phi0 * mean_E_primary_bin
energy_fluence_scattered_est = L1_scattered_fraction_fluence * mean_scattered_E_gamma
total_energy_fluence_est = energy_fluence_primary + energy_fluence_scattered_est
L1_mean_photon_energy_MeV = total_energy_fluence_est / L1_total_fluence_per_phi0
print(f"L1_mean_photon_energy_MeV = {L1_mean_photon_energy_MeV:.4f}")

# Scattered fraction of energy fluence
L1_scattered_fraction_energy_fluence = energy_fluence_scattered_est / total_energy_fluence_est
print(f"L1_scattered_fraction_energy_fluence = {L1_scattered_fraction_energy_fluence:.6f}")

# Scattered photon fluence spectrum (primarily Compton-scattered down to lower energies)
L1_photon_fluence_spectrum_scattered = np.zeros(n_photon_bins)
for i in range(n_photon_bins):
    mask_i = (g_energies >= photon_energy_edges[i]) & (g_energies < photon_energy_edges[i+1])
    bin_frac = np.sum(w_KN[mask_i] * d_cos) / total_int
    L1_photon_fluence_spectrum_scattered[i] = L1_scattered_fraction_fluence * bin_frac

# Normalized angular scattered distribution
# Scattered photons come mainly from Compton scattering, which produces photons peaked backward but with broad distribution
# For simplicity, assume roughly uniform over forward hemisphere for first-order estimate
# Actually, at 2 MeV, Compton scattering has significant backward component
L1_angular_scattered = np.ones(n_cos_bins) * (1.0 / n_cos_bins)  # Rough approximation
# Normalize to sum to 1
L1_angular_scattered = L1_angular_scattered / np.sum(L1_angular_scattered)
print(f"L1_angular_scattered sum: {np.sum(L1_angular_scattered):.6f}")