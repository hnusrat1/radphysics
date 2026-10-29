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

# KN distribution helpers
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

# Discretize
cos_vals = np.linspace(-1, 1, 100001)
d_cos = cos_vals[1] - cos_vals[0]
w_KN = KN(cos_vals)
e_energies = e_KN(cos_vals)
g_energies = gamma_KN(cos_vals)
total_int = np.sum(w_KN) * d_cos

# Primary fluence
L1_primary_fluence_per_phi0 = np.exp(-mu_total * depth)

# Layer 1 primary spectrum
L1_photon_fluence_spectrum_primary = np.zeros(n_photon_bins)
for i in range(n_photon_bins):
    if photon_energy_edges[i] <= E_photon < photon_energy_edges[i+1]:
        L1_photon_fluence_spectrum_primary[i] = L1_primary_fluence_per_phi0

L1_angular_primary = np.zeros(n_cos_bins)
L1_angular_primary[-1] = 1.0

# Scattered estimates
mean_scattered_E_gamma = np.sum(g_energies * w_KN * d_cos) / total_int
P_interact = 1 - np.exp(-mu_total * depth)
f_compton_frac = mu_compton / mu_total
avg_atten_scatter = np.exp(-mu_total * depth / 2)
est_scattered_fraction = P_interact * f_compton_frac * avg_atten_scatter

L1_scattered_fraction_fluence = est_scattered_fraction
L1_total_fluence_per_phi0 = L1_primary_fluence_per_phi0 + L1_scattered_fraction_fluence

mean_E_primary_bin = E_photon
energy_fluence_primary = L1_primary_fluence_per_phi0 * mean_E_primary_bin
energy_fluence_scattered_est = L1_scattered_fraction_fluence * mean_scattered_E_gamma
total_energy_fluence_est = energy_fluence_primary + energy_fluence_scattered_est
L1_mean_photon_energy_MeV = total_energy_fluence_est / L1_total_fluence_per_phi0
L1_scattered_fraction_energy_fluence = energy_fluence_scattered_est / total_energy_fluence_est

# Scattered photon spectrum
L1_photon_fluence_spectrum_scattered = np.zeros(n_photon_bins)
for i in range(n_photon_bins):
    mask_i = (g_energies >= photon_energy_edges[i]) & (g_energies < photon_energy_edges[i+1])
    bin_frac = np.sum(w_KN[mask_i] * d_cos) / total_int
    L1_photon_fluence_spectrum_scattered[i] = L1_scattered_fraction_fluence * bin_frac

# Angular scattered - use actual KN angular distribution of scattered photons
# The intensity is proportional to KN(ct), where ct = cos(theta_gamma)
L1_angular_scattered_raw = np.zeros(n_cos_bins)
for i in range(n_cos_bins):
    mask_i = (cos_vals >= cos_theta_edges[i]) & (cos_vals < cos_theta_edges[i+1])
    L1_angular_scattered_raw[i] = np.sum(w_KN[mask_i] * d_cos) / total_int
L1_angular_scattered = L1_angular_scattered_raw / np.sum(L1_angular_scattered_raw)

print("Layer 1 complete")
print(f"L1_primary: {L1_primary_fluence_per_phi0:.6f}")
print(f"L1_total: {L1_total_fluence_per_phi0:.6f}")
print(f"Mean E: {L1_mean_photon_energy_MeV:.4f} MeV")

# Layer 2
f_compton = mu_compton / mu_total
f_pair = mu_pair / mu_total
f_rayleigh = mu_rayleigh / mu_total
f_photo = mu_photo / mu_total

L2_interaction_fractions = {"compton": f_compton, "photoelectric": f_photo, "pair": f_pair, "rayleigh": f_rayleigh}

frac_transferred_compton = (E_photon - mean_scattered_E_gamma) / E_photon
frac_transferred_pair = (E_photon - 1.022) / E_photon
num_ET = (f_compton * frac_transferred_compton + f_pair * frac_transferred_pair + f_photo + f_rayleigh * 0)
L2_energy_transfer_fractions = {
    "compton": f_compton * frac_transferred_compton / num_ET,
    "photoelectric": f_photo / num_ET,
    "pair": f_pair * frac_transferred_pair / num_ET,
    "rayleigh": 0.0
}

L2_interaction_fractions_primary = dict(L2_interaction_fractions)
L2_interaction_fractions_scattered = dict(L2_interaction_fractions)
L2_scattered_share_of_interactions = est_scattered_fraction / (1 - np.exp(-mu_total * depth)) * 0.5

print("\nLayer 2 complete")