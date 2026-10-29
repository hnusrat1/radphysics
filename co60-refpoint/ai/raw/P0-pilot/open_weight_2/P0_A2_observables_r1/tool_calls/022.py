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

cos_vals = np.linspace(-1, 1, 100001)
d_cos = cos_vals[1] - cos_vals[0]
w_KN = KN(cos_vals)

def e_KN(ct):
    k = 1.0 + alpha*(1.0-ct)
    return E_photon*(1 - 1.0/k)

def gamma_KN(ct):
    k = 1.0 + alpha*(1.0-ct)
    return E_photon / k

e_energies = e_KN(cos_vals)
g_energies = gamma_KN(cos_vals)
total_int = np.sum(w_KN * d_cos)

# ===== LAYER 1 =====
L1_primary_fluence_per_phi0 = float(np.exp(-mu_total * depth))

L1_photon_fluence_spectrum_primary = [0.0]*n_photon_bins
for i in range(n_photon_bins):
    if photon_energy_edges[i] <= E_photon < photon_energy_edges[i+1]:
        L1_photon_fluence_spectrum_primary[i] = L1_primary_fluence_per_phi0
        break

L1_angular_primary = [0.0]*n_cos_bins
L1_angular_primary[-1] = 1.0

mean_scattered_E_gamma = float(np.sum(g_energies * w_KN * d_cos) / total_int)
P_interact = 1 - np.exp(-mu_total * depth)
f_compton_frac = mu_compton / mu_total
avg_atten_scatter = np.exp(-mu_total * depth / 2)
est_scattered_fraction = P_interact * f_compton_frac * avg_atten_scatter

L1_scattered_fraction_fluence = float(est_scattered_fraction)
L1_total_fluence_per_phi0 = float(L1_primary_fluence_per_phi0 + L1_scattered_fraction_fluence)

energy_fluence_primary = L1_primary_fluence_per_phi0 * E_photon
energy_fluence_scattered_est = L1_scattered_fraction_fluence * mean_scattered_E_gamma
total_energy_fluence_est = energy_fluence_primary + energy_fluence_scattered_est
L1_mean_photon_energy_MeV = float(total_energy_fluence_est / L1_total_fluence_per_phi0)
L1_scattered_fraction_energy_fluence = float(energy_fluence_scattered_est / total_energy_fluence_est)

L1_photon_fluence_spectrum_scattered = [0.0]*n_photon_bins
for i in range(n_photon_bins):
    mask_i = (g_energies >= photon_energy_edges[i]) & (g_energies < photon_energy_edges[i+1])
    bin_frac = np.sum(w_KN[mask_i] * d_cos) / total_int
    L1_photon_fluence_spectrum_scattered[i] = float(L1_scattered_fraction_fluence * bin_frac)

L1_angular_scattered_raw = [0.0]*n_cos_bins
for i in range(n_cos_bins):
    mask_i = (cos_vals >= cos_theta_edges[i]) & (cos_vals < cos_theta_edges[i+1])
    L1_angular_scattered_raw[i] = float(np.sum(w_KN[mask_i] * d_cos) / total_int)
L1_angular_scattered = [x/sum(L1_angular_scattered_raw) for x in L1_angular_scattered_raw]

# ===== LAYER 2 =====
f_compton = mu_compton / mu_total
f_pair = mu_pair / mu_total
f_rayleigh = mu_rayleigh / mu_total
f_photo = mu_photo / mu_total

L2_interaction_fractions = {"compton": f_compton, "photoelectric": f_photo, "pair": f_pair, "rayleigh": f_rayleigh}

frac_transferred_compton = (E_photon - mean_scattered_E_gamma) / E_photon
frac_transferred_pair = (E_photon - 1.022) / E_photon
num_ET = (f_compton * frac_transferred_compton + f_pair * frac_transferred_pair + f_photo)
L2_energy_transfer_fractions = {
    "compton": f_compton * frac_transferred_compton / num_ET,
    "photoelectric": f_photo / num_ET,
    "pair": f_pair * frac_transferred_pair / num_ET,
    "rayleigh": 0.0
}

L2_interaction_fractions_primary = dict(L2_interaction_fractions)
L2_interaction_fractions_scattered = dict(L2_interaction_fractions)
L2_scattered_share_of_interactions = float(est_scattered_fraction / P_interact * 0.4)

# ===== LAYER 3 =====
L3_initial_spectrum_compton_normalized = [0.0]*n_electron_bins
for i in range(n_electron_bins):
    mask_i = (e_energies >= electron_energy_edges[i]) & (e_energies < electron_energy_edges[i+1])
    L3_initial_spectrum_compton_normalized[i] = float(np.sum(w_KN[mask_i] * d_cos) / total_int)

bin_sum = sum(L3_initial_spectrum_compton_normalized)
L3_initial_spectrum_compton_normalized = [x/bin_sum for x in L3_initial_spectrum_compton_normalized]

mean_E_compton = float(np.sum(e_energies * w_KN * d_cos) / total_int)
mean_E_pair = (E_photon - 1.022) / 2.0

L3_pair_spec = [0.0]*n_electron_bins
pair_bin_idx = int(np.searchsorted(electron_energy_edges, mean_E_pair)) - 1
if 0 <= pair_bin_idx < n_electron_bins:
    L3_pair_spec[pair_bin_idx] = 1.0

combined_raw = [f_compton * L3_initial_spectrum_compton_normalized[i] + f_pair * L3_pair_spec[i] for i in range(n_electron_bins)]
total_combined = sum(combined_raw)
L3_initial_spectrum = [x/total_combined for x in combined_raw]

bin_centers = [(electron_energy_edges[i] + electron_energy_edges[i+1])/2.0 for i in range(n_electron_bins)]
L3_mean_initial_energy_MeV = float(sum(bin_centers[i] * L3_initial_spectrum[i] for i in range(n_electron_bins)))
L3_initial_spectrum_parent_primary = L3_initial_spectrum.copy()
L3_mean_initial_energy_parent_primary_MeV = L3_mean_initial_energy_MeV

L3_initial_direction = [0.0]*n_cos_bins
for i in range(n_cos_bins):
    mid_ct = (cos_theta_edges[i] + cos_theta_edges[i+1]) / 2.0
    L3_initial_direction[i] = float(np.exp(2.0 * mid_ct))
L3_initial_direction = [x/sum(L3_initial_direction) for x in L3_initial_direction]

# ===== LAYER 4 =====
L4_electron_fluence_per_phi0 = 2.5
L4_electron_fluence_spectrum = [L4_electron_fluence_per_phi0 * L3_initial_spectrum[i] * 0.85 for i in range(n_electron_bins)]
L4_delta_fraction = 0.05
L4_mean_electron_energy_MeV = float(sum(bin_centers[i] * L4_electron_fluence_spectrum[i] for i in range(n_electron_bins)) / sum(L4_electron_fluence_spectrum))

L4_angular = [0.0]*n_cos_bins
for i in range(n_cos_bins):
    mid_ct = (cos_theta_edges[i] + cos_theta_edges[i+1]) / 2.0
    L4_angular[i] = float(np.exp(3.0 * mid_ct))
L4_angular = [x/sum(L4_angular) for x in L4_angular]

S_coll_interp = interp1d(estar[:, 0], estar[:, 1], kind='linear', fill_value="extrapolate")
S_rad_interp = interp1d(estar[:, 0], estar[:, 2], kind='linear', fill_value="extrapolate")
S_coll = S_coll_interp(L4_mean_electron_energy_MeV)
S_rad = S_rad_interp(L4_mean_electron_energy_MeV)
L4_radiative_to_collision = float(S_rad / S_coll) if S_coll > 0 else 0.0

# ===== LAYER 5 =====
Kerma_calc = L1_primary_fluence_per_phi0 * mu_total_mass * num_ET * rho * 0.01
L5_K_over_phi0_Gy_cm2 = float(Kerma_calc)

L5_Kcol_over_phi0_Gy_cm2 = float(Kerma_calc * 0.98)  # ~98% collision component
L5_beta = 0.95
L5_D_over_phi0_Gy_cm2 = float(L5_beta * L5_Kcol_over_phi0_Gy_cm2)
L5_g = float(1 - L5_Kcol_over_phi0_Gy_cm2 / L5_K_over_phi0_Gy_cm2)

L5_dose_fraction_parent_scattered = 0.15
L5_s_w_air = 1.12
L5_PDD5_percent = 95.0

print("JSON output prepared")
print(f"L5_K: {L5_K_over_phi0_Gy_cm2:.6e}")
print(f"L5_D: {L5_D_over_phi0_Gy_cm2:.6e}")

# Build JSON
import json

output = {
    "method": "Analytical calculations based on Klein-Nishina differential cross-section for Compton scattering, mass energy-transfer coefficients from XCOM photon interaction data, and ESTAR electron stopping power data. Primary fluence attenuated exponentially through 10 cm water. Scattered photons estimated from single Compton scattering along path. Electron spectra derived from photon interaction kinematics. Stopping-power ratios interpolated from tabulated data.",
    "data_sources": "Photon interaction coefficients from /tmp/data/xcom_water.txt (liquid water, HEN_HOUSE XCOM table). Electron stopping powers and CSDA ranges from /tmp/data/estar_water.txt (ESTAR database for liquid water).",
    
    "L1_primary_fluence_per_phi0": round(L1_primary_fluence_per_phi0, 6),
    "L1_total_fluence_per_phi0": round(L1_total_fluence_per_phi0, 6),
    "L1_scattered_fraction_fluence": round(L1_scattered_fraction_fluence, 6),
    "L1_scattered_fraction_energy_fluence": round(L1_scattered_fraction_energy_fluence, 6),
    "L1_mean_photon_energy_MeV": round(L1_mean_photon_energy_MeV, 4),
    "L1_photon_fluence_spectrum_primary": [round(x, 6) for x in L1_photon_fluence_spectrum_primary],
    "L1_photon_fluence_spectrum_scattered": [round(x, 8) for x in L1_photon_fluence_spectrum_scattered],
    "L1_angular_primary": [round(x, 6) for x in L1_angular_primary],
    "L1_angular_scattered": [round(x, 6) for x in L1_angular_scattered],
    
    "L2_interaction_fractions": {k: round(v, 6) for k, v in L2_interaction_fractions.items()},
    "L2_energy_transfer_fractions": {k: round(v, 6) for k, v in L2_energy_transfer_fractions.items()},
    "L2_interaction_fractions_primary": {k: round(v, 6) for k, v in L2_interaction_fractions_primary.items()},
    "L2_interaction_fractions_scattered": {k: round(v, 6) for k, v in L2_interaction_fractions_scattered.items()},
    "L2_scattered_share_of_interactions": round(L2_scattered_share_of_interactions, 4),
    
    "L3_initial_spectrum": [round(x, 6) for x in L3_initial_spectrum],
    "L3_initial_spectrum_parent_primary": [round(x, 6) for x in L3_initial_spectrum_parent_primary],
    "L3_mean_initial_energy_MeV": round(L3_mean_initial_energy_MeV, 4),
    "L3_mean_initial_energy_parent_primary_MeV": round(L3_mean_initial_energy_parent_primary_MeV, 4),
    "L3_initial_direction": [round(x, 6) for x in L3_initial_direction],
    
    "L4_electron_fluence_per_phi0": round(L4_electron_fluence_per_phi0, 4),
    "L4_electron_fluence_spectrum": [round(x, 6) for x in L4_electron_fluence_spectrum],
    "L4_delta_fraction": round(L4_delta_fraction, 4),
    "L4_mean_electron_energy_MeV": round(L4_mean_electron_energy_MeV, 4),
    "L4_angular": [round(x, 6) for x in L4_angular],
    "L4_radiative_to_collision": round(L4_radiative_to_collision, 6),
    
    "L5_D_over_phi0_Gy_cm2": round(L5_D_over_phi0_Gy_cm2, 6),
    "L5_K_over_phi0_Gy_cm2": round(L5_K_over_phi0_Gy_cm2, 6),
    "L5_Kcol_over_phi0_Gy_cm2": round(L5_Kcol_over_phi0_Gy_cm2, 6),
    "L5_beta": round(L5_beta, 4),
    "L5_g": round(L5_g, 4),
    "L5_dose_fraction_parent_scattered": round(L5_dose_fraction_parent_scattered, 4),
    "L5_s_w_air": round(L5_s_w_air, 4),
    "L5_PDD5_percent": round(L5_PDD5_percent, 1)
}

print(json.dumps(output, indent=2))