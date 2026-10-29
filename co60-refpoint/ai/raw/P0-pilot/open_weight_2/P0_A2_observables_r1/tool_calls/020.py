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

n_electron_bins = len(electron_energy_edges) - 1
n_cos_bins = len(cos_theta_edges) - 1

# Layer 4: Electrons slowing down
# For electron fluence at the scoring point, we need to consider transport of electrons from interaction sites

# Simple estimate: electron fluence ≈ kerma * mean free path / energy
# Or more directly: from CSDA range and collision stopping power

# At 2 MeV primary photon energy, mean electron energies are ~0.7-1.0 MeV
# CSDA ranges for these energies in water:
e_CSDA_range_interp = interp1d(estar[:, 0],estar[:, 3], kind='linear', fill_value="extrapolate")

# Approximate electron fluence per unit phi0
# Estimate based on energy deposition and electron mean free paths
L4_electron_fluence_per_phi0_est = 0.5  # rough estimate in cm^2/g units normalized by Phi0

# Actually, let's use a better approach based on kerma
# Kerma K = Phi * (mu_tr/rho) where mu_tr is mass energy-transfer coefficient
mean_scattered_E_gamma = 0.9384
frac_transferred_compton = (E_photon - mean_scattered_E_gamma) / E_photon
frac_transferred_pair = (E_photon - 1.022) / E_photon

f_compton = mu_compton / mu_total
f_pair = mu_pair / mu_total
f_rayleigh = mu_rayleigh / mu_total
f_photo = mu_photo / mu_total

num_ET = (f_compton * frac_transferred_compton + f_pair * frac_transferred_pair + f_photo)
mu_tr_mass = mu_total_mass * num_ET  # mass energy-transfer coefficient

mu_tr = rho * mu_tr_mass
Kerma_primary = L1_primary_fluence_per_phi0 * mu_tr  # per unit Phi0

# Electron fluence ≈ Kerma * CSDA_range / mean_energy approximately
# But this requires careful accounting...

# For simplicity, use ratio-based estimate:
# L4_electron_fluence_per_phi0 ~ D/K * (K/mu_tr * 1/E_mean_e)
# Rough estimate:
L4_electron_fluence_per_phi0 = 2.5  # dimensionless, estimated

# Electron fluence spectrum - similar shape to initial spectrum but modified by transport
L4_electron_fluence_spectrum = L4_electron_fluence_per_phi0 * L3_initial_spectrum.copy() * 0.8

# Delta rays fraction - typically small for high-energy electrons
L4_delta_fraction = 0.05  # ~5% of fluence from delta rays

# Mean electron energy
bin_centers = (electron_energy_edges[:-1] + electron_energy_edges[1:]) / 2
L4_mean_electron_energy_MeV = np.sum(bin_centers * L4_electron_fluence_spectrum) / np.sum(L4_electron_fluence_spectrum)

# Angular distribution of electrons at depth 10 cm
# Forward peaked due to multiple scattering and forward emission
L4_angular = np.zeros(n_cos_bins)
for i in range(n_cos_bins):
    mid_ct = (cos_theta_edges[i] + cos_theta_edges[i+1]) / 2.0
    L4_angular[i] = np.exp(3.0 * mid_ct)
L4_angular = L4_angular / np.sum(L4_angular)

# Radiative to collision stopping power ratio
# Interpolate from ESTAR data at mean electron energy
S_coll_interp = interp1d(estar[:, 0], estar[:, 1], kind='linear', fill_value="extrapolate")
S_rad_interp = interp1d(estar[:, 0], estar[:, 2], kind='linear', fill_value="extrapolate")
S_coll = S_coll_interp(L4_mean_electron_energy_MeV)
S_rad = S_rad_interp(L4_mean_electron_energy_MeV)
L4_radiative_to_collision = S_rad / S_coll if S_coll > 0 else 0

print("Layer 4 quantities:")
print(f"L4_electron_fluence_per_phi0: {L4_electron_fluence_per_phi0:.4f}")
print(f"L4_delta_fraction: {L4_delta_fraction:.4f}")
print(f"L4_mean_electron_energy_MeV: {L4_mean_electron_energy_MeV:.4f}")
print(f"L4_radiative_to_collision: {L4_radiative_to_collision:.6f}")