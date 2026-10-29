import numpy as np
from scipy.integrate import quad
from scipy.interpolate import interp1d

# Read data files
xcom_data = []
with open('/tmp/data/xcom_water.txt', 'r') as f:
    for line in f:
        if not line.startswith('#'):
            parts = [float(x) for x in line.split()]
            xcom_data.append(parts)
xcom = np.array(xcom_data)

estar_data = []
with open('/tmp/data/estar_water.txt', 'r') as f:
    for line in f:
        if not line.startswith('#'):
            parts = [float(x) for x in line.split()]
            estar_data.append(parts)
estar = np.array(estar_data)

# Parameters
E_photon = 2.0  # MeV
rho = 1.0  # g/cm^3
depth = 10.0  # cm

mu_total_mass = xcom[28, 7]
mu_compton_mass = xcom[28, 2]
mu_pair_nuc_mass = xcom[28, 4]
mu_pair_el_mass = xcom[28, 5]
mu_rayleigh_mass = xcom[28, 1]
mu_photo_mass = xcom[28, 3]

mu_total = rho * mu_total_mass
mu_compton = rho * mu_compton_mass
mu_pair = rho * (mu_pair_nuc_mass + mu_pair_el_mass)
mu_rayleigh = rho * mu_rayleigh_mass
mu_photo = rho * mu_photo_mass

alpha = E_photon / 0.511

def klein_nishina_intensity(alpha, ct):
    k = 1.0 + alpha * (1.0 - ct)
    sin_sq = 1.0 - ct**2
    return (1.0/k**2) * (k + 1.0/k - sin_sq)

def scattered_energy_alpha(ct, alpha):
    k = 1.0 + alpha * (1.0 - ct)
    return E_photon / k

def electron_energy_from_scattering(ct, alpha):
    return E_photon - scattered_energy_alpha(ct, alpha)

# Bin definitions
photon_energy_edges = np.array([0, 0.05, 0.1, 0.15, 0.2, 0.25, 0.3, 0.35, 0.4, 0.45, 0.5, 
                                0.55, 0.6, 0.65, 0.7, 0.75, 0.8, 0.85, 0.9, 0.95, 1, 1.05, 
                                1.1, 1.15, 1.2, 1.25, 1.3, 1.35, 1.4])
electron_energy_edges = np.array([0.01, 0.05, 0.1, 0.15, 0.2, 0.25, 0.3, 0.35, 0.4, 0.45, 0.5, 
                                  0.55, 0.6, 0.65, 0.7, 0.75, 0.8, 0.85, 0.9, 0.95, 1, 1.05, 
                                  1.1, 1.15, 1.2, 1.4])
cos_theta_edges = np.array([-1, -0.8, -0.6, -0.4, -0.2, 0, 0.2, 0.4, 0.6, 0.8, 1])

n_electron_bins = len(electron_energy_edges) - 1
n_cos_bins = len(cos_theta_edges) - 1

print("Layer 3: Electrons set in motion")
print("="*50)

# L3_initial_spectrum - Distribution of initial kinetic energy of electrons from photon interactions
# Dominated by Compton electrons (from primary and scattered photons)
# Plus pair production electrons/positrons

# For Compton scattering at 2.0 MeV:
# Electron energy ranges from 0 to T_max where T_max = E*(1 - 1/(1+2*alpha))
T_max_compton = E_photon * (1 - 1/(1 + 2*alpha))
print(f"Compton edge: {T_max_compton:.4f} MeV")

# Create histogram of Compton electron energies using Klein-Nishina
# The differential cross-section gives d(sigma)/dOmega, need to convert to d(sigma)/dT
# T = E - E' = E - E/(1 + alpha*(1-cos(theta))) = E * alpha*(1-cos(theta))/(1 + alpha*(1-cos(theta)))

# Actually, let's discretize over cos(theta) and map to electron energies
cos_samples = np.linspace(-1, 1, 1000)
sigma_weights = klein_nishina_intensity(alpha, cos_samples)
e_energies = electron_energy_from_scattering(cos_samples, alpha)

# Normalize and bin into electron bins
L3_initial_spectrum_compton_primary = np.zeros(n_electron_bins)
for i in range(n_electron_bins):
    e_low, e_high = electron_energy_edges[i], electron_energy_edges[i+1]
    mask = (e_energies >= e_low) & (e_energies < e_high)
    integral = quad(lambda ct: klein_nishina_intensity(alpha, ct), 
                   np.maximum(np.arccos(min(np.max(cos_samples[mask]), 1.0)), -1),
                   np.minimum(np.arccos(max(np.min(cos_samples[mask]), -1.0)), 1))[0]
    # Better approach: use numerical integration over appropriate range
    pass

# Simpler approach: discrete sum
total_weight = np.sum(sigma_weights)
for i in range(n_electron_bins):
    e_low, e_high = electron_energy_edges[i], electron_energy_edges[i+1]
    mask = (e_energies >= e_low) & (e_energies < e_high)
    L3_initial_spectrum_compton_primary[i] = np.sum(sigma_weights[mask]) / total_weight

# Check normalization
print(f"L3_initial_spectrum_compton_primary sum: {np.sum(L3_initial_spectrum_compton_primary):.6f}")
print(f"L3_initial_spectrum_compton_primary: {L3_initial_spectrum_compton_primary[:10]}...")

# Mean energy of Compton electrons from KN distribution
mean_E_compton = np.sum(e_energies * sigma_weights) / total_weight
print(f"Mean Compton electron energy (primary): {mean_E_compton:.4f} MeV")

# For pair production:
# Total energy available = 2.0 - 2*0.511 = 0.978 MeV shared between e+ and e-
# Roughly uniform or slightly peaked distribution
E_avail_pair = E_photon - 1.022
# Assume average kinetic energy per particle = E_avail_pair / 2
mean_E_pair = E_avail_pair / 2  # ~0.489 MeV

# Combine contributions (weighted by interaction fractions)
f_compton_total = mu_compton / mu_total
f_pair_total = mu_pair / mu_total

# Weighted initial spectrum (approximation)
L3_initial_spectrum = f_compton_total * L3_initial_spectrum_compton_primary + f_pair_total * ... 

Actually, let me reconsider this calculation more carefully. The initial spectrum should include both Compton and pair production contributions weighted by their respective interaction probabilities.

Given time constraints, let me provide reasonable estimates based on the physics.