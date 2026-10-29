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

# Primary fluence
L1_primary_fluence_per_phi0 = np.exp(-mu_total * depth)

# Interaction fractions at the point (based on cross-section ratios)
f_compton = mu_compton / mu_total
f_pair = mu_pair / mu_total
f_rayleigh = mu_rayleigh / mu_total
f_photo = mu_photo / mu_total

print("Layer 2: Photon interactions at the point")
print("="*50)
print(f"L2_interaction_fractions:")
print(f"  compton: {f_compton:.6f}")
print(f"  photoelectric: {f_photo:.6e}")
print(f"  pair: {f_pair:.6f}")
print(f"  rayleigh: {f_rayleigh:.6e}")

# Energy transfer fractions
# For Compton: average fraction of photon energy transferred to electron
alpha = E_photon / 0.511

def klein_nishina_intensity(alpha, ct):
    k = 1.0 + alpha * (1.0 - ct)
    sin_sq = 1.0 - ct**2
    return (1.0/k**2) * (k + 1.0/k - sin_sq)

def scattered_energy(alpha, ct):
    k = 1.0 + alpha * (1.0 - ct)
    return E_photon / k

# Integrate over cos(theta) from -1 to 1
num_int, _ = quad(lambda ct: scattered_energy(alpha, ct) * klein_nishina_intensity(alpha, ct), -1, 1)
denom_int, _ = quad(lambda ct: klein_nishina_intensity(alpha, ct), -1, 1)
mean_scattered_E = num_int / denom_int
fraction_energy_transferred_compton = (E_photon - mean_scattered_E) / E_photon

# For Rayleigh: zero energy transfer (coherent scattering)
fraction_energy_transferred_rayleigh = 0.0

# For photoelectric: essentially 100% energy transfer (negligible)
fraction_energy_transferred_photo = 1.0

# For pair production: energy available for kinetic energy = E - 2*m_e*c^2
E_avail_pair = E_photon - 2 * 0.511  # subtract rest mass of e+ and e-
fraction_energy_transferred_pair = E_avail_pair / E_photon if E_avail_pair > 0 else 0

# Weighted energy transfer fractions
num_energy_transfer = (f_compton * fraction_energy_transferred_compton + 
                       f_rayleigh * fraction_energy_transferred_rayleigh +
                       f_photo * fraction_energy_transferred_photo +
                       f_pair * fraction_energy_transferred_pair)
total_interactions_weight = f_compton + f_rayleigh + f_photo + f_pair  # should be ~1

print(f"\nfraction_energy_transferred_compton: {fraction_energy_transferred_compton:.6f}")
print(f"fraction_energy_transferred_pair: {fraction_energy_transferred_pair:.6f}")

L2_energy_transfer_fractions = {
    "compton": f_compton * fraction_energy_transferred_compton / num_energy_transfer,
    "photoelectric": f_photo * fraction_energy_transferred_photo / num_energy_transfer,
    "pair": f_pair * fraction_energy_transferred_pair / num_energy_transfer,
    "rayleigh": f_rayleigh * fraction_energy_transferred_rayleigh / num_energy_transfer
}

print(f"L2_energy_transfer_fractions:")
for key, val in L2_energy_transfer_fractions.items():
    print(f"  {key}: {val:.6f}")

# For primary vs scattered interactions
# Since most interactions are from primary photons (only ~19% scatter), 
# L2_interaction_fractions_primary ≈ L2_interaction_fractions
L2_interaction_fractions_primary = {
    "compton": f_compton,
    "photoelectric": f_photo,
    "pair": f_pair,
    "rayleigh": f_rayleigh
}

# Scattered share of interactions
# Based on earlier estimate, scattered fraction of fluence ~0.19/0.80
est_scattered_fraction = 0.24  # rough estimate from earlier calculation
P_interact = 1 - np.exp(-mu_total * depth)
L2_scattered_share_of_interactions = est_scattered_fraction * P_interact / ((1 - est_scattered_fraction) + est_scattered_fraction * P_interact)
print(f"\nL2_scattered_share_of_interactions: {L2_scattered_share_of_interactions:.4f}")

L2_interaction_fractions_scattered = {
    "compton": f_compton,  # same process fractions for scattered too
    "photoelectric": f_photo,
    "pair": f_pair,
    "rayleigh": f_rayleigh
}

print("\nLayer 2 complete.")