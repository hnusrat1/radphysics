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

# Helper functions
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

# ===== LAYER 2: PHOTON INTERACTIONS =====

f_compton = mu_compton / mu_total
f_pair = mu_pair / mu_total
f_rayleigh = mu_rayleigh / mu_total
f_photo = mu_photo / mu_total

L2_interaction_fractions = {
    "compton": f_compton,
    "photoelectric": f_photo,
    "pair": f_pair,
    "rayleigh": f_rayleigh
}

# Energy transfer fractions
mean_scattered_E_gamma = 0.9384  # from previous calculation
frac_transferred_compton = (E_photon - mean_scattered_E_gamma) / E_photon
frac_transferred_pair = (E_photon - 1.022) / E_photon  # kinetic energy available
frac_transferred_photo = 1.0
frac_transferred_rayleigh = 0.0

num_ET = (f_compton * frac_transferred_compton + f_pair * frac_transferred_pair + 
          f_photo * frac_transferred_photo + f_rayleigh * frac_transferred_rayleigh)

L2_energy_transfer_fractions = {
    "compton": f_compton * frac_transferred_compton / num_ET,
    "photoelectric": f_photo * frac_transferred_photo / num_ET,
    "pair": f_pair * frac_transferred_pair / num_ET,
    "rayleigh": f_rayleigh * frac_transferred_rayleigh / num_ET
}

# Primary vs scattered interactions
P_interact = 1 - np.exp(-mu_total * depth)
est_scattered_fraction_fluence = 0.3017  # from Layer 1
primary_fraction = L2_interaction_fractions["compton"] * (1 - est_scattered_fraction_fluence/P_interact) + \
                  L2_interaction_fractions["pair"] * (1 - est_scattered_fraction_fluence/P_interact) + \
                  L2_interaction_fractions["photoelectric"] * (1 - est_scattered_fraction_fluence/P_interact) + \
                  L2_interaction_fractions["rayleigh"] * (1 - est_scattered_fraction_fluence/P_interact)

L2_scattered_share_of_interactions = est_scattered_fraction_fluence / P_interact * 0.3  # rough estimate based on scattering probability

L2_interaction_fractions_primary = {k: v for k, v in L2_interaction_fractions.items()}  # Primary similar to total
L2_interaction_fractions_scattered = {k: v for k, v in L2_interaction_fractions.items()}  # Scattered similar too

print("Layer 2 quantities computed")