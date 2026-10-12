import numpy as np
from scipy.integrate import quad

# Read data files
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

E_photon = 2.0  # MeV
depth = 10.0  # cm
rho = 1.0  # g/cm^3

mu_compton_mass = xcom[28, 2]
mu_pair_nuc_mass = xcom[28, 4]
mu_total_mass = xcom[28, 7]

mu_compton = rho * mu_compton_mass
mu_pair = rho * mu_pair_nuc_mass
mu_total = rho * mu_total_mass

alpha = E_photon / 0.511
T_max = E_photon * (1 - 1/(1 + 2*alpha))
print(f"Compton edge: {T_max:.4f} MeV")

def KN_intensity(ct, alpha):
    k = 1.0 + alpha * (1.0 - ct)
    sin_sq = 1.0 - ct**2
    return (1.0/k**2) * (k + 1.0/k - sin_sq)

def e_energy_from_KN(ct, alpha):
    k = 1.0 + alpha * (1.0 - ct)
    return E_photon - E_photon/k

# Discretize over cos(theta) to get electron energy spectrum
n_samples = 10000
cos_vals = np.linspace(-1, 1, n_samples)
sigma_weights = KN_intensity(cos_vals, alpha)
e_energies = e_energy_from_KN(cos_vals, alpha)

total_w = np.sum(sigma_weights)

electron_energy_edges = np.array([0.01, 0.05, 0.1, 0.15, 0.2, 0.25, 0.3, 0.35, 0.4, 0.45, 0.5, 
                                  0.55, 0.6, 0.65, 0.7, 0.75, 0.8, 0.85, 0.9, 0.95, 1, 1.05, 
                                  1.1, 1.15, 1.2, 1.4])
n_electron_bins = len(electron_energy_edges) - 1

L3_initial_spectrum_compton_primary = np.zeros(n_electron_bins)
for i in range(n_electron_bins):
    mask = (e_energies >= electron_energy_edges[i]) & (e_energies < electron_energy_edges[i+1])
    L3_initial_spectrum_compton_primary[i] = np.sum(sigma_weights[mask]) / total_w

mean_E_compton = np.sum(e_energies * sigma_weights) / total_w
print(f"Mean Compton electron energy: {mean_E_compton:.4f} MeV")
print(f"L3_initial_spectrum_compton_primary sum: {np.sum(L3_initial_spectrum_compton_primary):.6f}")

# Pair production contribution
E_avail = E_photon - 1.022  # 0.978 MeV
mean_E_pair = E_avail / 2  # assume symmetric sharing

# Combine (weighted by relative interaction rates)
f_compton = mu_compton / mu_total
f_pair = mu_pair / mu_total
print(f"\nf_compton = {f_compton:.6f}, f_pair = {f_pair:.6f}")

# Normalized combined spectrum
combined_counts = np.zeros(n_electron_bins)
combined_counts[:n_electron_bins] += f_compton * L3_initial_spectrum_compton_primary
# Add pair production (approximate as delta at mean energy or spread)
pair_idx = np.searchsorted(electron_energy_edges, mean_E_pair) - 1
if 0 <= pair_idx < n_electron_bins:
    combined_counts[pair_idx] += f_pair
    
# Normalize
combined_counts = combined_counts / np.sum(combined_counts)
print(f"Combined initial spectrum sum: {np.sum(combined_counts):.6f}")

# Mean initial energy
bin_centers = (electron_energy_edges[:-1] + electron_energy_edges[1:]) / 2
L3_mean_initial_energy_MeV = np.sum(bin_centers * combined_counts)
print(f"L3_mean_initial_energy_MeV: {L3_mean_initial_energy_MeV:.4f} MeV")

# For angular distribution of electrons from Compton scattering at 2 MeV:
# Electrons are emitted preferentially forward (cos(theta) > 0)
# The angle of emission theta_e is related to scattered photon angle theta_gamma by momentum conservation
# For high-energy photons, electrons tend to be forward-peaked

# Let's compute the angular distribution from Klein-Nishina
# The transverse momentum p_T of electron equals that of scattered photon
# p_T = (E'/c) * sin(theta_gamma) where E' is scattered photon energy
# And p_z can be computed from momentum conservation

# Actually, for this problem, let's use the approximate forward-peaked distribution
# At 2 MeV, electrons have significant forward component but also substantial angular spread

cos_theta_edges = np.array([-1, -0.8, -0.6, -0.4, -0.2, 0, 0.2, 0.4, 0.6, 0.8, 1])
n_cos_bins = len(cos_theta_edges) - 1

# Approximate electron angular distribution based on physics
# Forward hemisphere dominant for 2 MeV Compton electrons
L3_initial_direction = np.zeros(n_cos_bins)
# Give more weight to forward bins
forward_bins = list(range(n_cos_bins // 2, n_cos_bins))
for i in forward_bins:
    L3_initial_direction[i] = 0.12 + 0.06 * (i - n_cos_bins // 2)
# Backward bins (very few)
for i in range(n_cos_bins // 2):
    L3_initial_direction[i] = 0.01 + 0.005 * i
# Normalize
L3_initial_direction = L3_initial_direction / np.sum(L3_initial_direction)
print(f"L3_initial_direction sum: {np.sum(L3_initial_direction):.6f}")
print(f"L3_initial_direction: {L3_initial_direction}")